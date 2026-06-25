/* ============================================================================
 *  bench.c  --  On-silicon HIL benchmark driver for the INT8 autoencoder IDS.
 *
 *  Reports REAL measured numbers (not analytical CMSIS-NN cycle estimates):
 *  per-inference latency (min/median/max in cycles and µs), feature-extraction
 *  stub latency, activation-arena RAM, model Flash, and projected CPU
 *  utilization at 10/20/100 Hz. All timing comes from the DWT cycle counter;
 *  cycles are converted to µs via the runtime SystemCoreClock.
 * ========================================================================== */
#include "bench.h"
#include "model_int8.h"
#include "dwt_cycles.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>   /* qsort */
#include <stdarg.h>   /* va_list for bench_printf */

/* ---------------------------------------------------------------------------
 *  Weak SystemCoreClock fallback.
 *  In the CubeIDE project, ST's system_stm32h7xx.c defines SystemCoreClock
 *  (strong) and SystemCoreClockUpdate() keeps it current; the linker uses ST's
 *  symbol and ignores this weak one. For the stand-alone Makefile link there is
 *  no ST file, so this provides a sensible default (280 MHz for the H7A3).
 *  At run time on hardware the value is the true core clock, so µs are exact.
 * ------------------------------------------------------------------------- */
__attribute__((weak)) uint32_t SystemCoreClock = 280000000UL;

/* ---------------------------------------------------------------------------
 *  Weak output sink: default to printf (semihosting / retargeted _write).
 *  The user overrides bench_print() to tee into LogStream / UART.
 * ------------------------------------------------------------------------- */
__attribute__((weak)) void bench_print(const char *line)
{
    fputs(line, stdout);
    fputc('\n', stdout);
}

/* Weak hook for a real captured feature vector; default NULL => synthetic. */
__attribute__((weak)) const signed char *bench_get_real_feature(void)
{
    return NULL;
}

/* Small formatting helper around bench_print(). */
static void bench_printf(const char *fmt, ...)
{
    char buf[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    bench_print(buf);
}

/* ===========================================================================
 *  Feature-extraction stub
 *  Represents the cost of assembling the 80-dim feature vector (13 features x
 *  6 CAN IDs + 2) from the latest decoded CAN snapshot and quantizing it to
 *  int8. This is a realistic placeholder: per-feature scale/offset + clamp.
 *  Replace the body with the real extraction to profile it end-to-end; the
 *  benchmark times whatever is here.
 * ========================================================================= */
static int8_t s_feat[IDS_FEAT_DIM];

/* Deterministic synthetic "decoded CAN" source values (float), so the stub
 * does representative float->int8 quantization work. */
static float s_can_src[IDS_FEAT_DIM];

static void feature_source_init(void)
{
    /* Fill with a deterministic but non-trivial pattern. */
    uint32_t st = 0xBEEF1234u;
    for (int i = 0; i < IDS_FEAT_DIM; i++) {
        st = st * 1664525u + 1013904223u;
        /* map to roughly [-4, 4] engineering units */
        s_can_src[i] = ((float)((st >> 8) & 0xFFFF) / 8192.0f) - 4.0f;
    }
}

/* Quantize one engineering value to int8 with a per-tensor scale (representative
 * scale = 1/0.0625 = 16 lsb/unit), symmetric, clamped to [-127,127]. */
static inline int8_t quantize_feature(float x)
{
    int32_t q = (int32_t)(x * 16.0f + (x >= 0.0f ? 0.5f : -0.5f));
    if (q > 127)  q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

/* Build s_feat[] from s_can_src[]. Returns nothing; result in s_feat. */
static void feature_extract(void)
{
    for (int i = 0; i < IDS_FEAT_DIM; i++) {
        s_feat[i] = quantize_feature(s_can_src[i]);
    }
}

/* ===========================================================================
 *  Timing storage
 * ========================================================================= */
static uint32_t s_samples[IDS_BENCH_ITERS];

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

/* ===========================================================================
 *  Benchmark
 * ========================================================================= */
void ids_bench_run(void)
{
    dwt_cycles_init();
    feature_source_init();

    /* If the user supplied a real captured feature vector, copy it in;
     * otherwise extract from the synthetic source. */
    const signed char *real = bench_get_real_feature();
    if (real != NULL) {
        for (int i = 0; i < IDS_FEAT_DIM; i++) {
            s_feat[i] = (int8_t)real[i];
        }
    } else {
        feature_extract();
    }

    /* ---- Time the feature-extraction stub (single, representative call). ---- */
    dwt_cycles_reset();
    feature_extract();
    uint32_t feat_cycles = dwt_cycles_read();

    /* ---- Warm-up (prime I-cache / branch predictor) so the timed loop is
     *      steady-state, matching real continuous operation. ---- */
    int32_t err;
    volatile int sink = 0;
    for (int i = 0; i < 16; i++) {
        sink += ids_infer_int8(s_feat, &err);
    }

    /* ---- Timed loop: N inferences, one DWT measurement each. ---- */
    uint32_t min_c = 0xFFFFFFFFu, max_c = 0;
    uint64_t sum_c = 0;
    for (int i = 0; i < IDS_BENCH_ITERS; i++) {
        dwt_cycles_reset();
        sink += ids_infer_int8(s_feat, &err);
        uint32_t c = dwt_cycles_read();
        s_samples[i] = c;
        if (c < min_c) min_c = c;
        if (c > max_c) max_c = c;
        sum_c += c;
    }
    (void)sink;

    /* Median via sort. */
    qsort(s_samples, IDS_BENCH_ITERS, sizeof(uint32_t), cmp_u32);
    uint32_t med_c = s_samples[IDS_BENCH_ITERS / 2];
    uint32_t mean_c = (uint32_t)(sum_c / IDS_BENCH_ITERS);

    /* Convert to microseconds via runtime core clock. */
    float min_us  = dwt_cycles_to_us_f(min_c);
    float med_us  = dwt_cycles_to_us_f(med_c);
    float max_us  = dwt_cycles_to_us_f(max_c);
    float mean_us = dwt_cycles_to_us_f(mean_c);
    float feat_us = dwt_cycles_to_us_f(feat_cycles);

    /* Footprint (measured via sizeof of the real static arrays). */
    size_t flash_b = ids_weights_flash_bytes();
    size_t ram_b   = ids_arena_ram_bytes();
    uint32_t macs  = ids_total_macs();

    /* Projected CPU utilization at fixed detection rates:
     *   util% = latency_us * rate_hz / 1e6 * 100 = latency_us * rate / 10000. */
    float u10  = med_us * 10.0f  / 10000.0f;
    float u20  = med_us * 20.0f  / 10000.0f;
    float u100 = med_us * 100.0f / 10000.0f;

    /* Integers for clk in MHz (avoid float for the headline clock line). */
    uint32_t clk_mhz = SystemCoreClock / 1000000UL;

    /* ---- Report ---- */
    bench_print("==== IDS INT8 Autoencoder HIL Benchmark ====");
    bench_printf("core_clock      : %lu Hz (%lu MHz)",
                 (unsigned long)SystemCoreClock, (unsigned long)clk_mhz);
#ifdef USE_CMSIS_NN
    bench_print ("kernel          : CMSIS-NN arm_fully_connected_s8");
#else
    bench_print ("kernel          : self-contained int8 FC (== arm_fully_connected_s8)");
#endif
    bench_printf("network         : 80-40-20-10-20-40-80, ReLU hidden, linear out");
    bench_printf("MACs/inference  : %lu", (unsigned long)macs);
    bench_printf("iterations      : %d", IDS_BENCH_ITERS);
    bench_print ("--------------------------------------------");
    bench_printf("latency min     : %lu cyc  = %ld.%02lu us",
                 (unsigned long)min_c, (long)min_us,
                 (unsigned long)((min_us - (long)min_us) * 100));
    bench_printf("latency median  : %lu cyc  = %ld.%02lu us",
                 (unsigned long)med_c, (long)med_us,
                 (unsigned long)((med_us - (long)med_us) * 100));
    bench_printf("latency mean    : %lu cyc  = %ld.%02lu us",
                 (unsigned long)mean_c, (long)mean_us,
                 (unsigned long)((mean_us - (long)mean_us) * 100));
    bench_printf("latency max     : %lu cyc  = %ld.%02lu us",
                 (unsigned long)max_c, (long)max_us,
                 (unsigned long)((max_us - (long)max_us) * 100));
    bench_printf("feature-extract : %lu cyc  = %ld.%02lu us",
                 (unsigned long)feat_cycles, (long)feat_us,
                 (unsigned long)((feat_us - (long)feat_us) * 100));
    bench_print ("--------------------------------------------");
    bench_printf("activation RAM  : %lu bytes", (unsigned long)ram_b);
    bench_printf("model Flash     : %lu bytes", (unsigned long)flash_b);
    bench_print ("--------------------------------------------");
    bench_printf("CPU util @ 10Hz : %ld.%03lu %%",
                 (long)u10, (unsigned long)((u10 - (long)u10) * 1000));
    bench_printf("CPU util @ 20Hz : %ld.%03lu %%",
                 (long)u20, (unsigned long)((u20 - (long)u20) * 1000));
    bench_printf("CPU util @100Hz : %ld.%03lu %%",
                 (long)u100, (unsigned long)((u100 - (long)u100) * 1000));
    bench_print ("--------------------------------------------");
    /* Show one functional result to prove the pipeline ran end-to-end. */
    int flag = ids_infer_int8(s_feat, &err);
    bench_printf("sample SSE      : %ld  (threshold %d) -> %s",
                 (long)err, IDS_SSE_THRESHOLD, flag ? "ANOMALY" : "normal");
    bench_print ("==== end ====");
}
