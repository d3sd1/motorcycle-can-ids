/* ============================================================================
 *  bench.h  --  On-silicon HIL benchmark for the INT8 autoencoder IDS.
 * ========================================================================== */
#ifndef BENCH_H
#define BENCH_H

#ifdef __cplusplus
extern "C" {
#endif

/* Number of timed inferences per benchmark run. */
#ifndef IDS_BENCH_ITERS
#define IDS_BENCH_ITERS 1000
#endif

/**
 * @brief Run the full HIL benchmark and print the report over UART/semihosting.
 *
 * Measures, on real silicon:
 *   - per-inference latency (min / median / max, in cycles and microseconds),
 *   - feature-extraction stub latency,
 *   - static RAM used by the activation arena,
 *   - Flash used by the model weights,
 *   - projected CPU utilization at 10 / 20 / 100 Hz detection rates.
 *
 * cycles->µs conversion uses the runtime SystemCoreClock, so the reported
 * microseconds are correct on any Cortex-M7 (e.g. H743 @ 480 MHz or
 * H7A3 @ 280 MHz) with no code change.
 *
 * Call once from main() after the system clock is configured.
 */
void ids_bench_run(void);

/**
 * @brief Output sink for one line of benchmark text (NUL-terminated, no newline).
 *
 * Weak default writes to stdout via printf (semihosting / retargeted _write).
 * Override it to route the line to your UART / LogStream, e.g.:
 *     void bench_print(const char *s) { LogStream_Push(LOG_LVL_INFO, s); }
 */
void bench_print(const char *line);

/**
 * @brief Optional hook: supply a real captured 80-int8 CAN feature vector.
 *
 * Weak default returns NULL, which makes the benchmark use a deterministic
 * synthetic input. Override to return a pointer to 80 int8 values to profile
 * on a real captured frame (timing is input-value-independent, so this only
 * affects the reported anomaly score, not the latency).
 */
const signed char *bench_get_real_feature(void);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_H */
