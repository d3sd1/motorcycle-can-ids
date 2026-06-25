/* ============================================================================
 *  dwt_cycles.h  --  Cortex-M7 cycle-accurate timing via the DWT CYCCNT.
 *
 *  Board-agnostic: cycles -> microseconds conversion uses the runtime value of
 *  SystemCoreClock, so it is correct on ANY Cortex-M7 regardless of the PLL
 *  configuration (e.g. STM32H743 @ 480 MHz or STM32H7A3 @ 280 MHz).
 *
 *  Self-contained: the DWT / CoreDebug registers are at architecturally fixed
 *  addresses on every ARMv7-M core, so this header does NOT depend on the ST
 *  device headers and compiles stand-alone.  When the CMSIS core header
 *  (core_cm7.h) is already included in the translation unit it provides the
 *  same DWT/CoreDebug definitions; we therefore guard our raw definitions so
 *  there is no redefinition clash.
 * ========================================================================== */
#ifndef DWT_CYCLES_H
#define DWT_CYCLES_H

#include <stdint.h>

/* SystemCoreClock is defined (strongly) by ST's system_stm32h7xx.c inside the
 * CubeIDE project and is kept up to date by SystemCoreClockUpdate().  For the
 * stand-alone Makefile build we provide a weak fallback (see dwt_cycles is
 * header-only, so the weak symbol lives in bench.c). */
extern uint32_t SystemCoreClock;

/* ---- Architectural register addresses (ARMv7-M, valid on all Cortex-M7) ---- */
#ifndef CoreDebug_DEMCR
#define DWT_CYCLES_CoreDebug_DEMCR  (*(volatile uint32_t *)0xE000EDFCUL)
#define DWT_CYCLES_DWT_CTRL         (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCLES_DWT_CYCCNT       (*(volatile uint32_t *)0xE0001004UL)
#define DWT_CYCLES_DWT_LAR          (*(volatile uint32_t *)0xE0001FB0UL)
#define DWT_CYCLES_DEMCR_TRCENA     (1UL << 24)
#define DWT_CYCLES_CTRL_CYCCNTENA   (1UL << 0)
#else
/* CMSIS core_cm7.h is present: reuse its register structs. */
#define DWT_CYCLES_CoreDebug_DEMCR  (CoreDebug->DEMCR)
#define DWT_CYCLES_DWT_CTRL         (DWT->CTRL)
#define DWT_CYCLES_DWT_CYCCNT       (DWT->CYCCNT)
#define DWT_CYCLES_DWT_LAR          (*(volatile uint32_t *)0xE0001FB0UL)
#define DWT_CYCLES_DEMCR_TRCENA     (1UL << 24)
#define DWT_CYCLES_CTRL_CYCCNTENA   (1UL << 0)
#endif

/* DWT software-unlock key.  Required on some Cortex-M7 implementations before
 * the cycle counter can be written; harmless where the lock is not present. */
#define DWT_CYCLES_LAR_KEY  0xC5ACCE55UL

/** Enable the trace subsystem and the DWT cycle counter. Call once at start-up. */
static inline void dwt_cycles_init(void)
{
    DWT_CYCLES_CoreDebug_DEMCR |= DWT_CYCLES_DEMCR_TRCENA;   /* enable trace */
    DWT_CYCLES_DWT_LAR          = DWT_CYCLES_LAR_KEY;        /* unlock DWT (M7) */
    DWT_CYCLES_DWT_CYCCNT       = 0U;                        /* reset counter  */
    DWT_CYCLES_DWT_CTRL        |= DWT_CYCLES_CTRL_CYCCNTENA; /* start counting */
}

/** Reset the cycle counter to zero (call immediately before a timed region). */
static inline void dwt_cycles_reset(void)
{
    DWT_CYCLES_DWT_CYCCNT = 0U;
}

/** Read the current 32-bit cycle count. Wraps every 2^32 cycles (~15 s @ 280 MHz). */
static inline uint32_t dwt_cycles_read(void)
{
    return DWT_CYCLES_DWT_CYCCNT;
}

/** Convert a cycle count to microseconds using the runtime core clock.
 *  Uses 64-bit intermediate maths so there is no overflow and no float on the
 *  hot path of the conversion itself. */
static inline uint32_t dwt_cycles_to_us(uint32_t cycles)
{
    uint32_t hz = SystemCoreClock ? SystemCoreClock : 280000000UL;
    return (uint32_t)(((uint64_t)cycles * 1000000ULL) / hz);
}

/** Same as above but returns microseconds as a float (for sub-µs resolution). */
static inline float dwt_cycles_to_us_f(uint32_t cycles)
{
    uint32_t hz = SystemCoreClock ? SystemCoreClock : 280000000UL;
    return ((float)cycles * 1000000.0f) / (float)hz;
}

#endif /* DWT_CYCLES_H */
