/* ============================================================================
 *  main.c  --  Standalone bare-metal entry point for the STM32H7A3ZIT6Q HIL
 *              INT8 inference benchmark image.
 *
 *  Boot path (see startup_stm32h7a3zitxq.s):
 *      Reset_Handler -> ExitRun0Mode() -> SystemInit() -> copy .data / zero .bss
 *                    -> __libc_init_array() -> main()
 *
 *  What main() does:
 *    1. Re-syncs SystemCoreClock from the live RCC registers via
 *       SystemCoreClockUpdate(). We do NOT bring up the 280 MHz PLL: the image
 *       runs at the post-reset default clock (HSI, nominally 64 MHz). Because
 *       the benchmark converts DWT cycles -> microseconds using the runtime
 *       SystemCoreClock, the reported microseconds are correct for whatever
 *       clock is actually active. The benchmark also prints the confirmed core
 *       clock so the human can read it back.
 *    2. Enables the DWT cycle counter (dwt_cycles_init()).
 *    3. Routes bench_print() to ARM semihosting (SYS_WRITE0 via "bkpt 0xAB"),
 *       so output returns through ST-Link/openocd with no UART wiring.
 *       This is guarded: semihosting is only emitted when a debugger is
 *       attached (CoreDebug DHCSR C_DEBUGEN bit set), so the same image does
 *       NOT HardFault if it ever runs on bare silicon with no debugger.
 *    4. Runs ids_bench_run() once, then loops forever.
 *
 *  Build links -specs=nano.specs -specs=rdimon.specs, so newlib-nano's printf /
 *  vsnprintf / qsort and rdimon's semihosted syscalls (_write, _sbrk, _exit, ...)
 *  resolve for free. We additionally override bench_print() here with a direct
 *  SYS_WRITE0 call so benchmark output does not depend on rdimon stdout file
 *  handles or the heap -- it works as long as openocd has semihosting enabled.
 * ========================================================================== */

#include "stm32h7xx.h"   /* CoreDebug, SystemCoreClockUpdate prototype, CMSIS */
#include "bench.h"
#include "dwt_cycles.h"

/* Set in main() once we know whether a debugger is listening for semihosting. */
static volatile int s_semihost_ok = 0;

/* ---------------------------------------------------------------------------
 *  Minimal ARM semihosting SYS_WRITE0 (operation 0x04): writes a NUL-terminated
 *  string to the debugger's console. Triggers a host trap via "bkpt 0xAB".
 *  Safe to call ONLY when a debugger is attached -- otherwise the breakpoint
 *  escalates to a HardFault. We gate it on s_semihost_ok.
 * ------------------------------------------------------------------------- */
static void sh_write0(const char *s)
{
    register int        op __asm__("r0") = 0x04;   /* SYS_WRITE0 */
    register const char *p __asm__("r1") = s;
    __asm__ volatile ("bkpt 0xAB" : "+r"(op) : "r"(p) : "memory", "cc");
    (void)op;
}

/* ---------------------------------------------------------------------------
 *  Strong override of the weak bench_print() in bench.c: tee each line to the
 *  semihosting console (with CRLF). bench.c's bench_printf() formats into a
 *  stack buffer with vsnprintf() and then calls this.
 * ------------------------------------------------------------------------- */
void bench_print(const char *line)
{
    if (!s_semihost_ok) {
        return;
    }
    sh_write0(line);
    sh_write0("\r\n");
}

int main(void)
{
    /* SystemInit() already ran from the startup before main(). Re-sync the
     * cached SystemCoreClock from the live RCC registers so the benchmark's
     * cycles->microseconds conversion is exact for the active clock. */
    SystemCoreClockUpdate();

    /* Enable the DWT CYCCNT cycle counter used for all benchmark timing. */
    dwt_cycles_init();

    /* Emit semihosting only if a debugger (openocd) is attached. C_DEBUGEN is
     * bit 0 of CoreDebug->DHCSR and is set by the debug probe on connect. */
    s_semihost_ok =
        (CoreDebug->DHCSR & CoreDebug_DHCSR_C_DEBUGEN_Msk) ? 1 : 0;

    /* Run the full on-silicon INT8 inference benchmark and print the report. */
    ids_bench_run();

    /* Done: spin forever so the debugger can read the console / halt cleanly. */
    for (;;) {
        __asm__ volatile ("nop");
    }
}
