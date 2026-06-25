/* ============================================================================
 *  test_main.c  --  Minimal entry point for the STAND-ALONE Makefile build.
 *
 *  NOT used in the CubeIDE drop-in route (there you call ids_bench_run() from
 *  the project's own main()). This exists only so the Makefile can produce a
 *  self-contained object/ELF and so `make check` can prove the package compiles
 *  clean for cortex-m7. Semihosting printf is the default output; wire
 *  bench_print() to your UART for the real board run.
 * ========================================================================== */
#include "bench.h"

int main(void)
{
    /* On real hardware the system clock + SystemCoreClockUpdate() run before
     * this; in the bare object-compile path SystemCoreClock falls back to the
     * weak 280 MHz default. */
    ids_bench_run();
    for (;;) { /* spin */ }
    return 0;
}
