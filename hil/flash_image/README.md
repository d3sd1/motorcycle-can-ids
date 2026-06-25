# Flashable STM32H7A3ZIT6Q HIL image — INT8 IDS inference benchmark

Standalone, self-contained bare-metal firmware that runs the INT8 autoencoder
intrusion-detection benchmark (`ids_bench_run()`) on real silicon and prints the
measured numbers back through ARM **semihosting** (ST-Link/openocd) — no UART
wiring required.

## What this is

`ids_bench.elf` / `.bin` / `.hex` flash into the STM32H7A3ZIT6Q internal FLASH
(`0x08000000`). On boot it:

1. Runs the ST startup (`ExitRun0Mode` → `SystemInit` → copy `.data` / zero
   `.bss` → `__libc_init_array` → `main`).
2. In `main()` calls `SystemCoreClockUpdate()` to read the **actual** core clock,
   enables the DWT cycle counter, then runs `ids_bench_run()` once and spins.
3. Each report line is emitted via semihosting `SYS_WRITE0` (`bkpt 0xAB`) and is
   captured by openocd/gdb.

## Core clock

The image deliberately does **NOT** bring up the 280 MHz PLL. It runs at the
**post-reset default clock**: the internal HSI oscillator. `SystemInit()` selects
HSI as SYSCLK; on the STM32H7A3 this is **nominally 64 MHz** at reset.

This is correct and intentional:

- The benchmark converts DWT cycles → microseconds using the **runtime**
  `SystemCoreClock` value (re-synced by `SystemCoreClockUpdate()` from the live
  RCC registers), so the reported microseconds are exact for whatever clock is
  actually active — no code change needed if you later add a PLL.
- The report prints `core_clock : <Hz> (<MHz>)` so the human can confirm the
  real clock the numbers were measured at. **Read that line back.**

If you want the 280 MHz numbers, add PLL configuration before `ids_bench_run()`;
the timing math will follow `SystemCoreClock` automatically.

## Files

| File | Origin |
|------|--------|
| `main.c` | NEW — standalone entry, semihosting `bench_print`, clock/DWT setup |
| `bench.c`, `bench.h` | copied from `../` (benchmark driver, `ids_bench_run`) |
| `model_int8.c/.h`, `model_weights.h` | copied from `../` (INT8 inference + weights) |
| `dwt_cycles.h` | copied from `../` (DWT cycle counter helper) |
| `startup_stm32h7a3zitxq.s` | copied from motloud-v3 datalogger |
| `system_stm32h7xx.c` | copied from motloud-v3 datalogger (`SystemInit`, `SystemCoreClockUpdate`, `ExitRun0Mode`) |
| `STM32H7A3ZITXQ_FLASH.ld` | copied from motloud-v3 datalogger, `/DISCARD/` of libc/libm/libgcc removed |
| `cmsis/Include/*` | copied CMSIS Cortex-M7 core headers |
| `cmsis/Device/Include/*` | copied CMSIS STM32H7 device headers (`stm32h7a3xxq.h`, …) |

> Device define is `-DSTM32H7A3xxQ` (the SMPS "Q" variant matching the ZIT6**Q**
> part). ST ships only the Q device header in this CMSIS pack; the plain
> `stm32h7a3xx.h` is absent. Same Cortex-M7 core either way.

## Build

```bash
make            # builds ids_bench.elf + .bin + .hex + .map, prints size
make size       # re-print section sizes
make clean
```

Toolchain: `arm-none-eabi-gcc` 15.2 at
`/c/Program Files/Arm/GNU Toolchain mingw-w64-x86_64-arm-none-eabi/bin/`.

Compile/link flags:
`-mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -O2 -std=c11 -Wall -Wextra`
linked with `-T STM32H7A3ZITXQ_FLASH.ld -specs=nano.specs -specs=rdimon.specs
-Wl,--gc-sections`.

## Flash + read the numbers

See [`FLASH_AND_READ.md`](FLASH_AND_READ.md) for the exact openocd + gdb sequence
(ST-Link V3, backup-first, semihosting capture) for a Mac.

## Semihosting safety

`main()` only emits the `bkpt 0xAB` semihosting traps when a debugger is attached
(checks `CoreDebug->DHCSR` C_DEBUGEN). Under openocd with `arm semihosting enable`
you get the full report; on bare silicon with no debugger the image simply runs
and spins without faulting.
