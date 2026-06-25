# BUILD SUMMARY — flashable STM32H7A3ZIT6Q HIL image

## Result

`make` produces a fully-linked, flashable image with **ZERO** compile/link
errors and **zero warnings** for Cortex-M7, verified by `arm-none-eabi-size`.

Outputs in `flash_image/`:

| Artifact | Purpose |
|----------|---------|
| `ids_bench.elf` | linked firmware (flash this; openocd reads addresses from it) |
| `ids_bench.bin` | raw binary for `program … 0x08000000` |
| `ids_bench.hex` | Intel HEX |
| `ids_bench.map` | full link map (`--cref`) |

## Files in the image

- `main.c` — NEW standalone entry point: `SystemCoreClockUpdate()` →
  `dwt_cycles_init()` → semihosting-gated `bench_print()` override →
  `ids_bench_run()` → spin.
- `bench.c`, `bench.h`, `model_int8.c`, `model_int8.h`, `model_weights.h`,
  `dwt_cycles.h` — copied from `../hil/` (benchmark + INT8 inference + weights).
- `startup_stm32h7a3zitxq.s`, `system_stm32h7xx.c` — copied from
  `D:/motloud-v3/firmware/datalogger/` (read-only source).
- `STM32H7A3ZITXQ_FLASH.ld` — copied from same; the `/DISCARD/` block that threw
  away `libc.a`/`libm.a`/`libgcc.a` was removed so newlib-nano + rdimon +
  libgcc resolve.
- `cmsis/Include/*`, `cmsis/Device/Include/*` — copied CMSIS core + STM32H7
  device headers.

## Exact verified build command

```bash
cd drafts/submitted/2026-canbus-anomaly-detection-motorcycle/hil/flash_image
make
```

which compiles each source with

```
arm-none-eabi-gcc -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb \
  -O2 -std=c11 -Wall -Wextra -DSTM32H7A3xxQ \
  -I. -Icmsis/Include -Icmsis/Device/Include \
  -ffunction-sections -fdata-sections -fno-common -c <src> -o <obj>
```

and links with

```
arm-none-eabi-gcc <objs> -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb \
  -TSTM32H7A3ZITXQ_FLASH.ld -specs=nano.specs -specs=rdimon.specs \
  -Wl,--gc-sections -Wl,-Map=ids_bench.map,--cref -Wl,--print-memory-usage \
  -o ids_bench.elf
```

> Device define is `-DSTM32H7A3xxQ` (the ZIT6**Q** SMPS variant; only the Q
> device header ships in this CMSIS pack). Toolchain: arm-none-eabi-gcc 15.2.

## `arm-none-eabi-size ids_bench.elf`

```
   text	   data	    bss	    dec	    hex	filename
  17868	     92	  14148	  32108	   7d6c	ids_bench.elf
```

Linker memory usage (from `--print-memory-usage`):

```
Memory region   Used Size  Region Size  %age Used
       FLASH:      17964 B         2 MB      0.86%   (text+rodata+vectors)
         RAM:      14240 B         1 MB      1.36%   (data+bss+heap+stack)
```

Everything fits comfortably: `.text`+`.rodata`+`.isr_vector` in FLASH
(`0x08000000`), `.data`/`.bss`/heap/stack in RAM (`0x24000000`).

## Verification (objdump / nm)

- `objdump -h` — `.isr_vector`@`0x08000000`, `.text`@`0x080002b0`,
  `.rodata`@`0x08001da4` all in FLASH; `.data`@`0x24000000`, `.bss`@`0x2400005c`
  in RAM. ✔
- `nm` confirms present: `ids_bench_run`, `ids_infer_int8`,
  `ids_weights_flash_bytes`, `ids_arena_ram_bytes`, `ids_total_macs`,
  `bench_print` (our override), `Reset_Handler`, `SystemInit`,
  `SystemCoreClock`, `SystemCoreClockUpdate`. ✔
- Model weight/bias arrays in FLASH (`.rodata`, `R`): `ae_w0..ae_w5`,
  `ae_b0..ae_b5`. ✔

## Numbers the human reports back after flashing

After `make` on the Mac side and flashing per `FLASH_AND_READ.md`, copy back the
`==== IDS INT8 Autoencoder HIL Benchmark ==== … ==== end ====` block, in
particular:

1. **Confirmed core clock** — `core_clock : <Hz> (<MHz>)` (expect ~64 MHz HSI).
2. **Latency** — min / **median** / mean / max, in cycles and µs.
3. **Feature-extract** latency (µs).
4. **Activation RAM** (bytes) and **model Flash** (bytes).
5. **CPU utilization** at 10 / 20 / 100 Hz (%).
6. **sample SSE** vs threshold (120000) → normal/ANOMALY (proves the pipeline
   ran end-to-end).
