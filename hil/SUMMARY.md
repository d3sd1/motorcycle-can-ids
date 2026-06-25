# HIL Benchmark Package — SUMMARY

Self-contained hardware-in-the-loop benchmark to obtain a **real on-silicon**
measurement of the INT8 dense autoencoder IDS (`80-40-20-10-20-40-80`, ~8400
MACs) on the Motloud datalogger (**STM32H7A3ZIT6Q, Cortex-M7 @ 280 MHz**),
replacing the analytical CMSIS-NN cycle estimates the reviewers rejected
(JISA R1#2 / R2#1).

## Files in `hil/`

| File | What it is |
|------|-----------|
| `model_int8.h` | Public inference API: `int ids_infer_int8(const int8_t *feat80, int32_t *recon_err_out)` + footprint helpers. |
| `model_int8.c` | Genuine integer-only INT8 forward pass. Self-contained int8 fully-connected kernel (symmetric per-tensor, int32 accumulate, fixed-point requantize, saturate to int8), functionally equivalent to CMSIS-NN `arm_fully_connected_s8`. ReLU fused as activation clamp. `USE_CMSIS_NN` switch calls the real ARM kernel. |
| `model_weights.h` | Auto-generated representative int8 weights (8400) + int32 biases (210). Deterministic, clearly commented as placeholders; resource/timing are weight-value-independent. |
| `dwt_cycles.h` | DWT CYCCNT cycle counter (DEMCR TRCENA + DWT LAR unlock + CYCCNTENA). Board-agnostic cycles->us via runtime `SystemCoreClock`. Self-contained (architectural register addresses; reuses CMSIS defs if present). |
| `bench.h` | Benchmark API: `void ids_bench_run(void)`, weak `bench_print()`, weak `bench_get_real_feature()`. |
| `bench.c` | Runs N=1000 inferences, records min/median/max/mean cycles, converts to us, times the feature-extraction stub, reports activation-arena RAM, model Flash, and projected CPU utilization at 10/20/100 Hz. Weak `printf` (semihosting) default output. |
| `test_main.c` | Minimal `main()` for the stand-alone Makefile build only (not for CubeIDE route). |
| `Makefile` | `make check` = dependency-free compile proof for cortex-m7 (-Werror); `make elf/bin` = optional standalone link with user-supplied startup + linker script. |
| `cmsis_nn/README.md` | How to vendor + build against the real CMSIS-NN `arm_fully_connected_s8` (equivalence confirmation). |
| `tools/gen_weights.py` | Regenerates `model_weights.h` deterministically. |
| `README_HIL.md` | Step-by-step for the human (CubeIDE drop-in = primary; Makefile = secondary). |

## Build command verified (zero errors/warnings)

Toolchain: `arm-none-eabi-gcc (Arm GNU Toolchain 15.2.Rel1) 15.2.1`.

```bash
cd hil && make check
# -> compiles model_int8.c, bench.c, test_main.c with:
#    -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -O2 -std=c11
#    -Wall -Wextra -Werror -ffunction-sections -fdata-sections -ffreestanding -fno-common
# -> "OK: all sources compiled clean for cortex-m7 (-Werror)."
```

Also verified clean: the `USE_CMSIS_NN` wrapper path (compiled against a
type-accurate CMSIS-NN stub header).

## Numerical correctness verified (exact integer emulation)

- `SaturatingRoundingDoublingHighMul` reproduces the gemmlowp/CMSIS-NN
  reference **bit-for-bit** (max diff 0 over 200k random inputs).
- Full int8 forward pass over 3000 synthetic inputs: no int32 accumulator
  overflow, all activations stay within `[-128, 127]`, reconstruction SSE always
  >= 0 and within int32. Requantize is `RoundingDivideByPOT(SDHM(acc<<ls, mult), rs)`.

## Static footprint (exact, from `sizeof` of the real arrays)

- **Flash (weights + biases): 9240 bytes** (8400 int8 weights + 210 int32 biases).
- **Static RAM (activation arena): 160 bytes** (two int8[80] ping-pong buffers).
- Plus `bench.c` timing buffer `uint32_t[1000]` = 4000 bytes (benchmark scratch
  only — not part of the deployed IDS; excludable from the paper's model
  footprint).
- MACs/inference: **8400** (architecture constant).

## Numbers the HUMAN must measure on-board and report back

Flash these via Route A (CubeIDE drop-in) and copy from the printed report into
the paper's embedded-resources table:

1. **Median inference latency (us)** — headline (`latency median`).
2. **Min / Max inference latency (us)** — spread (`latency min` / `latency max`).
3. **Feature-extraction latency (us)** (`feature-extract`).
4. **Static RAM, activation arena (bytes)** (`activation RAM` -> expect 160).
5. **Flash, weights+biases (bytes)** (`model Flash` -> expect 9240).
6. **Projected CPU utilization (%) at 10 / 20 / 100 Hz** (`CPU util @ ...`).
7. **Confirmed core clock (MHz)** (`core_clock` -> expect 280 on the H7A3) — the
   basis for the us conversion; report it alongside the latencies.

Items 4, 5 and 7 are deterministic and already known (160 B / 9240 B / 280 MHz);
items 1-3 and 6 are the genuinely new on-silicon measurements that close the
reviewers' objection.
