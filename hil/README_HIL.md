# HIL Benchmark — INT8 Autoencoder IDS on STM32H7 (Cortex-M7)

This package gives a **real on-silicon measurement** of the INT8 dense
autoencoder intrusion-detection model (`80-40-20-10-20-40-80`, ~8400 MACs) on
the Motloud datalogger board. It replaces the previously-reported *analytical
CMSIS-NN cycle estimates* with **measured** inference latency, RAM, Flash and
CPU utilization, using the Cortex-M7 DWT cycle counter.

- **Target board:** STM32H7A3ZIT6Q, Cortex-M7 @ **280 MHz** (the real Motloud
  datalogger). The harness is **board-agnostic**: it reads the actual core
  clock from `SystemCoreClock` at run time, so cycles→µs is correct whether the
  silicon runs at 280 MHz (H7A3) or 480 MHz (H743). **You do not need to edit
  any clock constant** — flash and read the printed `core_clock` line to
  confirm.
- **Inference is genuine integer-only INT8** (int8 weights, int8 activations,
  int32 accumulate, fixed-point requantize), using a self-contained kernel that
  is functionally equivalent to CMSIS-NN `arm_fully_connected_s8`. Zero external
  dependencies by default; a `USE_CMSIS_NN` switch builds against the real ARM
  kernel (see `cmsis_nn/README.md`).

> **Note on weight values:** `model_weights.h` holds *representative* int8
> weights (deterministic PRNG). Inference **latency, RAM and Flash are
> architecture- and quantization-dependent and weight-value-independent**, so
> the measured resource/timing numbers are valid for the trained model too.
> Only the detection **F1** depends on the trained weights, and that is measured
> offline in software — never on this benchmark. (The benchmark's printed
> `sample SSE → ANOMALY/normal` line is illustrative of the pipeline running,
> not a detection result.)

---

## Files

| File | Purpose |
|------|---------|
| `model_int8.h/.c` | Integer-only INT8 forward pass + anomaly score. Public API `ids_infer_int8()`. |
| `model_weights.h` | Auto-generated representative int8 weights/biases (regen with `tools/gen_weights.py`). |
| `dwt_cycles.h` | DWT cycle counter; cycles→µs via `SystemCoreClock`. |
| `bench.h/.c` | `ids_bench_run()` — runs N=1000 inferences, prints the report. |
| `cmsis_nn/` | How to build against the real CMSIS-NN kernel (optional). |
| `test_main.c` | Tiny `main()` for the stand-alone Makefile build only. |
| `Makefile` | `make check` (compile proof) and `make elf` (optional standalone link). |

---

## Route A — CubeIDE drop-in (PRIMARY, recommended)

This is the supported flashing path on the real datalogger.

1. **Copy the files** into the CubeIDE datalogger project, e.g. create a folder
   `Core/Ids/` and copy into it:
   ```
   model_int8.c  model_int8.h  model_weights.h
   bench.c       bench.h       dwt_cycles.h
   ```
   (Do **not** copy `test_main.c` or the `Makefile` — the project has its own
   `main()` and build system.)

2. **Add the include path:** *Project ▸ Properties ▸ C/C++ Build ▸ Settings ▸
   MCU GCC Compiler ▸ Include paths* → add `../Core/Ids`. CubeIDE compiles
   `.c` files in source folders automatically; if `Core/Ids` is not already a
   source location, add it under *C/C++ General ▸ Paths and Symbols ▸ Source
   Location*.

3. **Call the benchmark once** from the project's `main()`, *after*
   `SystemClock_Config()` and `SystemCoreClockUpdate()` (so `SystemCoreClock`
   holds the true core frequency):
   ```c
   #include "bench.h"
   ...
   int main(void) {
       HAL_Init();
       SystemClock_Config();
       SystemCoreClockUpdate();   /* ensures SystemCoreClock is current */
       ...
       ids_bench_run();           /* prints the HIL report once */
       ...
   }
   ```
   Place it **before** the FreeRTOS scheduler starts (or in a one-shot task) so
   it runs uninterrupted.

4. **Wire the output to UART** (recommended) by overriding the weak
   `bench_print()` — drop this into any `.c` of the project (e.g. next to
   `log_stream.c`):
   ```c
   #include "log_stream.h"
   void bench_print(const char *line) {
       LogStream_Push(LOG_LVL_INFO, line);     /* -> MQTT + SD */
       /* or push directly to USART3 via your SAFE_PRINTF wrapper */
   }
   ```
   If you skip this, output goes to `printf` (works if semihosting or a
   `_write` UART retarget is enabled; the datalogger already has a weak
   `_write` in `syscalls.c`).

5. **Build ▸ Flash** with STM32CubeProgrammer (or the CubeIDE *Run* button with
   the ST-LINK). Open a serial terminal on the datalogger's debug UART (USART3)
   — or watch the MQTT/SD log if you routed `bench_print` to `LogStream` — and
   read the report.

---

## Route B — Stand-alone Makefile (secondary / CI proof)

Use this to **prove the code compiles clean** for the target without CubeIDE,
and optionally to produce a standalone `.elf`.

```bash
cd hil
make check          # compile-only, -Werror, cortex-m7 -> "OK: all sources compiled clean"
```

`make check` requires only the Arm GNU toolchain (verified with
`arm-none-eabi-gcc 15.2.Rel1`). It has **zero other dependencies**.

A full standalone link needs the startup file + linker script from your CubeIDE
project (the benchmark itself has no `SystemClock_Config`/vector table):

```bash
make elf \
  STARTUP=/path/to/Core/Startup/startup_stm32h7a3zitxq.s \
  LDSCRIPT=/path/to/STM32H7A3ZITXQ_FLASH.ld
make bin            # -> ids_hil.bin to flash with STM32CubeProgrammer
```

For the actual board measurement, **Route A is preferred** — it runs inside the
real firmware with the real clock tree.

---

## Numbers to copy back into the paper

Read these straight off the printed report (the embedded-resources table):

| Report line | Paper field |
|-------------|-------------|
| `latency median ... = X.XX us` | **Median inference latency (µs)** ← headline |
| `latency min` / `latency max` | latency spread (min/max µs) |
| `feature-extract ... = X.XX us` | feature-extraction latency (µs) |
| `activation RAM : N bytes` | **Static RAM (activation arena), bytes** |
| `model Flash : N bytes` | **Flash (weights+biases), bytes** |
| `CPU util @ 10/20/100Hz : X.XXX %` | **Projected CPU utilization (%)** at each rate |
| `core_clock : ... MHz` | confirm the board clock the µs were derived from |

Report the **median** latency as the headline (robust to outliers); include
min/max as the spread. The RAM/Flash bytes are exact (`sizeof` of the real
static arrays). CPU% is `median_µs × rate / 10000`.

---

## Optional — profile on a real captured CAN feature vector

Timing is input-value-independent, but if you want the benchmark to run on a
real captured 80-int8 feature vector (13 features × 6 CAN IDs + 2), override the
weak hook anywhere in the project:

```c
static const signed char captured_feat[80] = { /* your int8 quantized features */ };
const signed char *bench_get_real_feature(void) { return captured_feat; }
```

The latency/RAM/Flash numbers are unchanged; only the printed `sample SSE` line
reflects the real frame.

---

## Clock note (280 vs 480 MHz)

The datalogger is an **STM32H7A3 @ 280 MHz**, not the H743 @ 480 MHz that an
earlier paper config assumed. **No edit is needed:** the harness divides the
measured cycle count by the runtime `SystemCoreClock`, so the reported µs
auto-correct to whatever the board actually runs. Always copy the µs values
**and** the printed `core_clock` line into the paper so the conversion basis is
explicit.
