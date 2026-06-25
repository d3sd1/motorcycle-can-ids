# FLASH AND READ — STM32H7A3ZIT6Q on-silicon HIL benchmark

Exact openocd + gdb sequence to flash `ids_bench.elf`, enable semihosting, and
capture the benchmark output. Target: **STM32H7A3ZIT6Q** (Cortex-M7) via an
**ST-Link V3**. Commands written for a **Mac** where openocd lives at:

```
~/tools/xpack-openocd-0.12.0-6/bin/openocd
```

Copy `ids_bench.elf` (and optionally `ids_bench.bin`) to the Mac first.

Throughout, `OOCD=~/tools/xpack-openocd-0.12.0-6/bin/openocd`.

---

## 0. (Recommended) BACKUP-FIRST — save whatever is currently on the chip

Do this **before** programming so you can restore the spare board's existing
firmware if it had anything worth keeping. Dumps the entire 2 MB main FLASH
bank (`0x08000000`, length `0x200000`).

```bash
OOCD=~/tools/xpack-openocd-0.12.0-6/bin/openocd

$OOCD -f interface/stlink.cfg -f target/stm32h7x.cfg \
  -c "init" \
  -c "reset halt" \
  -c "dump_image backup_spare.bin 0x08000000 0x200000" \
  -c "shutdown"
```

You now have `backup_spare.bin`. **To restore it later** (only if the spare had
firmware you want back):

```bash
$OOCD -f interface/stlink.cfg -f target/stm32h7x.cfg \
  -c "init" \
  -c "program backup_spare.bin 0x08000000 verify reset exit"
```

---

## 1. Flash the benchmark image

One-shot, non-interactive — programs, verifies, resets, and exits:

```bash
$OOCD -f interface/stlink.cfg -f target/stm32h7x.cfg \
  -c "program ids_bench.elf verify reset exit"
```

(`program` accepts the ELF directly; addresses come from the ELF. Equivalent
with the raw binary: `program ids_bench.bin 0x08000000 verify reset exit`.)

---

## 2. Run and capture the semihosted output

The image writes its report via ARM semihosting, so you must enable semihosting
and let the core run. Two ways — pick ONE.

### Option A — pure openocd (simplest)

openocd prints semihosted writes to its own stdout/log.

```bash
$OOCD -f interface/stlink.cfg -f target/stm32h7x.cfg \
  -c "init" \
  -c "reset halt" \
  -c "arm semihosting enable" \
  -c "reset run"
```

Leave it running. The benchmark prints once (it runs `ids_bench_run()` then
spins). The report appears in openocd's console. Press `Ctrl-C` to stop openocd
after you see `==== end ====`.

To save it to a file, redirect openocd's output:

```bash
$OOCD -f interface/stlink.cfg -f target/stm32h7x.cfg \
  -c "init" -c "reset halt" -c "arm semihosting enable" -c "reset run" \
  2>&1 | tee bench_output.txt
```

### Option B — gdb driving openocd (if you prefer a gdb session)

Terminal 1 — start openocd as a gdb server with semihosting on:

```bash
$OOCD -f interface/stlink.cfg -f target/stm32h7x.cfg \
  -c "init" -c "arm semihosting enable"
```

Terminal 2 — connect arm-none-eabi-gdb:

```bash
arm-none-eabi-gdb ids_bench.elf \
  -ex "target extended-remote localhost:3333" \
  -ex "monitor arm semihosting enable" \
  -ex "load" \
  -ex "monitor reset halt" \
  -ex "continue"
```

Semihosted output appears in the **openocd** terminal (Terminal 1). After
`==== end ====`, `Ctrl-C` in gdb, then `quit`.

---

## 3. Which exact lines to copy back

The report is bracketed by `==== IDS INT8 Autoencoder HIL Benchmark ====` and
`==== end ====`. Copy back **these** lines verbatim:

- `core_clock      : <N> Hz (<M> MHz)`  ← **confirmed core clock** (expect ~64 MHz HSI)
- `latency min     : <c> cyc  = <µs> us`  ← **min latency**
- `latency median  : <c> cyc  = <µs> us`  ← **median latency (headline)**
- `latency mean    : <c> cyc  = <µs> us`
- `latency max     : <c> cyc  = <µs> us`  ← **max latency**
- `feature-extract : <c> cyc  = <µs> us`  ← **feature-extraction µs**
- `activation RAM  : <bytes> bytes`
- `model Flash     : <bytes> bytes`
- `CPU util @ 10Hz : <p> %`
- `CPU util @ 20Hz : <p> %`
- `CPU util @100Hz : <p> %`  ← **CPU% lines**
- `sample SSE      : <e>  (threshold 120000) -> normal|ANOMALY`

The whole `==== … ==== end ====` block is fine to paste back in full.

---

## Notes / troubleshooting

- **No output appears:** semihosting was not enabled before the core ran, or the
  core was reset without it. Always `arm semihosting enable` **before**
  `reset run`/`continue`. The image guards the `bkpt 0xAB` traps on
  C_DEBUGEN, so output only flows while the debugger is attached.
- **`reset run` doesn't re-trigger the print:** the benchmark runs once per boot.
  Issue another `reset run` (semihosting still enabled) to re-run it.
- **openocd can't find the target:** confirm the ST-Link V3 firmware is current
  and try adding `-c "adapter speed 1800"` before `init`; for some boards
  `-f target/stm32h7x_dual_bank.cfg` is needed instead of `stm32h7x.cfg`.
- **Core clock reads ~64 MHz, not 280:** expected — this image runs at the
  post-reset HSI clock by design; microseconds are computed from the runtime
  `SystemCoreClock`, so they are correct for 64 MHz. See README.md.
