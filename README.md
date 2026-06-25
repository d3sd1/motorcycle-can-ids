# Lightweight CAN-Bus Intrusion Detection for Competition Motorcycles

Reproducibility package for the paper *"Lightweight Autoencoder-Based Anomaly
Detection for CAN Bus in Competition Motorcycles Designed for ARM Cortex-M7"*.

## What this is
A **hybrid multi-layer IDS** for the CAN bus of competition motorcycles:
- a deterministic **rule layer** — per-node heartbeat monitoring (node disappearance)
  and per-identifier frequency counting (DoS flooding);
- a learned **INT8 autoencoder** layer for stealthy in-range manipulations.

The hybrid (F1 0.892, 5 seeds) outperforms the strongest single baseline
(threshold, F1 0.814) and the INT8 autoencoder alone (0.738), and is deployable
on an ARM Cortex-M7.

## Layout
- `run_all.py` — one-command pipeline (5 seeds: 42,123,456,789,1024). `SKIP_LSTM=1` skips the LSTM baseline.
- `train.py`, `feature_extraction.py`, `attack_injection.py`, `hybrid_detector.py`,
  `evaluate.py`, `generate_figures.py` — components.
- `config.json` — all hyperparameters and the 6-class attack taxonomy.
- `requirements.txt` — pinned dependencies.
- `results/aggregated_results.json`, `results/HYBRID_SUMMARY.md` — aggregated metrics (mean±std).
- `hil/` — **on-silicon profiling**: genuine integer-only INT8 kernel, DWT cycle
  benchmark, and a flashable STM32H7A3 image (`hil/flash_image/`) with the raw
  measurement log (`measurement_stm32h7a3_64mhz.txt`). Measured end-to-end
  141,137 cycles (≈0.50 ms at 280 MHz; ≈2.2 ms at the 64 MHz bench clock).

## Reproduce
```bash
python -m venv .venv && . .venv/bin/activate   # (Scripts\activate on Windows)
pip install -r requirements.txt
python run_all.py            # full pipeline; SKIP_LSTM=1 python run_all.py for the fast path
```
The reconstructed dataset is released separately (see the paper's Data Availability statement).

## License
MIT (see `LICENSE`).
