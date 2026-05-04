# CAN Bus Anomaly Detection for Competition Motorcycles

Reproducibility package for the paper:

> **Lightweight Autoencoder-Based Anomaly Detection for CAN Bus in Competition Motorcycles Deployed on ARM Cortex-M7**
> *Journal of Information Security and Applications (JISA), Elsevier*

## Requirements

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

## Dataset

The CAN bus dataset from the competition motorcycle testbed will be published on Zenodo upon paper acceptance. Place the dataset files (`.xlsm` Excel session files from the Sevcon Gen4 motor controller) in the `data/` directory.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python run_all.py
```

## Pipeline

`run_all.py` executes the full experiment pipeline:

1. **Data loading** — Parses Sevcon Gen4 Excel session files, reconstructs CAN frames with realistic timing
2. **Feature extraction** — Sliding windows (100 ms, 50% overlap), 80 features per window (13 per CAN ID x 6 IDs + 2 cross-signal)
3. **Data split** — Random 60/20/20 (train/val/test), attacks injected only in test set
4. **Attack injection** — 6 motorcycle-specific attack types at feature level (20 instances each)
5. **Training** — Autoencoder + 4 baselines, 5 seeds (42, 123, 456, 789, 1024)
6. **Evaluation** — Per-attack and overall metrics, INT8 quantization analysis, embedded resource estimation
7. **Figures** — 5 publication-quality PDF figures

## Scripts

| Script | Purpose |
|--------|---------|
| `run_all.py` | One-command orchestrator |
| `data_loader.py` | Parse Sevcon Excel to CAN frames |
| `feature_extraction.py` | Sliding window feature extraction |
| `attack_injection.py` | 6-type attack injection framework |
| `train.py` | Autoencoder + baseline training |
| `evaluate.py` | Metrics computation + embedded estimation |
| `generate_figures.py` | Publication figures (PDF) |
| `config.json` | All hyperparameters |

## Configuration

All hyperparameters are centralized in `config.json`. Key parameters:

- Autoencoder: 80-40-20-10-20-40-80, ReLU, Adam lr=0.001, batch 64, early stopping patience 20
- Seeds: [42, 123, 456, 789, 1024]
- Window: 100 ms duration, 50% overlap
- Threshold: 99th percentile of validation reconstruction error
- Target MCU: STM32H743VIT6 (Cortex-M7, 480 MHz, 512 KB SRAM)

## Attack Taxonomy

Six motorcycle-specific attack types:

1. **TPS Spoofing (A1)** — Throttle position sensor value manipulation
2. **Lean Angle Injection (A2)** — Fabricated lean angle readings
3. **BMS Disappearance (A3)** — Battery management system node silence
4. **Replay (A4)** — Replayed CAN traffic segments
5. **Fuzzing (A5)** — Random payload injection
6. **DoS Flooding (A6)** — Bus saturation attacks

## Outputs

- `results/aggregated_results.json` — All metrics (mean +/- std over 5 seeds)
- `results/results_seed_*.json` — Per-seed detailed results
- `results/data_statistics.json` — Dataset statistics
- `figures/*.pdf` — Publication figures

## Hardware

Training was performed on CPU (AMD Ryzen 9, 24 threads). No GPU required. Total execution time: ~35 minutes.

## License

MIT
