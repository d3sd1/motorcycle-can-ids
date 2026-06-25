# Experiment Plan — CAN Bus Anomaly Detection for Competition Motorcycles

## Paper
- **Title**: Lightweight Autoencoder-Based Anomaly Detection for CAN Bus in Competition Motorcycles Deployed on ARM Cortex-M7
- **Venue**: JISA (Elsevier)
- **Draft path**: `drafts/ideas/2026-canbus-anomaly-detection-motorcycle/`

## Overview

The experiments must:
1. Parse real CAN traffic from the motorcycle testbed dataset
2. Implement the attack injection framework (6 attack types)
3. Train an autoencoder on normal traffic
4. Evaluate detection performance (F1, precision, recall, FPR, detection latency)
5. Compare against 4 baselines (threshold, OC-SVM, Isolation Forest, LSTM-AE)
6. Analyze INT8 quantization impact on accuracy and size
7. Estimate embedded resource usage (model size, inference latency on Cortex-M7)
8. Run 5 seeds (42, 123, 456, 789, 1024) and report mean +/- std

## Data Sources

### Real CAN Data
- **Location**: `datasets/dataset_motostudent_electric/`
- **Format**: Excel files from Sevcon Gen4 data exports and session logs
- **Key files**:
  - `Plantilla_Sevcon_V7.xltm` — Sevcon CAN data template
  - `FK1_*.xlsm` — FK1 circuit sessions (multiple)
  - `Free_Practice_1.xlsm` — Free practice sessions
  - `Test_*.xlsm` — Test sessions
  - `JARAMA*.xlsx` — Jarama circuit sessions
  - `Banco_EME18E_*.xlsm` — Bench test sessions (lab environment)
  - `session_jarama_*.csv.xlsx` — Jarama CSV exports

### Data Preprocessing
1. Parse Excel/CSV files to extract CAN-like signals (timestamps, motor speed, motor current, TPS, temperatures, voltages, SOC)
2. Since the raw data is likely Sevcon application-layer decoded (not raw CAN frames), reconstruct CAN-like traffic by:
   - Assigning CAN IDs to signal groups (VCU=0x100-0x10F, Motor=0x200-0x20F, BMS=0x300-0x30F, IMU=0x400-0x40F, Dashboard=0x500)
   - Encoding signals back into 8-byte CAN payloads using a synthetic DBC
   - Applying realistic CAN timing (10ms, 20ms, 100ms periods by signal type)
3. Validate reconstructed traffic statistics against known CAN bus characteristics

## Experiments

### Experiment 1: Data Preparation
- Parse all available session files
- Reconstruct CAN traffic from decoded signals
- Compute statistics: number of sessions, duration, unique IDs, message counts
- Split: 60% train / 20% val / 20% test
- **Output**: `results/data_statistics.json`

### Experiment 2: Attack Injection
- For each of the 6 attack types (A1-A6), inject N instances into the test set
- Randomize attack parameters within defined bounds
- Label all frames as normal/attack with attack type
- **Output**: `results/attack_injection_summary.json`

### Experiment 3: Autoencoder Training and Evaluation
- Extract features using sliding windows (Tw=100ms, 50% overlap)
- Train autoencoder (80-40-20-10-20-40-80 architecture)
- 5 seeds: 42, 123, 456, 789, 1024
- Calibrate threshold on validation set (99th percentile)
- Evaluate on test set with injected attacks
- Compute per-attack and overall metrics
- **Output**: `results/autoencoder_results.json`, `results/per_attack_results.json`

### Experiment 4: Baseline Comparison
- Train and evaluate all 4 baselines on the same data:
  - Threshold-based (3-sigma)
  - One-Class SVM (RBF, nu=0.01)
  - Isolation Forest (100 trees, contamination=0.01)
  - LSTM-AE (64 hidden units, sequence length 10)
- 5 seeds each
- **Output**: `results/baseline_comparison.json`

### Experiment 5: Quantization Analysis
- Apply post-training INT8 quantization to the trained autoencoder
- Compare FP32 vs INT8: F1, model size, inference latency
- Estimate Cortex-M7 resource usage from model structure
- **Output**: `results/quantization_analysis.json`

### Experiment 6: Embedded Deployment Estimation
- Calculate model size in bytes (INT8 weights + biases)
- Estimate inference latency from MAC operations + Cortex-M7 clock
- Estimate RAM usage (peak activation tensor)
- **Output**: `results/embedded_analysis.json`

## Expected Metrics

Based on the literature and the simpler motorcycle CAN traffic:
- **Autoencoder F1**: 0.95-0.98 (overall)
- **DoS/Fuzzing F1**: >0.99 (easy to detect)
- **Replay F1**: 0.85-0.92 (hardest)
- **INT8 model size**: <10 KB
- **INT8 inference latency**: <1 ms (estimated from MAC operations)
- **Total RAM**: <15 KB (model + activations + buffers)

## Figure Generation

### Figure 1: System Architecture
- Block diagram: CAN bus -> Feature Extraction -> Autoencoder -> Threshold -> Alert
- With STM32H7 deployment context
- Format: PDF, TikZ

### Figure 2: ROC Curves
- One ROC curve per method (5 methods)
- Per-attack type subplot (6 subplots)
- Format: PDF, matplotlib

### Figure 3: Reconstruction Error Distribution
- Histogram of reconstruction errors for normal vs attack traffic
- Threshold line shown
- Format: PDF, matplotlib

### Figure 4: Per-Attack F1 Bar Chart
- Grouped bar chart: 5 methods x 6 attack types
- Format: PDF, matplotlib

### Figure 5: Quantization Impact
- Model size vs F1-score scatter for FP32, FP16, INT8
- Format: PDF, matplotlib

## Dependencies

```
torch>=2.1.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
openpyxl>=3.1.0
xlrd>=2.0.0
seaborn>=0.12.0
```

## File Structure

```
experiments/
├── config.json           # All hyperparameters
├── requirements.txt      # Dependencies
├── train.py              # Train autoencoder and baselines
├── evaluate.py           # Evaluate all methods
├── attack_injection.py   # Attack injection framework
├── data_loader.py        # Parse CAN data from dataset
├── feature_extraction.py # Sliding window feature extraction
├── quantize.py           # INT8 quantization
├── generate_figures.py   # All paper figures
├── run_all.py            # One-command orchestrator
├── results/
│   ├── aggregated_results.json
│   └── (per-experiment JSONs)
├── figures/
│   └── (generated PDFs)
└── README.md
```
