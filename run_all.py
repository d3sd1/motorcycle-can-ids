#!/usr/bin/env python3
"""
One-command orchestrator for all experiments.

Runs the complete pipeline:
1. Data loading and CAN traffic reconstruction
2. Feature extraction from all normal traffic
3. Random split of feature windows (60/20/20)
4. Attack injection at feature level for test set
5. Model training (autoencoder + baselines, 5 seeds)
6. Evaluation (overall + per-attack + quantization + embedded)
7. Results aggregation
8. Figure generation

Paper: Lightweight Autoencoder-Based Anomaly Detection for CAN Bus
       in Competition Motorcycles Deployed on ARM Cortex-M7

Usage:
    python run_all.py              # Full pipeline
    python run_all.py --dry-run    # Quick test with minimal data
"""

import json
import sys
import time
import os
import platform
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_all_sessions, CANFrame, CAN_SIGNAL_MAP
from feature_extraction import (extract_dataset_features, normalize_features,
                                MONITORED_IDS, FE_CONFIG)
from attack_injection import inject_all_attacks, CONFIG as ATK_CONFIG
from train import (CANAutoencoder, LSTMAutoencoder, train_autoencoder,
                   calibrate_threshold, quantize_model_int8,
                   train_ocsvm, train_isolation_forest, train_lstm_ae,
                   create_sequences, set_all_seeds)
from evaluate import (evaluate_detector, evaluate_per_attack,
                      threshold_detector, estimate_embedded_resources,
                      estimate_detection_latency_per_attack,
                      ATTACK_NAMES, METHODS)

# Load config
CONFIG_PATH = SCRIPT_DIR / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

RESULTS_DIR = SCRIPT_DIR / CONFIG["output"]["results_dir"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = SCRIPT_DIR / CONFIG["output"]["figures_dir"]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = SCRIPT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def get_hardware_info() -> dict:
    """Collect hardware information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
    return info


def inject_attacks_feature_level(X_normal: np.ndarray, seed: int,
                                 n_instances_per_type: int = 20) -> tuple:
    """Inject attacks at the feature level by modifying normal windows.

    For each attack type, takes a subset of normal windows and modifies
    their features in ways consistent with each attack's effect on CAN
    traffic characteristics.

    Returns: (X_test, y_test, attack_types)
    """
    rng = np.random.default_rng(seed)
    n_features = X_normal.shape[1]  # 80 features = 6 IDs x 13 + 2

    # Feature layout: for each of 6 CAN IDs (13 features each):
    #   [0:3]  = timing (mean_iat, std_iat, msg_count)
    #   [3:11] = payload (mean bytes 0-7)
    #   [11]   = entropy
    #   [12]   = max_change_rate
    # + 2 cross-signal features at the end [78, 79]

    # ID indices in MONITORED_IDS: 0x100=0, 0x200=1, 0x201=2, 0x300=3, 0x301=4, 0x400=5

    attack_windows = []
    attack_labels = []
    attack_types_list = []

    def get_id_slice(id_idx):
        """Get feature slice for CAN ID at index id_idx."""
        start = id_idx * 13
        return start, start + 13

    for attack_name in ATTACK_NAMES:
        for _ in range(n_instances_per_type):
            # Pick a random normal window
            idx = rng.integers(0, len(X_normal))
            x = X_normal[idx].copy()

            if attack_name == "A1_tps_spoofing":
                # TPS spoofing: modify VCU (ID 0, 0x100) features
                s, e = get_id_slice(0)
                # Timing: higher msg count (injected msgs), slightly lower IAT
                x[s + 0] *= rng.uniform(0.3, 0.8)   # mean IAT decreases
                x[s + 1] *= rng.uniform(2.0, 5.0)    # std IAT increases (irregular)
                x[s + 2] *= rng.uniform(1.5, 3.0)    # msg count increases
                # Payload: spoofed TPS value (bytes 0-1 change drastically)
                x[s + 3] = rng.uniform(0, 255)        # mean byte 0
                x[s + 4] = rng.uniform(0, 255)        # mean byte 1
                x[s + 11] *= rng.uniform(0.3, 0.7)   # entropy drops (constant spoofed val)
                x[s + 12] *= rng.uniform(3.0, 10.0)  # max change rate spikes
                # Cross-signal: throttle-current correlation breaks
                x[78] *= rng.uniform(-0.5, 0.3)

            elif attack_name == "A2_lean_injection":
                # Lean angle spoofing: modify IMU (ID 5, 0x400) features
                s, e = get_id_slice(5)
                x[s + 0] *= rng.uniform(0.5, 0.9)    # IAT slightly changes
                x[s + 1] *= rng.uniform(1.5, 4.0)    # std IAT increases
                x[s + 2] *= rng.uniform(1.0, 2.0)    # msg count may increase
                x[s + 3] = rng.uniform(0, 255)        # spoofed lean angle bytes
                x[s + 4] = rng.uniform(0, 255)
                x[s + 11] *= rng.uniform(0.3, 0.6)   # entropy drops
                x[s + 12] *= rng.uniform(5.0, 15.0)  # huge change rate
                # Lean-gyro consistency breaks
                x[79] *= rng.uniform(-0.5, 0.2)

            elif attack_name == "A3_bms_disappearance":
                # BMS disappearance: zero out BMS features (IDs 3,4 = 0x300, 0x301)
                for bms_id in [3, 4]:
                    s, e = get_id_slice(bms_id)
                    x[s:e] = 0.0  # all BMS features go to zero

            elif attack_name == "A4_replay":
                # Replay: replace with a different normal window's features
                # but keep timing slightly off (duplicate messages)
                other_idx = rng.integers(0, len(X_normal))
                x_other = X_normal[other_idx].copy()
                # Mix: use payload from other window but timing is doubled
                for id_idx in range(6):
                    s, e = get_id_slice(id_idx)
                    # Keep original timing but slightly modified
                    x[s + 0] *= rng.uniform(0.7, 0.9)   # IAT decreases
                    x[s + 1] *= rng.uniform(1.5, 3.0)    # std IAT increases
                    x[s + 2] *= rng.uniform(1.3, 2.0)    # msg count increases
                    # Payload from the replayed window
                    x[s + 3:s + 11] = x_other[s + 3:s + 11]
                    # Entropy slightly different
                    x[s + 11] = x_other[s + 11] * rng.uniform(0.8, 1.2)
                    x[s + 12] *= rng.uniform(1.5, 4.0)   # change rate higher

            elif attack_name == "A5_fuzzing":
                # Fuzzing: random payloads on random IDs
                n_fuzzed = rng.integers(1, 4)  # fuzz 1-3 IDs
                fuzzed_ids = rng.choice(6, n_fuzzed, replace=False)
                for id_idx in fuzzed_ids:
                    s, e = get_id_slice(id_idx)
                    # Timing: extra messages at irregular intervals
                    x[s + 1] *= rng.uniform(3.0, 8.0)    # std IAT very high
                    x[s + 2] *= rng.uniform(1.2, 2.0)    # more messages
                    # Random payloads
                    x[s + 3:s + 11] = rng.uniform(50, 200, 8)  # random byte means
                    x[s + 11] = rng.uniform(6.0, 8.0)    # high entropy (random data)
                    x[s + 12] = rng.uniform(200, 255)     # max change rate very high
                # Cross-signal correlations break
                x[78] = rng.uniform(-0.3, 0.3)
                x[79] = rng.uniform(-0.3, 0.3)

            elif attack_name == "A6_dos_flooding":
                # DoS: massive increase in message count, new ID 0x000
                # All IDs see timing disruption
                for id_idx in range(6):
                    s, e = get_id_slice(id_idx)
                    x[s + 0] *= rng.uniform(1.5, 5.0)    # IAT increases (bus congestion)
                    x[s + 1] *= rng.uniform(3.0, 10.0)   # std IAT very high
                    x[s + 2] *= rng.uniform(0.3, 0.7)    # fewer legitimate msgs get through
                # Cross-signal: everything breaks
                x[78] = rng.uniform(-0.2, 0.2)
                x[79] = rng.uniform(-0.2, 0.2)

            attack_windows.append(x)
            attack_labels.append(1)
            attack_types_list.append(attack_name)

    # Build test set: normal windows + attack windows
    n_test_normal = len(X_normal)
    X_attack = np.array(attack_windows)
    X_test = np.vstack([X_normal, X_attack])
    y_test = np.concatenate([np.zeros(n_test_normal, dtype=np.int32),
                              np.ones(len(attack_windows), dtype=np.int32)])
    types = np.concatenate([np.array([""] * n_test_normal),
                             np.array(attack_types_list)])

    # Shuffle
    perm = rng.permutation(len(X_test))
    X_test = X_test[perm]
    y_test = y_test[perm]
    types = types[perm]

    return X_test, y_test, types


def run_pipeline(dry_run: bool = False):
    """Execute the complete experiment pipeline."""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("CAN Bus Anomaly Detection -- Full Experiment Pipeline")
    print(f"Timestamp: {timestamp}")
    print(f"Mode: {'DRY RUN (minimal data)' if dry_run else 'FULL'}")
    print("=" * 70, flush=True)

    hardware = get_hardware_info()
    print(f"\nHardware: {hardware['processor']}, {hardware['cpu_count']} cores")
    if hardware["cuda_available"]:
        print(f"GPU: {hardware['gpu']} ({hardware['gpu_memory_gb']} GB)")
    else:
        print("GPU: Not available (using CPU)")

    seeds = CONFIG["random_seeds"]
    print(f"Seeds: {seeds}", flush=True)

    # ================================================================
    # STEP 1: Data Loading
    # ================================================================
    print(f"\n{'='*70}")
    print("[1/7] Loading CAN traffic data from testbed...")
    print(f"{'='*70}", flush=True)

    max_sessions = 3 if dry_run else 20
    all_frames, data_stats = load_all_sessions(max_sessions=max_sessions)

    if not all_frames:
        print("ERROR: No CAN frames loaded. Check dataset path in config.json.")
        sys.exit(1)

    print(f"\nData summary:")
    print(f"  Sessions loaded: {data_stats['n_sessions']}")
    print(f"  Total frames: {data_stats['total_frames']:,}")
    total_dur_h = data_stats['total_duration_s'] / 3600
    print(f"  Total duration: {data_stats['total_duration_s']:.0f} s ({total_dur_h:.1f} h)")
    print(f"  Unique CAN IDs: {data_stats['unique_can_ids']}")
    print(f"  CAN nodes: {data_stats['can_nodes']}", flush=True)

    with open(RESULTS_DIR / "data_statistics.json", "w") as f:
        json.dump(data_stats, f, indent=2)

    # ================================================================
    # STEP 2: Feature extraction from ALL normal traffic
    # ================================================================
    print(f"\n{'='*70}")
    print("[2/7] Extracting features from all normal traffic...")
    print(f"{'='*70}", flush=True)

    X_all, y_all, types_all = extract_dataset_features(all_frames, MONITORED_IDS)
    assert np.all(y_all == 0), "All raw data should be normal traffic"
    print(f"  Total feature windows: {X_all.shape[0]}")
    print(f"  Feature dimension: {X_all.shape[1]}", flush=True)

    # ================================================================
    # STEP 3: Random split of windows (60/20/20)
    # ================================================================
    print(f"\n{'='*70}")
    print("[3/7] Splitting windows randomly (60/20/20)...")
    print(f"{'='*70}", flush=True)

    n_total = len(X_all)
    rng_split = np.random.default_rng(0)  # fixed seed for split
    perm = rng_split.permutation(n_total)

    n_train = int(n_total * CONFIG["data"]["train_ratio"])
    n_val = int(n_total * CONFIG["data"]["val_ratio"])

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    X_train_raw = X_all[train_idx]
    X_val_raw = X_all[val_idx]
    X_test_normal = X_all[test_idx]

    print(f"  Train: {len(X_train_raw)} windows (normal)")
    print(f"  Val:   {len(X_val_raw)} windows (normal)")
    print(f"  Test (normal part): {len(X_test_normal)} windows", flush=True)

    # ================================================================
    # STEP 4: Attack injection at feature level + normalization
    # ================================================================
    print(f"\n{'='*70}")
    print("[4/7] Injecting attacks into test set (feature-level)...")
    print(f"{'='*70}", flush=True)

    n_instances = 5 if dry_run else CONFIG["attack_injection"]["instances_per_attack_type"]
    X_test_raw, y_test, types_test = inject_attacks_feature_level(
        X_test_normal, seed=42, n_instances_per_type=n_instances
    )

    n_attack = int(np.sum(y_test == 1))
    n_normal = int(np.sum(y_test == 0))
    print(f"  Attack instances per type: {n_instances}")
    print(f"  Test set: {n_normal} normal + {n_attack} attack = {len(y_test)} total")
    print(f"  Attack ratio: {n_attack / len(y_test) * 100:.1f}%", flush=True)

    # Normalize features
    print("\n  Normalizing features to [0, 1]...")
    X_train_norm, X_val_norm, X_test_norm, feat_min, feat_max = normalize_features(
        X_train_raw, X_val_raw, X_test_raw
    )

    # Handle NaN/Inf
    for arr_name, arr in [("X_train", X_train_norm), ("X_val", X_val_norm),
                           ("X_test", X_test_norm)]:
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        if n_nan > 0 or n_inf > 0:
            print(f"  WARNING: {arr_name} has {n_nan} NaN and {n_inf} Inf. Replacing with 0.")
            arr[np.isnan(arr)] = 0.0
            arr[np.isinf(arr)] = 0.0

    print(f"\n  Feature shapes: train={X_train_norm.shape}, val={X_val_norm.shape}, "
          f"test={X_test_norm.shape}", flush=True)

    # Save injection summary
    injection_summary = {
        "n_instances_per_type": n_instances,
        "n_attack_types": 6,
        "total_attack_windows": n_attack,
        "total_normal_windows": n_normal,
        "attack_ratio": round(n_attack / len(y_test), 4),
    }
    with open(RESULTS_DIR / "attack_injection_summary.json", "w") as f:
        json.dump(injection_summary, f, indent=2)

    # ================================================================
    # STEP 5: Training & Evaluation (5 seeds)
    # ================================================================
    print(f"\n{'='*70}")
    print("[5/7] Training and evaluating all methods (5 seeds)...")
    print(f"{'='*70}", flush=True)

    # Model info
    input_dim = X_train_norm.shape[1]
    latent_dim = CONFIG["autoencoder"]["latent_dim"]
    dummy_ae = CANAutoencoder(input_dim, latent_dim)
    n_params = dummy_ae.count_parameters()
    layers_info = dummy_ae.get_layer_info()
    print(f"\n  Autoencoder: {n_params} parameters")
    print(f"  Input dim: {input_dim}, Latent dim: {latent_dim}")
    print(f"  Architecture: {CONFIG['autoencoder']['architecture']}", flush=True)

    # LSTM sequences
    seq_len = CONFIG["baselines"]["lstm_ae"]["sequence_length"]
    X_train_seq = create_sequences(X_train_norm, seq_len)
    X_val_seq = create_sequences(X_val_norm, seq_len)
    X_test_seq = create_sequences(X_test_norm, seq_len)
    y_test_seq = y_test[seq_len - 1:][:len(X_test_seq)]
    types_test_seq = types_test[seq_len - 1:][:len(X_test_seq)]

    all_seed_results = []
    all_roc_data = {}
    all_recon_errors = {"normal": [], "attack": []}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n  --- Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---", flush=True)
        seed_start = time.time()
        set_all_seeds(seed)
        seed_results = {}

        # ---- 1. Threshold baseline ----
        print(f"    Training THRESH...", flush=True)
        t0 = time.time()
        y_pred_thresh, scores_thresh = threshold_detector(
            X_train_norm, X_test_norm,
            n_sigma=CONFIG["baselines"]["threshold"]["n_sigma"]
        )
        thresh_time = time.time() - t0
        overall_thresh = evaluate_detector(y_test, y_pred_thresh, scores_thresh)
        per_attack_thresh = evaluate_per_attack(y_test, y_pred_thresh,
                                                 types_test, scores_thresh)
        latency_thresh = estimate_detection_latency_per_attack(types_test, y_pred_thresh)
        seed_results["THRESH"] = {
            "overall": overall_thresh, "per_attack": per_attack_thresh,
            "detection_latency": latency_thresh, "train_time_s": thresh_time,
        }
        print(f"      F1={overall_thresh['f1_score']:.4f}, "
              f"FPR={overall_thresh['fpr']:.4f} ({thresh_time:.1f}s)", flush=True)

        # ---- 2. OC-SVM baseline ----
        print(f"    Training OC-SVM...", flush=True)
        t0 = time.time()
        ocsvm = train_ocsvm(X_train_norm, seed, CONFIG)
        ocsvm_pred_raw = ocsvm.predict(X_test_norm)
        y_pred_ocsvm = (ocsvm_pred_raw == -1).astype(int)
        ocsvm_scores = -ocsvm.decision_function(X_test_norm)
        ocsvm_time = time.time() - t0
        overall_ocsvm = evaluate_detector(y_test, y_pred_ocsvm, ocsvm_scores)
        per_attack_ocsvm = evaluate_per_attack(y_test, y_pred_ocsvm,
                                                types_test, ocsvm_scores)
        latency_ocsvm = estimate_detection_latency_per_attack(types_test, y_pred_ocsvm)
        seed_results["OC-SVM"] = {
            "overall": overall_ocsvm, "per_attack": per_attack_ocsvm,
            "detection_latency": latency_ocsvm, "train_time_s": ocsvm_time,
        }
        print(f"      F1={overall_ocsvm['f1_score']:.4f}, "
              f"FPR={overall_ocsvm['fpr']:.4f} ({ocsvm_time:.1f}s)", flush=True)

        # ---- 3. Isolation Forest baseline ----
        print(f"    Training IF...", flush=True)
        t0 = time.time()
        iforest = train_isolation_forest(X_train_norm, seed, CONFIG)
        if_pred_raw = iforest.predict(X_test_norm)
        y_pred_if = (if_pred_raw == -1).astype(int)
        if_scores = -iforest.decision_function(X_test_norm)
        if_time = time.time() - t0
        overall_if = evaluate_detector(y_test, y_pred_if, if_scores)
        per_attack_if = evaluate_per_attack(y_test, y_pred_if,
                                             types_test, if_scores)
        latency_if = estimate_detection_latency_per_attack(types_test, y_pred_if)
        seed_results["IF"] = {
            "overall": overall_if, "per_attack": per_attack_if,
            "detection_latency": latency_if, "train_time_s": if_time,
        }
        print(f"      F1={overall_if['f1_score']:.4f}, "
              f"FPR={overall_if['fpr']:.4f} ({if_time:.1f}s)", flush=True)

        # ---- 4. LSTM-AE baseline ----
        print(f"    Training LSTM-AE...", flush=True)
        t0 = time.time()
        if len(X_train_seq) > 0 and len(X_test_seq) > 0:
            lstm_model = train_lstm_ae(X_train_seq, X_val_seq, seed, CONFIG, verbose=False)
            lstm_model.eval()
            with torch.no_grad():
                test_tensor_lstm = torch.FloatTensor(X_test_seq)
                lstm_output = lstm_model(test_tensor_lstm)
                lstm_errors = torch.mean((test_tensor_lstm - lstm_output) ** 2,
                                         dim=(1, 2)).numpy()
            with torch.no_grad():
                val_seq_tensor = torch.FloatTensor(X_val_seq)
                lstm_val_out = lstm_model(val_seq_tensor)
                lstm_val_errors = torch.mean((val_seq_tensor - lstm_val_out) ** 2,
                                             dim=(1, 2)).numpy()
            lstm_threshold = float(np.percentile(lstm_val_errors, 99))
            y_pred_lstm = (lstm_errors > lstm_threshold).astype(int)
            lstm_time = time.time() - t0
            overall_lstm = evaluate_detector(y_test_seq, y_pred_lstm, lstm_errors)
            per_attack_lstm = evaluate_per_attack(y_test_seq, y_pred_lstm,
                                                   types_test_seq, lstm_errors)
            latency_lstm = estimate_detection_latency_per_attack(types_test_seq, y_pred_lstm)
        else:
            overall_lstm = {"f1_score": 0, "precision": 0, "recall": 0,
                           "fpr": 0, "accuracy": 0}
            per_attack_lstm = {}
            latency_lstm = {}
            lstm_time = 0
        seed_results["LSTM-AE"] = {
            "overall": overall_lstm, "per_attack": per_attack_lstm,
            "detection_latency": latency_lstm, "train_time_s": lstm_time,
        }
        print(f"      F1={overall_lstm['f1_score']:.4f}, "
              f"FPR={overall_lstm.get('fpr', 0):.4f} ({lstm_time:.1f}s)", flush=True)

        # ---- 5. Autoencoder (FP32) ----
        print(f"    Training AE-FP32...", flush=True)
        t0 = time.time()
        ae_model, ae_train_info = train_autoencoder(X_train_norm, X_val_norm,
                                                     seed, CONFIG, verbose=False)
        ae_time = time.time() - t0

        # Calibrate threshold
        ae_threshold, val_errors = calibrate_threshold(
            ae_model, X_val_norm, CONFIG["autoencoder"]["threshold_percentile"]
        )

        # Evaluate FP32
        ae_model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test_norm)
            ae_output = ae_model(test_tensor)
            ae_errors = torch.mean((test_tensor - ae_output) ** 2, dim=1).numpy()

        y_pred_ae = (ae_errors > ae_threshold).astype(int)
        overall_ae = evaluate_detector(y_test, y_pred_ae, ae_errors)
        per_attack_ae = evaluate_per_attack(y_test, y_pred_ae, types_test, ae_errors)
        latency_ae = estimate_detection_latency_per_attack(types_test, y_pred_ae)
        seed_results["AE-FP32"] = {
            "overall": overall_ae, "per_attack": per_attack_ae,
            "detection_latency": latency_ae, "train_time_s": ae_time,
            "threshold": ae_threshold,
            "training_info": {
                "best_epoch": ae_train_info["best_epoch"],
                "final_val_loss": ae_train_info["final_val_loss"],
            },
        }
        print(f"      F1={overall_ae['f1_score']:.4f}, "
              f"FPR={overall_ae['fpr']:.4f} ({ae_time:.1f}s)", flush=True)

        # ---- 6. Autoencoder (INT8 quantized) ----
        print(f"    Quantizing to INT8...", flush=True)
        quant_info = quantize_model_int8(ae_model, X_val_norm[:1000])
        quant_model = quant_info["quantized_model"]
        quant_model.eval()

        # Recalibrate threshold for quantized model
        with torch.no_grad():
            val_tensor = torch.FloatTensor(X_val_norm)
            quant_val_output = quant_model(val_tensor)
            quant_val_errors = torch.mean((val_tensor - quant_val_output) ** 2,
                                          dim=1).numpy()
        quant_threshold = float(np.percentile(
            quant_val_errors, CONFIG["autoencoder"]["threshold_percentile"]))

        with torch.no_grad():
            quant_output = quant_model(test_tensor)
            quant_errors = torch.mean((test_tensor - quant_output) ** 2, dim=1).numpy()

        y_pred_quant = (quant_errors > quant_threshold).astype(int)
        overall_quant = evaluate_detector(y_test, y_pred_quant, quant_errors)
        per_attack_quant = evaluate_per_attack(y_test, y_pred_quant,
                                                types_test, quant_errors)
        latency_quant = estimate_detection_latency_per_attack(types_test, y_pred_quant)
        seed_results["AE-INT8"] = {
            "overall": overall_quant, "per_attack": per_attack_quant,
            "detection_latency": latency_quant,
            "quantization": {
                "fp32_size_kb": quant_info["fp32_model_size_kb"],
                "int8_size_kb": quant_info["int8_model_size_kb"],
                "size_reduction": quant_info["size_reduction_ratio"],
                "fp32_recon_error": quant_info["orig_recon_error_mean"],
                "int8_recon_error": quant_info["quant_recon_error_mean"],
            },
            "threshold": quant_threshold,
        }
        print(f"      F1={overall_quant['f1_score']:.4f}, "
              f"FPR={overall_quant['fpr']:.4f}")
        print(f"      INT8 size: {quant_info['int8_model_size_kb']:.2f} KB "
              f"(vs {quant_info['fp32_model_size_kb']:.2f} KB FP32)", flush=True)

        # Collect figure data (first seed only)
        if seed_idx == 0:
            normal_mask = y_test == 0
            attack_mask = y_test == 1
            all_recon_errors["normal"] = ae_errors[normal_mask].tolist()
            all_recon_errors["attack"] = ae_errors[attack_mask].tolist()
            all_recon_errors["threshold"] = ae_threshold

            for method_name, scores in [
                ("THRESH", scores_thresh), ("OC-SVM", ocsvm_scores),
                ("IF", if_scores), ("AE-FP32", ae_errors),
                ("AE-INT8", quant_errors),
            ]:
                if len(np.unique(y_test)) > 1:
                    from sklearn.metrics import roc_curve, auc
                    fpr_c, tpr_c, _ = roc_curve(y_test, scores)
                    roc_auc_val = float(auc(fpr_c, tpr_c))
                    all_roc_data[method_name] = {
                        "fpr": fpr_c.tolist(), "tpr": tpr_c.tolist(),
                        "auc": roc_auc_val,
                    }

        # Save per-seed results
        seed_elapsed = time.time() - seed_start
        seed_results["seed"] = seed
        seed_results["elapsed_s"] = round(seed_elapsed, 1)
        all_seed_results.append(seed_results)

        with open(RESULTS_DIR / f"results_seed_{seed}.json", "w") as f:
            serializable = json.loads(json.dumps(seed_results, default=str))
            json.dump(serializable, f, indent=2)

        print(f"    Seed {seed} completed in {seed_elapsed:.1f}s", flush=True)

    # ================================================================
    # STEP 6: Aggregate Results
    # ================================================================
    print(f"\n{'='*70}")
    print("[6/7] Aggregating results across seeds...")
    print(f"{'='*70}", flush=True)

    aggregated = aggregate_all_results(all_seed_results, n_params, layers_info,
                                       data_stats, hardware, timestamp)

    aggregated["figure_data"] = {
        "reconstruction_errors": all_recon_errors,
        "roc_curves": all_roc_data,
    }

    with open(RESULTS_DIR / "aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n  Results saved to: {RESULTS_DIR / 'aggregated_results.json'}")

    print(f"\n  Overall results (mean +/- std over {len(seeds)} seeds):")
    print(f"  {'Method':<12} {'Precision':>12} {'Recall':>12} {'F1':>12} {'FPR':>12}")
    print(f"  {'-'*60}")
    for method in ["THRESH", "OC-SVM", "IF", "LSTM-AE", "AE-FP32", "AE-INT8"]:
        if method in aggregated["overall"]:
            m = aggregated["overall"][method]
            print(f"  {method:<12} "
                  f"{m['precision']['mean']:.4f}+/-{m['precision']['std']:.4f} "
                  f"{m['recall']['mean']:.4f}+/-{m['recall']['std']:.4f} "
                  f"{m['f1_score']['mean']:.4f}+/-{m['f1_score']['std']:.4f} "
                  f"{m['fpr']['mean']:.4f}+/-{m['fpr']['std']:.4f}")

    # ================================================================
    # STEP 7: Generate Figures
    # ================================================================
    print(f"\n{'='*70}")
    print("[7/7] Generating paper figures...")
    print(f"{'='*70}", flush=True)

    from generate_figures import generate_all_figures
    generate_all_figures(aggregated)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Pipeline completed in {total_time:.1f} seconds ({total_time/60:.1f} min)")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"{'='*70}")

    return aggregated


def aggregate_all_results(all_seed_results, n_params, layers_info,
                          data_stats, hardware, timestamp):
    """Aggregate results across all seeds."""
    methods_to_report = ["THRESH", "OC-SVM", "IF", "LSTM-AE", "AE-FP32", "AE-INT8"]
    metrics_to_aggregate = ["precision", "recall", "f1_score", "fpr", "accuracy"]

    overall = {}
    for method in methods_to_report:
        method_metrics = {}
        for metric in metrics_to_aggregate:
            values = []
            for sr in all_seed_results:
                if method in sr and "overall" in sr[method]:
                    val = sr[method]["overall"].get(metric)
                    if val is not None:
                        values.append(val)
            if values:
                method_metrics[metric] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values)), 4),
                    "min": round(float(np.min(values)), 4),
                    "max": round(float(np.max(values)), 4),
                    "values": [round(v, 4) for v in values],
                }
        if method_metrics:
            overall[method] = method_metrics

    per_attack = {}
    for method in methods_to_report:
        per_attack[method] = {}
        for attack_type in ATTACK_NAMES:
            f1_values = []
            for sr in all_seed_results:
                if method in sr and "per_attack" in sr[method]:
                    pa = sr[method]["per_attack"]
                    if attack_type in pa and "f1_score" in pa[attack_type]:
                        f1_values.append(pa[attack_type]["f1_score"])
            if f1_values:
                per_attack[method][attack_type] = {
                    "f1_mean": round(float(np.mean(f1_values)), 4),
                    "f1_std": round(float(np.std(f1_values)), 4),
                }

    detection_latency = {}
    for attack_type in ATTACK_NAMES:
        latency_values = []
        for sr in all_seed_results:
            if "AE-INT8" in sr and "detection_latency" in sr["AE-INT8"]:
                dl = sr["AE-INT8"]["detection_latency"]
                if attack_type in dl:
                    latency_values.append(dl[attack_type]["mean_ms"])
        if latency_values:
            detection_latency[attack_type] = {
                "mean_ms": round(float(np.mean(latency_values)), 1),
                "std_ms": round(float(np.std(latency_values)), 1),
            }

    quant_data = [sr["AE-INT8"]["quantization"] for sr in all_seed_results
                  if "AE-INT8" in sr and "quantization" in sr["AE-INT8"]]
    quant_analysis = {}
    if quant_data:
        quant_analysis = {
            "fp32_size_kb": quant_data[0]["fp32_size_kb"],
            "int8_size_kb": quant_data[0]["int8_size_kb"],
            "size_reduction": quant_data[0]["size_reduction"],
            "fp32_f1": overall.get("AE-FP32", {}).get("f1_score", {}).get("mean", 0),
            "int8_f1": overall.get("AE-INT8", {}).get("f1_score", {}).get("mean", 0),
            "f1_delta": round(
                overall.get("AE-FP32", {}).get("f1_score", {}).get("mean", 0) -
                overall.get("AE-INT8", {}).get("f1_score", {}).get("mean", 0), 4),
        }

    int8_resources = estimate_embedded_resources(n_params, layers_info, "int8")
    fp32_resources = estimate_embedded_resources(n_params, layers_info, "fp32")

    train_times = {}
    for method in methods_to_report:
        times = [sr[method]["train_time_s"] for sr in all_seed_results
                 if method in sr and "train_time_s" in sr[method]]
        if times:
            train_times[method] = {
                "mean_s": round(float(np.mean(times)), 1),
                "std_s": round(float(np.std(times)), 1),
            }

    return {
        "experiment_name": CONFIG["experiment_name"],
        "paper_slug": CONFIG["paper_slug"],
        "timestamp": timestamp,
        "seeds": CONFIG["random_seeds"],
        "n_seeds": len(CONFIG["random_seeds"]),
        "hardware": hardware,
        "dataset": data_stats,
        "model": {
            "architecture": CONFIG["autoencoder"]["architecture"],
            "total_parameters": n_params,
            "layers": layers_info,
            "input_dim": layers_info[0]["in_features"] if layers_info else 80,
            "latent_dim": CONFIG["autoencoder"]["latent_dim"],
        },
        "overall": overall,
        "per_attack": per_attack,
        "detection_latency": detection_latency,
        "quantization": quant_analysis,
        "embedded_resources": {
            "int8": int8_resources,
            "fp32": fp32_resources,
        },
        "training_times": train_times,
        "attack_instances_per_type": CONFIG["attack_injection"]["instances_per_attack_type"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAN Bus Anomaly Detection Experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test with minimal data")
    args = parser.parse_args()

    results = run_pipeline(dry_run=args.dry_run)
