#!/usr/bin/env python3
"""
One-command orchestrator for all experiments.

Runs the complete pipeline:
1. Data loading and CAN traffic reconstruction
2. Temporal split of CAN frames by session (60/20/20)
3. Feature extraction from train/val normal traffic
4. Per-seed: frame-level attack injection into test traffic
5. Feature extraction from attacked test traffic
6. Model training (autoencoder + baselines, 5 seeds)
7. Evaluation (overall + per-attack + quantization + embedded)
8. Results aggregation and figure generation

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
import copy
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

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
from hybrid_detector import (learn_rule_params, build_test_windows,
                            rule_layer_predict, rule_memory_cost)

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


def split_sessions_temporal(data_stats: dict, all_frames: list,
                            train_ratio: float = 0.6,
                            val_ratio: float = 0.2) -> tuple:
    """Split CAN frames by session boundaries (stratified shuffle).

    Shuffles sessions with a fixed seed so each split gets a mix of
    session types (bench, track, qualifying), then assigns to
    train/val/test.  No within-session leakage.
    """
    session_details = data_stats["session_details"]
    n_sessions = len(session_details)

    n_train = max(1, int(n_sessions * train_ratio))
    n_val = max(1, int(n_sessions * val_ratio))
    n_test = n_sessions - n_train - n_val
    if n_test < 1:
        n_val = max(1, n_val - 1)
        n_test = n_sessions - n_train - n_val

    # Compute per-session frame slices in the original order
    cum_frames = 0
    session_slices = []
    for sd in session_details:
        start = cum_frames
        cum_frames += sd["n_frames"]
        session_slices.append((start, cum_frames))

    # Shuffle session indices (fixed seed for reproducibility across runs)
    indices = list(range(n_sessions))
    rng = np.random.default_rng(seed=0)
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_frames = []
    for i in sorted(train_idx):
        s, e = session_slices[i]
        train_frames.extend(all_frames[s:e])

    val_frames = []
    for i in sorted(val_idx):
        s, e = session_slices[i]
        val_frames.extend(all_frames[s:e])

    test_frames = []
    for i in sorted(test_idx):
        s, e = session_slices[i]
        test_frames.extend(all_frames[s:e])

    train_names = [session_details[i]["file"] for i in sorted(train_idx)]
    val_names = [session_details[i]["file"] for i in sorted(val_idx)]
    test_names = [session_details[i]["file"] for i in sorted(test_idx)]

    split_info = {
        "n_sessions_train": n_train,
        "n_sessions_val": n_val,
        "n_sessions_test": n_test,
        "n_frames_train": len(train_frames),
        "n_frames_val": len(val_frames),
        "n_frames_test": len(test_frames),
        "train_sessions": train_names,
        "val_sessions": val_names,
        "test_sessions": test_names,
    }

    return train_frames, val_frames, test_frames, split_info


def clean_nans(X: np.ndarray, name: str) -> np.ndarray:
    n_nan = int(np.sum(np.isnan(X)))
    n_inf = int(np.sum(np.isinf(X)))
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING: {name} has {n_nan} NaN and {n_inf} Inf. Replacing with 0.")
        X[np.isnan(X)] = 0.0
        X[np.isinf(X)] = 0.0
    return X


def run_pipeline(dry_run: bool = False):
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("CAN Bus Anomaly Detection -- Full Experiment Pipeline v2")
    print("  Frame-level attack injection + per-seed test variation")
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
    print("[1/8] Loading CAN traffic data from testbed...")
    print(f"{'='*70}", flush=True)

    max_sessions = 3 if dry_run else 20
    all_frames, data_stats = load_all_sessions(max_sessions=max_sessions)

    if not all_frames:
        print("ERROR: No CAN frames loaded. Check dataset path in config.json.")
        sys.exit(1)

    # Fix total_duration_s: sum of per-session durations, not merged range
    total_duration_s = sum(sd["duration_s"] for sd in data_stats["session_details"])
    data_stats["total_duration_s"] = round(total_duration_s, 1)

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
    # STEP 2: Temporal split by session (60/20/20)
    # ================================================================
    print(f"\n{'='*70}")
    print("[2/8] Splitting CAN frames by session (60/20/20 shuffled)...")
    print(f"{'='*70}", flush=True)

    train_frames, val_frames, test_frames_normal, split_info = split_sessions_temporal(
        data_stats, all_frames,
        CONFIG["data"]["train_ratio"], CONFIG["data"]["val_ratio"]
    )

    print(f"  Train: {split_info['n_sessions_train']} sessions, "
          f"{split_info['n_frames_train']:,} frames")
    if "train_sessions" in split_info:
        for s in split_info["train_sessions"]:
            print(f"    - {s}")
    print(f"  Val:   {split_info['n_sessions_val']} sessions, "
          f"{split_info['n_frames_val']:,} frames")
    if "val_sessions" in split_info:
        for s in split_info["val_sessions"]:
            print(f"    - {s}")
    print(f"  Test:  {split_info['n_sessions_test']} sessions, "
          f"{split_info['n_frames_test']:,} frames")
    if "test_sessions" in split_info:
        for s in split_info["test_sessions"]:
            print(f"    - {s}")
    sys.stdout.flush()

    # ================================================================
    # STEP 3: Feature extraction from TRAIN and VAL (normal only, once)
    # ================================================================
    print(f"\n{'='*70}")
    print("[3/8] Extracting features from train/val normal traffic...")
    print(f"{'='*70}", flush=True)

    print("  Extracting train features...")
    X_train_raw, y_train, _ = extract_dataset_features(train_frames, MONITORED_IDS)
    assert np.all(y_train == 0), "Train data should be all normal"

    print("  Extracting val features...")
    X_val_raw, y_val, _ = extract_dataset_features(val_frames, MONITORED_IDS)
    assert np.all(y_val == 0), "Val data should be all normal"

    print(f"  Train windows: {X_train_raw.shape[0]}")
    print(f"  Val windows: {X_val_raw.shape[0]}")
    print(f"  Feature dimension: {X_train_raw.shape[1]}", flush=True)

    # Compute normalization from training data (once)
    X_train_norm, X_val_norm, feat_min, feat_max = normalize_features(
        X_train_raw, X_val_raw
    )
    X_train_norm = clean_nans(X_train_norm, "X_train")
    X_val_norm = clean_nans(X_val_norm, "X_val")

    # Normalization stats for reuse
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0

    # ----------------------------------------------------------------
    # Learn hybrid rule-layer parameters from NORMAL training traffic.
    # Seed-independent (train set is fixed), so computed once.
    # ----------------------------------------------------------------
    hb_cfg = CONFIG.get("hybrid", {})
    HB_K = hb_cfg.get("heartbeat_k", 3)
    FC_N = hb_cfg.get("freq_n_sigma", 5)
    print(f"\n  Learning hybrid rule-layer parameters from train traffic "
          f"(heartbeat k={HB_K}, freq n_sigma={FC_N})...", flush=True)
    rule_params = learn_rule_params(train_frames, MONITORED_IDS)
    print(f"    Expected periods (ms): "
          f"{{{', '.join(f'0x{cid:X}:{p/1000:.1f}' for cid, p in rule_params['periods_us'].items())}}}")
    print(f"    Normal per-window frame count: "
          f"mu={rule_params['global_count_mean']:.1f}, "
          f"sigma={rule_params['global_count_std']:.1f}, "
          f"max={rule_params['global_count_max']:.0f}")
    rule_mem = rule_memory_cost(rule_params)
    print(f"    Rule-layer state: {rule_mem['total_rule_state_bytes']} bytes "
          f"(heartbeat {rule_mem['heartbeat_state_bytes']} + "
          f"freq {rule_mem['freq_counter_state_bytes']})", flush=True)

    # ================================================================
    # STEP 4: Model info
    # ================================================================
    input_dim = X_train_norm.shape[1]
    latent_dim = CONFIG["autoencoder"]["latent_dim"]
    dummy_ae = CANAutoencoder(input_dim, latent_dim)
    n_params = dummy_ae.count_parameters()
    layers_info = dummy_ae.get_layer_info()
    print(f"\n  Autoencoder: {n_params} parameters")
    print(f"  Input dim: {input_dim}, Latent dim: {latent_dim}")
    print(f"  Architecture: {CONFIG['autoencoder']['architecture']}", flush=True)

    # ================================================================
    # STEP 5-6: Per-seed attack injection + training + evaluation
    # ================================================================
    print(f"\n{'='*70}")
    print("[4/8] Per-seed: frame-level injection -> train -> evaluate...")
    print(f"{'='*70}", flush=True)

    n_instances = 5 if dry_run else CONFIG["attack_injection"]["instances_per_attack_type"]
    all_seed_results = []
    all_roc_data = {}
    all_recon_errors = {"normal": [], "attack": []}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n  {'='*60}")
        print(f"  Seed {seed} ({seed_idx + 1}/{len(seeds)})")
        print(f"  {'='*60}", flush=True)
        seed_start = time.time()
        set_all_seeds(seed)
        seed_results = {}

        # ---- Frame-level attack injection into test traffic ----
        print(f"    [a] Injecting attacks at CAN frame level (seed={seed})...")
        test_frames_copy = [CANFrame(f.timestamp_us, f.can_id, f.dlc,
                                     f.data, f.label, f.attack_type)
                           for f in test_frames_normal]

        attacked_frames, injection_log = inject_all_attacks(
            test_frames_copy, seed=seed, n_instances=n_instances
        )
        n_attack_frames = sum(1 for f in attacked_frames if f.label == "attack")
        n_normal_frames = sum(1 for f in attacked_frames if f.label == "normal")
        print(f"      Frames after injection: {len(attacked_frames):,} "
              f"({n_normal_frames:,} normal + {n_attack_frames:,} attack)")
        for atk, info in injection_log.items():
            print(f"        {atk}: {info}", flush=True)

        # ---- Feature extraction from attacked test traffic ----
        print(f"    [b] Extracting features from attacked test traffic...")
        X_test_raw, y_test, types_test = extract_dataset_features(
            attacked_frames, MONITORED_IDS, verbose=False
        )

        # Normalize using training stats
        X_test_norm = (X_test_raw - feat_min) / feat_range
        X_test_norm = clean_nans(X_test_norm, "X_test")

        n_attack_win = int(np.sum(y_test == 1))
        n_normal_win = int(np.sum(y_test == 0))
        print(f"      Test windows: {n_normal_win} normal + {n_attack_win} attack "
              f"= {len(y_test)} total ({n_attack_win/len(y_test)*100:.1f}% attack)")

        # Per-attack window counts
        for atk in ATTACK_NAMES:
            n_atk = int(np.sum(types_test == atk))
            if n_atk > 0:
                print(f"        {atk}: {n_atk} windows")

        # ---- LSTM sequences for this test set ----
        seq_len = CONFIG["baselines"]["lstm_ae"]["sequence_length"]
        X_train_seq = create_sequences(X_train_norm, seq_len)
        X_val_seq = create_sequences(X_val_norm, seq_len)
        X_test_seq = create_sequences(X_test_norm, seq_len)
        y_test_seq = y_test[seq_len - 1:][:len(X_test_seq)]
        types_test_seq = types_test[seq_len - 1:][:len(X_test_seq)]

        # ---- 1. Threshold baseline ----
        print(f"    [c] Training THRESH...", flush=True)
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
        print(f"    [d] Training OC-SVM...", flush=True)
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
        print(f"    [e] Training IF...", flush=True)
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
        print(f"    [f] Training LSTM-AE...", flush=True)
        t0 = time.time()
        if os.environ.get("SKIP_LSTM") == "1":
            print("      [skipped via SKIP_LSTM=1]", flush=True)
        if os.environ.get("SKIP_LSTM") != "1" and len(X_train_seq) > 0 and len(X_test_seq) > 0:
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
        print(f"    [g] Training AE-FP32...", flush=True)
        t0 = time.time()
        ae_model, ae_train_info = train_autoencoder(X_train_norm, X_val_norm,
                                                     seed, CONFIG, verbose=False)
        ae_time = time.time() - t0

        ae_threshold, val_errors = calibrate_threshold(
            ae_model, X_val_norm, CONFIG["autoencoder"]["threshold_percentile"]
        )

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
        print(f"    [h] Quantizing to INT8...", flush=True)
        quant_info = quantize_model_int8(ae_model, X_val_norm[:1000])
        quant_model = quant_info["quantized_model"]
        quant_model.eval()

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

        # ---- 7. HYBRID multi-layer IDS (rules + learned layer) ----
        print(f"    [i] Hybrid rule layers (heartbeat + frequency)...", flush=True)
        t0 = time.time()
        windows_test = build_test_windows(attacked_frames)
        assert len(windows_test) == len(y_test), (
            f"window/label misalignment: {len(windows_test)} vs {len(y_test)}")
        rule_out = rule_layer_predict(windows_test, rule_params, k=HB_K, n_sigma=FC_N)
        hb_pred = rule_out["heartbeat"]
        fc_pred = rule_out["freq"]
        rule_pred = rule_out["rules"]
        rule_time = time.time() - t0

        # --- HYBRID = rules OR AE-INT8 (primary proposal) ---
        y_pred_hybrid = ((y_pred_quant == 1) | (rule_pred == 1)).astype(int)
        hybrid_scores = quant_errors / max(quant_threshold, 1e-12) + rule_pred.astype(float)
        overall_hybrid = evaluate_detector(y_test, y_pred_hybrid, hybrid_scores)
        per_attack_hybrid = evaluate_per_attack(y_test, y_pred_hybrid,
                                                types_test, hybrid_scores)
        latency_hybrid = estimate_detection_latency_per_attack(types_test, y_pred_hybrid)

        # Per-layer per-attack recall (diagnostic, for the summary table)
        per_attack_hb = evaluate_per_attack(y_test, hb_pred, types_test)
        per_attack_fc = evaluate_per_attack(y_test, fc_pred, types_test)
        rule_breakdown = {}
        for atk in ATTACK_NAMES:
            rule_breakdown[atk] = {
                "heartbeat_recall": per_attack_hb.get(atk, {}).get("recall", 0.0),
                "freq_recall": per_attack_fc.get(atk, {}).get("recall", 0.0),
            }

        seed_results["HYBRID"] = {
            "overall": overall_hybrid, "per_attack": per_attack_hybrid,
            "detection_latency": latency_hybrid, "train_time_s": rule_time,
            "rule_breakdown": rule_breakdown,
            "rule_thresholds": {
                "heartbeat_k": HB_K, "freq_n_sigma": FC_N,
                "global_count_threshold": round(
                    rule_params["global_count_mean"] + FC_N * rule_params["global_count_std"], 2),
            },
        }
        print(f"      F1={overall_hybrid['f1_score']:.4f}, "
              f"FPR={overall_hybrid['fpr']:.4f} "
              f"(rules: heartbeat+freq, {rule_time:.2f}s)", flush=True)

        # --- HYBRID-OCSVM = rules OR OC-SVM (comparison variant) ---
        y_pred_hybrid_ocsvm = ((y_pred_ocsvm == 1) | (rule_pred == 1)).astype(int)
        oc_ptp = float(np.ptp(ocsvm_scores)) if np.ptp(ocsvm_scores) > 0 else 1.0
        hybrid_ocsvm_scores = (ocsvm_scores - float(np.min(ocsvm_scores))) / oc_ptp + rule_pred.astype(float)
        overall_hybrid_oc = evaluate_detector(y_test, y_pred_hybrid_ocsvm, hybrid_ocsvm_scores)
        per_attack_hybrid_oc = evaluate_per_attack(y_test, y_pred_hybrid_ocsvm,
                                                   types_test, hybrid_ocsvm_scores)
        latency_hybrid_oc = estimate_detection_latency_per_attack(types_test, y_pred_hybrid_ocsvm)
        seed_results["HYBRID-OCSVM"] = {
            "overall": overall_hybrid_oc, "per_attack": per_attack_hybrid_oc,
            "detection_latency": latency_hybrid_oc, "train_time_s": ocsvm_time + rule_time,
        }
        print(f"      [HYBRID-OCSVM] F1={overall_hybrid_oc['f1_score']:.4f}, "
              f"FPR={overall_hybrid_oc['fpr']:.4f}", flush=True)

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
                ("AE-INT8", quant_errors), ("HYBRID", hybrid_scores),
            ]:
                if len(np.unique(y_test)) > 1:
                    from sklearn.metrics import roc_curve, auc
                    fpr_c, tpr_c, _ = roc_curve(y_test, scores)
                    roc_auc_val = float(auc(fpr_c, tpr_c))
                    all_roc_data[method_name] = {
                        "fpr": fpr_c.tolist(), "tpr": tpr_c.tolist(),
                        "auc": roc_auc_val,
                    }

        seed_elapsed = time.time() - seed_start
        seed_results["seed"] = seed
        seed_results["elapsed_s"] = round(seed_elapsed, 1)
        all_seed_results.append(seed_results)

        with open(RESULTS_DIR / f"results_seed_{seed}.json", "w") as f:
            serializable = json.loads(json.dumps(seed_results, default=str))
            json.dump(serializable, f, indent=2)

        print(f"    Seed {seed} completed in {seed_elapsed:.1f}s", flush=True)

    # ================================================================
    # STEP 7: Aggregate Results
    # ================================================================
    print(f"\n{'='*70}")
    print("[7/8] Aggregating results across seeds...")
    print(f"{'='*70}", flush=True)

    aggregated = aggregate_all_results(all_seed_results, n_params, layers_info,
                                       data_stats, hardware, timestamp,
                                       split_info, rule_params)

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
    for method in ["THRESH", "OC-SVM", "IF", "LSTM-AE", "AE-FP32", "AE-INT8",
                   "HYBRID", "HYBRID-OCSVM"]:
        if method in aggregated["overall"]:
            m = aggregated["overall"][method]
            print(f"  {method:<12} "
                  f"{m['precision']['mean']:.4f}+/-{m['precision']['std']:.4f} "
                  f"{m['recall']['mean']:.4f}+/-{m['recall']['std']:.4f} "
                  f"{m['f1_score']['mean']:.4f}+/-{m['f1_score']['std']:.4f} "
                  f"{m['fpr']['mean']:.4f}+/-{m['fpr']['std']:.4f}")

    # ================================================================
    # STEP 8: Generate Figures
    # ================================================================
    print(f"\n{'='*70}")
    print("[8/8] Generating paper figures...")
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
                          data_stats, hardware, timestamp, split_info,
                          rule_params):
    methods_to_report = ["THRESH", "OC-SVM", "IF", "LSTM-AE", "AE-FP32",
                         "AE-INT8", "HYBRID", "HYBRID-OCSVM"]
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
            precision_values = []
            recall_values = []
            for sr in all_seed_results:
                if method in sr and "per_attack" in sr[method]:
                    pa = sr[method]["per_attack"]
                    if attack_type in pa:
                        if "f1_score" in pa[attack_type]:
                            f1_values.append(pa[attack_type]["f1_score"])
                        if "precision" in pa[attack_type]:
                            precision_values.append(pa[attack_type]["precision"])
                        if "recall" in pa[attack_type]:
                            recall_values.append(pa[attack_type]["recall"])
            if f1_values:
                per_attack[method][attack_type] = {
                    "f1_mean": round(float(np.mean(f1_values)), 4),
                    "f1_std": round(float(np.std(f1_values)), 4),
                    "precision_mean": round(float(np.mean(precision_values)), 4) if precision_values else 0,
                    "recall_mean": round(float(np.mean(recall_values)), 4) if recall_values else 0,
                }

    def aggregate_latency(method):
        out = {}
        for attack_type in ATTACK_NAMES:
            latency_values = []
            for sr in all_seed_results:
                if method in sr and "detection_latency" in sr[method]:
                    dl = sr[method]["detection_latency"]
                    if attack_type in dl:
                        latency_values.append(dl[attack_type]["mean_ms"])
            if latency_values:
                out[attack_type] = {
                    "mean_ms": round(float(np.mean(latency_values)), 1),
                    "std_ms": round(float(np.std(latency_values)), 1),
                }
        return out

    # detection_latency reports the deployed detector (HYBRID); the single
    # AE-INT8 latency is kept under detection_latency_ae_int8 for the
    # before/after comparison.
    detection_latency = aggregate_latency("HYBRID")
    detection_latency_ae_int8 = aggregate_latency("AE-INT8")

    # Per-layer per-attack recall breakdown (mean over seeds), for the
    # response letter / summary: how much each rule contributes.
    rule_breakdown = {}
    for attack_type in ATTACK_NAMES:
        hb_vals, fc_vals = [], []
        for sr in all_seed_results:
            rb = sr.get("HYBRID", {}).get("rule_breakdown", {})
            if attack_type in rb:
                hb_vals.append(rb[attack_type]["heartbeat_recall"])
                fc_vals.append(rb[attack_type]["freq_recall"])
        if hb_vals:
            rule_breakdown[attack_type] = {
                "heartbeat_recall_mean": round(float(np.mean(hb_vals)), 4),
                "freq_recall_mean": round(float(np.mean(fc_vals)), 4),
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

    # Rule-layer (heartbeat + frequency counter) embedded cost
    rule_resources = rule_memory_cost(rule_params)
    # Hybrid total = INT8 learned layer + tiny rule state
    hybrid_resources = dict(int8_resources)
    hybrid_resources["rule_layers"] = rule_resources
    hybrid_resources["hybrid_total_ram_bytes"] = (
        int8_resources["runtime_ram_bytes"] + rule_resources["total_rule_state_bytes"])
    hybrid_resources["hybrid_total_ram_kb"] = round(
        hybrid_resources["hybrid_total_ram_bytes"] / 1024, 2)
    hybrid_resources["rule_overhead_ram_pct"] = round(
        100.0 * rule_resources["total_rule_state_bytes"]
        / int8_resources["runtime_ram_bytes"], 3)

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
        "split": split_info,
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
        "detection_latency_ae_int8": detection_latency_ae_int8,
        "rule_breakdown": rule_breakdown,
        "hybrid_config": {
            "heartbeat_k": CONFIG.get("hybrid", {}).get("heartbeat_k", 3),
            "freq_n_sigma": CONFIG.get("hybrid", {}).get("freq_n_sigma", 5),
            "unknown_id_min_count": CONFIG.get("hybrid", {}).get("unknown_id_min_count", 3),
            "reset_gap_us": CONFIG.get("hybrid", {}).get("reset_gap_us", 1000000),
            "window_duration_ms": CONFIG["feature_extraction"]["window_duration_ms"],
            "window_stride_ms": CONFIG["feature_extraction"]["window_duration_ms"]
                                * (1 - CONFIG["feature_extraction"]["window_overlap_ratio"]),
            "fusion": "OR",
            "expected_periods_ms": {f"0x{cid:X}": round(p / 1000, 2)
                                    for cid, p in rule_params["periods_us"].items()},
            "global_count_mean": round(rule_params["global_count_mean"], 2),
            "global_count_std": round(rule_params["global_count_std"], 2),
            "global_count_threshold": round(
                rule_params["global_count_mean"]
                + CONFIG.get("hybrid", {}).get("freq_n_sigma", 5)
                * rule_params["global_count_std"], 2),
        },
        "quantization": quant_analysis,
        "embedded_resources": {
            "int8": int8_resources,
            "fp32": fp32_resources,
            "rule_layers": rule_resources,
            "hybrid": hybrid_resources,
        },
        "training_times": train_times,
        "attack_instances_per_type": CONFIG["attack_injection"]["instances_per_attack_type"],
        "attack_injection_level": "frame-level",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAN Bus Anomaly Detection Experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test with minimal data")
    args = parser.parse_args()

    results = run_pipeline(dry_run=args.dry_run)
