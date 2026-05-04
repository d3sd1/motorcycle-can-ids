#!/usr/bin/env python3
"""
Evaluate all anomaly detection methods on the CAN bus test set.

Computes per-attack and overall metrics, quantization analysis,
and embedded resource estimation.

Paper: Lightweight Autoencoder-Based Anomaly Detection for CAN Bus
       in Competition Motorcycles Deployed on ARM Cortex-M7
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_curve, auc, confusion_matrix)
from typing import Dict, List, Tuple

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

RESULTS_DIR = Path(__file__).parent / CONFIG["output"]["results_dir"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_NAMES = {
    "A1_tps_spoofing": "A1: TPS Spoofing",
    "A2_lean_injection": "A2: Lean Angle Injection",
    "A3_bms_disappearance": "A3: BMS Disappearance",
    "A4_replay": "A4: Replay",
    "A5_fuzzing": "A5: Fuzzing",
    "A6_dos_flooding": "A6: DoS Flooding",
}

METHODS = ["THRESH", "OC-SVM", "IF", "LSTM-AE", "AE-FP32", "AE-INT8"]


def evaluate_detector(y_true: np.ndarray, y_pred: np.ndarray,
                      scores: np.ndarray = None) -> Dict:
    """Compute detection metrics."""
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

    result = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "fpr": fpr,
        "accuracy": accuracy,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    # ROC curve if scores provided
    if scores is not None and len(np.unique(y_true)) > 1:
        fpr_curve, tpr_curve, thresholds = roc_curve(y_true, scores)
        roc_auc = float(auc(fpr_curve, tpr_curve))
        result["roc_auc"] = roc_auc
        result["roc_fpr"] = fpr_curve.tolist()
        result["roc_tpr"] = tpr_curve.tolist()

    return result


def evaluate_per_attack(y_true: np.ndarray, y_pred: np.ndarray,
                        attack_type_labels: np.ndarray,
                        scores: np.ndarray = None) -> Dict:
    """Evaluate detection performance per attack type."""
    results = {}
    for attack_type in ATTACK_NAMES:
        # Select windows that are either normal or contain this attack type
        mask = (attack_type_labels == attack_type) | (y_true == 0)
        if mask.sum() > 0 and np.sum(y_true[mask] == 1) > 0:
            attack_scores = scores[mask] if scores is not None else None
            metrics = evaluate_detector(y_true[mask], y_pred[mask], attack_scores)
            results[attack_type] = metrics
    return results


def threshold_detector(X_train: np.ndarray, X_test: np.ndarray,
                       n_sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Threshold-based anomaly detector (baseline).

    Computes per-feature z-scores against training distribution.
    Uses the 99th percentile of training set anomaly scores as threshold
    to calibrate FPR around 1%.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0

    # Z-scores per feature for train and test
    z_train = np.abs((X_train - mean) / std)
    z_test = np.abs((X_test - mean) / std)

    # Anomaly score: mean z-score across features (more robust than max)
    scores_train = np.mean(z_train, axis=1)
    scores_test = np.mean(z_test, axis=1)

    # Calibrate threshold as 99th percentile of training scores
    threshold = float(np.percentile(scores_train, 99))

    y_pred = (scores_test > threshold).astype(int)

    return y_pred, scores_test


def estimate_embedded_resources(model_params: int, layers: List[Dict],
                                precision: str = "int8") -> Dict:
    """Estimate STM32H7 resource usage for the model.

    Based on CMSIS-NN benchmarks for ARM Cortex-M7 @ 480 MHz.
    """
    bytes_per_param = 1 if precision == "int8" else 4

    # Model weights in Flash
    model_weights_bytes = model_params * bytes_per_param
    # Per-layer scale/zero-point (negligible but count them)
    n_layers = len(layers)
    scale_bytes = n_layers * 2 * 4  # 2 float32 per layer (scale, zp)
    model_flash_bytes = model_weights_bytes + scale_bytes

    # Runtime RAM: largest activation + input/output buffers
    max_activation = max(l["out_features"] for l in layers) if layers else 80
    activation_bytes = max_activation * bytes_per_param

    # CAN message circular buffer (100ms window, ~100 messages, 16 bytes each)
    can_buffer_bytes = 100 * 16

    # Feature vector (always float32)
    feature_dim = layers[0]["in_features"] if layers else 80
    feature_bytes = feature_dim * 4

    # Output vector
    output_bytes = feature_dim * 4

    total_ram_bytes = (model_weights_bytes + activation_bytes +
                       can_buffer_bytes + feature_bytes + output_bytes)

    # MAC operations per inference
    total_macs = sum(l["macs"] for l in layers)

    # CMSIS-NN on Cortex-M7:
    # - INT8: ~2 cycles/MAC (using DSP SIMD instructions, 4x8-bit in 32-bit register)
    # - FP32: ~4 cycles/MAC (using FPU)
    cycles_per_mac = 2 if precision == "int8" else 4
    total_cycles = total_macs * cycles_per_mac

    # Add overhead for ReLU activations (~1 cycle per element per layer)
    relu_cycles = sum(l["out_features"] for l in layers)
    total_cycles += relu_cycles

    clock_hz = CONFIG["embedded"]["clock_mhz"] * 1e6
    inference_latency_ms = (total_cycles / clock_hz) * 1000

    # Feature extraction latency estimate:
    # Parsing 100ms of CAN traffic (~100 frames) + computing features
    # Dominated by entropy computation and correlation
    # Estimate ~50,000 cycles for feature extraction
    feature_extraction_cycles = 50000
    feature_latency_ms = (feature_extraction_cycles / clock_hz) * 1000

    total_latency_ms = inference_latency_ms + feature_latency_ms

    # CAN bus scheduling cycle is typically 10ms
    # IDS gets 50% of CPU budget in each cycle
    cpu_budget_ms = 10 * 0.5  # 5ms budget
    cpu_utilization_pct = (total_latency_ms / cpu_budget_ms) * 100

    return {
        "model_weights_flash_bytes": model_flash_bytes,
        "model_weights_flash_kb": round(model_flash_bytes / 1024, 2),
        "runtime_ram_bytes": total_ram_bytes,
        "runtime_ram_kb": round(total_ram_bytes / 1024, 2),
        "total_macs": total_macs,
        "inference_latency_ms": round(inference_latency_ms, 4),
        "feature_extraction_latency_ms": round(feature_latency_ms, 4),
        "total_detection_latency_ms": round(total_latency_ms, 4),
        "cpu_utilization_pct": round(cpu_utilization_pct, 2),
        "fits_ram_budget": total_ram_bytes < CONFIG["embedded"]["ids_ram_budget_kb"] * 1024,
        "meets_latency_target": total_latency_ms < CONFIG["embedded"]["latency_target_ms"],
        "max_activation_size": max_activation,
        "bytes_per_param": bytes_per_param,
    }


def estimate_detection_latency_per_attack(attack_types: np.ndarray,
                                          y_pred: np.ndarray,
                                          window_duration_ms: float = 100,
                                          overlap_ratio: float = 0.5) -> Dict:
    """Estimate detection latency for each attack type.

    Detection latency = number of windows from attack start to first detection
    multiplied by window stride.
    """
    stride_ms = window_duration_ms * (1 - overlap_ratio)
    results = {}

    for attack_type in ATTACK_NAMES:
        attack_mask = attack_types == attack_type
        if not np.any(attack_mask):
            continue

        # Find contiguous attack segments
        attack_indices = np.where(attack_mask)[0]
        if len(attack_indices) == 0:
            continue

        # Split into segments (gaps > 2 windows = new segment)
        segments = []
        current_segment = [attack_indices[0]]
        for i in range(1, len(attack_indices)):
            if attack_indices[i] - attack_indices[i-1] <= 2:
                current_segment.append(attack_indices[i])
            else:
                segments.append(current_segment)
                current_segment = [attack_indices[i]]
        segments.append(current_segment)

        latencies = []
        for segment in segments:
            # Find first detection within this segment
            detected = False
            for idx in segment:
                if y_pred[idx] == 1:
                    latency = (idx - segment[0]) * stride_ms + window_duration_ms
                    latencies.append(latency)
                    detected = True
                    break
            if not detected:
                # Not detected: latency = full segment duration
                latencies.append((len(segment)) * stride_ms + window_duration_ms)

        if latencies:
            results[attack_type] = {
                "mean_ms": round(float(np.mean(latencies)), 1),
                "std_ms": round(float(np.std(latencies)), 1),
                "min_ms": round(float(np.min(latencies)), 1),
                "max_ms": round(float(np.max(latencies)), 1),
                "n_segments": len(segments),
            }

    return results


if __name__ == "__main__":
    from train import CANAutoencoder

    print("=" * 60)
    print("CAN Bus Anomaly Detection -- Resource Estimation")
    print("=" * 60)

    model = CANAutoencoder(80, 10)
    n_params = model.count_parameters()
    layers = model.get_layer_info()

    print(f"\nModel: {n_params} parameters")
    for l in layers:
        print(f"  {l['name']}: {l['in_features']}x{l['out_features']} "
              f"= {l['params']} params, {l['macs']} MACs")

    print(f"\nINT8 deployment:")
    int8 = estimate_embedded_resources(n_params, layers, "int8")
    for k, v in int8.items():
        print(f"  {k}: {v}")

    print(f"\nFP32 deployment:")
    fp32 = estimate_embedded_resources(n_params, layers, "fp32")
    for k, v in fp32.items():
        print(f"  {k}: {v}")
