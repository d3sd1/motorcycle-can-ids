#!/usr/bin/env python3
"""
Feature extraction from CAN traffic using sliding windows.

Extracts timing, payload, and cross-signal features for each window.
Feature vector dimension: 13*K + 2 where K = number of monitored CAN IDs.

Paper: Lightweight Autoencoder-Based Anomaly Detection for CAN Bus
       in Competition Motorcycles Deployed on ARM Cortex-M7
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

# Import CANFrame from attack_injection (canonical definition)
from attack_injection import CANFrame

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

FE_CONFIG = CONFIG["feature_extraction"]

# Monitored CAN IDs (6 IDs matching what data_loader generates)
# VCU(0x100), Motor Speed(0x200), Motor Temp(0x201), BMS Pack(0x300), BMS SOC(0x301), IMU(0x400)
MONITORED_IDS = [0x100, 0x200, 0x201, 0x300, 0x301, 0x400]


def extract_timing_features(timestamps_us: np.ndarray) -> np.ndarray:
    """Extract timing features for frames of a specific CAN ID.

    Returns: [mean_iat, std_iat, message_count] (3 features)
    """
    if len(timestamps_us) < 2:
        return np.array([0.0, 0.0, float(len(timestamps_us))])

    iats = np.diff(timestamps_us).astype(np.float64)

    return np.array([
        np.mean(iats),
        np.std(iats) if len(iats) > 1 else 0.0,
        float(len(timestamps_us)),
    ])


def extract_payload_features(payloads: np.ndarray) -> np.ndarray:
    """Extract payload features for frames of a specific CAN ID.

    Args:
        payloads: shape (n_frames, 8) with byte values 0-255

    Returns: [mean_byte_0..7, entropy, max_change_rate] (10 features)
    """
    if len(payloads) < 2:
        return np.zeros(10)

    # Mean per byte position (8 features)
    mean_bytes = np.mean(payloads, axis=0)

    # Payload entropy across all bytes
    flat = payloads.flatten().astype(int)
    counts = np.bincount(flat, minlength=256)
    total = counts.sum()
    if total > 0:
        probs = counts / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
    else:
        entropy = 0.0

    # Max change rate (max absolute difference between consecutive frames)
    diffs = np.abs(np.diff(payloads, axis=0))
    max_change = float(np.max(diffs)) if diffs.size > 0 else 0.0

    return np.concatenate([mean_bytes, [entropy, max_change]])


def extract_cross_signal_features(frames_by_id: dict) -> np.ndarray:
    """Extract cross-signal correlation features.

    Returns: [throttle_current_corr, lean_gyro_consistency] (2 features)
    """
    # Feature 1: Correlation between throttle (0x100) and motor current (0x200)
    throttle_corr = 0.0
    vcu_frames = frames_by_id.get(0x100, [])
    motor_frames = frames_by_id.get(0x200, [])
    if len(vcu_frames) >= 3 and len(motor_frames) >= 3:
        # Extract throttle from byte 0-1 and motor current from byte 2-3
        thr_vals = np.array([int.from_bytes(f.data[0:2], 'little', signed=False)
                            for f in vcu_frames], dtype=np.float64)
        cur_vals = np.array([int.from_bytes(f.data[2:4], 'little', signed=True)
                            for f in motor_frames[:len(thr_vals)]], dtype=np.float64)
        min_len = min(len(thr_vals), len(cur_vals))
        if min_len >= 3 and np.std(thr_vals[:min_len]) > 0 and np.std(cur_vals[:min_len]) > 0:
            throttle_corr = float(np.corrcoef(thr_vals[:min_len], cur_vals[:min_len])[0, 1])
            if np.isnan(throttle_corr):
                throttle_corr = 0.0

    # Feature 2: Lean angle rate consistency (byte 0-1 vs byte 2-3 of IMU)
    lean_consistency = 0.0
    imu_frames = frames_by_id.get(0x400, [])
    if len(imu_frames) >= 3:
        lean_vals = np.array([int.from_bytes(f.data[0:2], 'little', signed=True)
                             for f in imu_frames], dtype=np.float64)
        rate_vals = np.array([int.from_bytes(f.data[2:4], 'little', signed=True)
                             for f in imu_frames], dtype=np.float64)
        # Check if lean rate is consistent with lean angle derivative
        if len(lean_vals) >= 3:
            lean_deriv = np.diff(lean_vals)
            rate_compare = rate_vals[1:]
            min_len = min(len(lean_deriv), len(rate_compare))
            if min_len >= 2 and np.std(lean_deriv[:min_len]) > 0 and np.std(rate_compare[:min_len]) > 0:
                lean_consistency = float(np.corrcoef(lean_deriv[:min_len],
                                                     rate_compare[:min_len])[0, 1])
                if np.isnan(lean_consistency):
                    lean_consistency = 0.0

    return np.array([throttle_corr, lean_consistency])


def extract_window_features(frames: List[CANFrame],
                            monitored_ids: List[int]) -> np.ndarray:
    """Extract full feature vector from a window of CAN frames.

    Returns: feature vector of dimension 13*K + 2
    """
    # Group frames by CAN ID
    frames_by_id = defaultdict(list)
    for f in frames:
        frames_by_id[f.can_id].append(f)

    features = []
    for can_id in monitored_ids:
        id_frames = frames_by_id.get(can_id, [])

        # Timing features (3)
        if id_frames:
            timestamps = np.array([f.timestamp_us for f in id_frames], dtype=np.float64)
            timing = extract_timing_features(timestamps)
        else:
            timing = np.zeros(3)

        # Payload features (10)
        if id_frames:
            payloads = np.array([list(f.data[:8]) for f in id_frames], dtype=np.float64)
            payload = extract_payload_features(payloads)
        else:
            payload = np.zeros(10)

        features.extend(timing)
        features.extend(payload)

    # Cross-signal features (2)
    cross = extract_cross_signal_features(frames_by_id)
    features.extend(cross)

    return np.array(features, dtype=np.float64)


def create_sliding_windows(frames: List[CANFrame],
                           window_duration_us: int,
                           overlap_ratio: float = 0.5
                           ) -> List[Tuple[List[CANFrame], str, str]]:
    """Create sliding windows from a sequence of CAN frames.

    Returns: list of (window_frames, label, attack_type) tuples
    """
    if not frames or len(frames) < 2:
        return []

    stride_us = int(window_duration_us * (1 - overlap_ratio))
    start_time = frames[0].timestamp_us
    end_time = frames[-1].timestamp_us

    # Pre-sort frames by timestamp for efficient windowing
    sorted_frames = sorted(frames, key=lambda f: f.timestamp_us)

    windows = []
    t = start_time
    frame_idx = 0  # sliding index for efficiency

    while t + window_duration_us <= end_time:
        # Advance frame_idx to the start of current window
        while frame_idx > 0 and sorted_frames[frame_idx].timestamp_us >= t:
            frame_idx -= 1

        while frame_idx < len(sorted_frames) and sorted_frames[frame_idx].timestamp_us < t:
            frame_idx += 1

        # Collect frames in window
        window_frames = []
        j = frame_idx
        while j < len(sorted_frames) and sorted_frames[j].timestamp_us < t + window_duration_us:
            window_frames.append(sorted_frames[j])
            j += 1

        if window_frames:
            # Window label
            has_attack = any(f.label == "attack" for f in window_frames)
            label = "attack" if has_attack else "normal"

            # Most common attack type in window
            attack_types = [f.attack_type for f in window_frames if f.attack_type]
            if attack_types:
                from collections import Counter
                attack_type = Counter(attack_types).most_common(1)[0][0]
            else:
                attack_type = ""

            windows.append((window_frames, label, attack_type))

        t += stride_us

    return windows


def extract_dataset_features(frames: List[CANFrame],
                             monitored_ids: List[int] = None,
                             verbose: bool = True
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from all windows in a frame sequence.

    Returns: (X, y, attack_types) where
        X: (n_windows, n_features) feature matrix
        y: (n_windows,) binary labels (0=normal, 1=attack)
        attack_types: (n_windows,) attack type strings
    """
    if monitored_ids is None:
        monitored_ids = MONITORED_IDS

    window_duration_us = FE_CONFIG["window_duration_ms"] * 1000
    overlap_ratio = FE_CONFIG["window_overlap_ratio"]

    if verbose:
        print(f"  Creating sliding windows (duration={FE_CONFIG['window_duration_ms']}ms, "
              f"overlap={overlap_ratio*100}%)...")

    windows = create_sliding_windows(frames, window_duration_us, overlap_ratio)

    if verbose:
        print(f"  Extracting features from {len(windows)} windows...")

    X = []
    y = []
    types = []

    for i, (window_frames, label, attack_type) in enumerate(windows):
        features = extract_window_features(window_frames, monitored_ids)
        X.append(features)
        y.append(1 if label == "attack" else 0)
        types.append(attack_type)

        if verbose and (i + 1) % 5000 == 0:
            print(f"    Processed {i+1}/{len(windows)} windows...")

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int32)
    types = np.array(types)

    if verbose:
        n_attack = np.sum(y == 1)
        n_normal = np.sum(y == 0)
        print(f"  Features extracted: {X.shape[0]} windows x {X.shape[1]} features")
        print(f"  Labels: {n_normal} normal, {n_attack} attack")

    return X, y, types


def normalize_features(X_train: np.ndarray,
                       X_val: np.ndarray = None,
                       X_test: np.ndarray = None
                       ) -> Tuple:
    """Normalize features to [0, 1] using train set statistics.

    Returns: (X_train_norm, X_val_norm, X_test_norm, min_vals, max_vals)
    """
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)

    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    X_train_norm = (X_train - min_vals) / range_vals

    result = [X_train_norm]
    if X_val is not None:
        X_val_norm = (X_val - min_vals) / range_vals
        X_val_norm = np.clip(X_val_norm, 0, 1)
        result.append(X_val_norm)
    if X_test is not None:
        X_test_norm = (X_test - min_vals) / range_vals
        # Don't clip test set to allow out-of-range detection
        result.append(X_test_norm)

    result.extend([min_vals, max_vals])
    return tuple(result)


if __name__ == "__main__":
    n_ids = FE_CONFIG["monitored_can_ids"]
    n_features = 13 * n_ids + 2
    print(f"Feature extraction configuration:")
    print(f"  Monitored CAN IDs: {n_ids}")
    print(f"  Features per window: {n_features}")
    print(f"  Window duration: {FE_CONFIG['window_duration_ms']} ms")
    print(f"  Window overlap: {FE_CONFIG['window_overlap_ratio'] * 100}%")
    print(f"  Monitored IDs: {[hex(x) for x in MONITORED_IDS]}")
