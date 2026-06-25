#!/usr/bin/env python3
"""
Hybrid multi-layer IDS rule layers for CAN bus anomaly detection.

Two lightweight, deterministic rule layers complement the learned
(autoencoder / OC-SVM) layer:

  1. Heartbeat / liveness rule -> targets A3 (BMS node disappearance).
     Learns each CAN ID's expected inter-arrival period from normal
     training traffic. At inference, flags a window if a monitored node
     has been absent for longer than k x expected_period. Deterministic
     detection latency ~= k x period. Memory: a few floats per ID
     (last-seen timestamp + expected period).

  2. Frequency-counter rule -> targets A6 (DoS flooding).
     Sliding-window message-rate counters (global bus load + per-ID).
     Learns the normal per-window frame-count distribution from training
     and flags a window whose rate exceeds mu + n*sigma (or, for CAN IDs
     never seen in training such as the 0x000 flood ID, more than a small
     fixed count). Detection latency = one window length. Memory: a few
     counters.

The learned layer (AE-INT8 / OC-SVM) catches stealthy in-range
manipulation (A1, A2, A5) that the rules cannot see.  Layers are fused
by OR: any layer flagging a window => anomaly.

Paper: Lightweight Autoencoder-Based Anomaly Detection for CAN Bus
       in Competition Motorcycles Deployed on ARM Cortex-M7
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

from feature_extraction import create_sliding_windows, FE_CONFIG, MONITORED_IDS

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

# Non-monitored marker ID used by the A3 injector to label suppression
# windows. It carries no payload and must be ignored by the rule layers.
A3_MARKER_ID = 0xFFF

# Default rule hyper-parameters (also surfaced in config / summary)
DEFAULT_K = 3            # heartbeat: flag if absent > k * expected_period
DEFAULT_N_SIGMA = 5      # frequency: flag if count > mu + n*sigma
UNKNOWN_ID_MIN_COUNT = 3  # flag a never-before-seen ID seen >= this in a window
RESET_GAP_US = 1_000_000  # >1 s total bus silence => session boundary, reset


def build_test_windows(frames: List) -> List[Tuple]:
    """Re-derive the exact sliding windows used by feature extraction.

    Returns a list of (window_frames, label, attack_type) tuples in the
    same order produced by ``extract_dataset_features`` so that rule
    predictions align index-for-index with X_test / y_test / types_test.
    """
    window_us = FE_CONFIG["window_duration_ms"] * 1000
    overlap = FE_CONFIG["window_overlap_ratio"]
    return create_sliding_windows(frames, window_us, overlap)


def learn_rule_params(train_frames: List,
                      monitored_ids: List[int] = None) -> Dict:
    """Learn rule-layer parameters from normal training traffic only.

    - expected inter-arrival period per CAN ID (median IAT, session gaps
      excluded)
    - per-window global frame-count distribution (mean, std)
    - per-window per-ID frame-count distribution (mean, std)
    - the set of CAN IDs observed during training (known IDs)
    """
    if monitored_ids is None:
        monitored_ids = MONITORED_IDS

    # --- expected period per CAN ID (heartbeat) ---
    ts_by_id = defaultdict(list)
    for f in train_frames:
        ts_by_id[f.can_id].append(f.timestamp_us)

    periods_us = {}
    for cid, ts in ts_by_id.items():
        arr = np.sort(np.asarray(ts, dtype=np.float64))
        if len(arr) >= 2:
            iats = np.diff(arr)
            iats = iats[iats < RESET_GAP_US]  # drop session-gap outliers
            if len(iats) > 0:
                periods_us[cid] = float(np.median(iats))

    # --- per-window frame-count statistics (frequency) ---
    windows = build_test_windows(train_frames)
    global_counts = []
    per_id_counts = defaultdict(list)
    for wf, _lbl, _atk in windows:
        global_counts.append(len(wf))
        c = defaultdict(int)
        for f in wf:
            c[f.can_id] += 1
        for cid in monitored_ids:
            per_id_counts[cid].append(c.get(cid, 0))

    gc = np.asarray(global_counts, dtype=np.float64)
    params = {
        "periods_us": periods_us,
        "global_count_mean": float(np.mean(gc)) if len(gc) else 0.0,
        "global_count_std": float(np.std(gc)) if len(gc) else 0.0,
        "global_count_max": float(np.max(gc)) if len(gc) else 0.0,
        "per_id_count_mean": {int(cid): float(np.mean(v))
                              for cid, v in per_id_counts.items()},
        "per_id_count_std": {int(cid): float(np.std(v))
                             for cid, v in per_id_counts.items()},
        "known_ids": sorted(int(c) for c in ts_by_id.keys()),
        "monitored_ids": [int(c) for c in monitored_ids],
        "n_train_windows": int(len(windows)),
    }
    return params


def heartbeat_predict(windows: List[Tuple], params: Dict,
                      k: float = DEFAULT_K,
                      liveness_ids: List[int] = None) -> np.ndarray:
    """Per-window liveness predictions (1 = a monitored node is silent).

    Stateful pass over windows in temporal order. For each window we use
    the most-recent frame timestamp as 'now'; a monitored node is flagged
    if now - last_seen[node] > k * expected_period[node].  A total bus
    silence > RESET_GAP_US (session boundary) re-initialises the tracker.
    """
    periods = params["periods_us"]
    if liveness_ids is None:
        liveness_ids = [cid for cid in params["monitored_ids"] if cid in periods]
    liveness_ids = [cid for cid in liveness_ids if cid in periods]

    preds = np.zeros(len(windows), dtype=int)
    last_seen = {}
    prev_now = None

    for i, (wf, _lbl, _atk) in enumerate(windows):
        if not wf:
            continue
        now = max(f.timestamp_us for f in wf)

        # Session-boundary reset: a long total silence is not a node
        # disappearance, it is a power cycle / new recording.
        if prev_now is not None and (now - prev_now) > RESET_GAP_US:
            last_seen = {}

        # Update last-seen for monitored IDs present in this window.
        for f in wf:
            cid = f.can_id
            if cid in liveness_ids:
                t = f.timestamp_us
                if cid not in last_seen or t > last_seen[cid]:
                    last_seen[cid] = t

        # Liveness check.
        flag = False
        for cid in liveness_ids:
            if cid in last_seen:
                if (now - last_seen[cid]) > k * periods[cid]:
                    flag = True
                    break
        preds[i] = 1 if flag else 0
        prev_now = now

    return preds


def freq_predict(windows: List[Tuple], params: Dict,
                 n_sigma: float = DEFAULT_N_SIGMA,
                 unknown_min: int = UNKNOWN_ID_MIN_COUNT) -> np.ndarray:
    """Per-window message-rate predictions (1 = abnormal bus load).

    Fires if the global per-window frame count exceeds mu + n*sigma, OR
    any monitored ID exceeds its own mu_id + n*sigma_id (+1 floor), OR a
    CAN ID never seen during training (e.g. the 0x000 DoS flood ID) is
    seen at least ``unknown_min`` times in the window.  The A3 marker ID
    is ignored.
    """
    g_thresh = params["global_count_mean"] + n_sigma * max(params["global_count_std"], 1e-9)
    per_mean = params["per_id_count_mean"]
    per_std = params["per_id_count_std"]
    known = set(params["known_ids"])

    preds = np.zeros(len(windows), dtype=int)
    for i, (wf, _lbl, _atk) in enumerate(windows):
        if not wf:
            continue
        total = len(wf)
        flag = total > g_thresh
        if not flag:
            c = defaultdict(int)
            for f in wf:
                c[f.can_id] += 1
            for cid, cnt in c.items():
                if cid == A3_MARKER_ID:
                    continue
                if cid in per_mean:
                    thr = per_mean[cid] + n_sigma * max(per_std[cid], 1e-9) + 1.0
                    if cnt > thr:
                        flag = True
                        break
                elif cid not in known:
                    if cnt >= unknown_min:
                        flag = True
                        break
        preds[i] = 1 if flag else 0
    return preds


def rule_layer_predict(windows: List[Tuple], params: Dict,
                       k: float = DEFAULT_K,
                       n_sigma: float = DEFAULT_N_SIGMA
                       ) -> Dict[str, np.ndarray]:
    """Run both rule layers and their OR fusion.

    Returns dict with 'heartbeat', 'freq', and 'rules' (heartbeat OR freq)
    per-window 0/1 arrays.
    """
    hb = heartbeat_predict(windows, params, k=k)
    fc = freq_predict(windows, params, n_sigma=n_sigma)
    rules = ((hb == 1) | (fc == 1)).astype(int)
    return {"heartbeat": hb, "freq": fc, "rules": rules}


def rule_memory_cost(params: Dict) -> Dict:
    """Quantify the embedded memory / latency cost of the rule layers.

    Heartbeat: per monitored ID store a 32-bit last-seen timestamp and a
    32-bit expected-period constant.  Frequency: per monitored ID a 16-bit
    sliding counter plus a 16-bit precomputed threshold, plus a global
    counter + threshold.  All updates are O(1) integer ops per frame.
    """
    n_ids = len(params["monitored_ids"])
    heartbeat_bytes = n_ids * (4 + 4)          # last_seen u32 + period u32
    freq_bytes = n_ids * (2 + 2) + (2 + 2)     # per-ID cnt+thr + global cnt+thr
    total_bytes = heartbeat_bytes + freq_bytes

    clock_hz = CONFIG["embedded"]["clock_mhz"] * 1e6
    # Per-frame update: compare-and-store last_seen + increment a counter.
    # ~6 Cortex-M7 instructions per frame (load/cmp/store/add). Per-window
    # liveness scan: ~3 ops per monitored ID.
    cycles_per_frame_update = 6
    cycles_per_window_check = 3 * n_ids
    update_latency_us = (cycles_per_frame_update / clock_hz) * 1e6
    window_check_latency_us = (cycles_per_window_check / clock_hz) * 1e6

    return {
        "heartbeat_state_bytes": heartbeat_bytes,
        "freq_counter_state_bytes": freq_bytes,
        "total_rule_state_bytes": total_bytes,
        "total_rule_state_kb": round(total_bytes / 1024, 4),
        "per_frame_update_cycles": cycles_per_frame_update,
        "per_frame_update_latency_us": round(update_latency_us, 4),
        "per_window_check_cycles": cycles_per_window_check,
        "per_window_check_latency_us": round(window_check_latency_us, 4),
        "n_monitored_ids": n_ids,
    }


if __name__ == "__main__":
    print("[OK] Hybrid rule layers ready")
    print(f"  Defaults: k={DEFAULT_K}, n_sigma={DEFAULT_N_SIGMA}, "
          f"unknown_min={UNKNOWN_ID_MIN_COUNT}, reset_gap_us={RESET_GAP_US}")
