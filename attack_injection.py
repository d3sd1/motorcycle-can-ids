#!/usr/bin/env python3
"""
Attack injection framework for CAN bus anomaly detection experiments.

Implements six attack types specific to competition motorcycle CAN networks:
A1: Throttle Position Spoofing
A2: Lean Angle Sensor Injection
A3: BMS Node Disappearance
A4: Replay Attack
A5: Fuzzing Attack
A6: DoS Flooding Attack

Paper: Lightweight Autoencoder-Based Anomaly Detection for CAN Bus
       in Competition Motorcycles Deployed on ARM Cortex-M7
"""

import json
import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import copy
import sys

# Import CANFrame from data_loader to avoid circular imports
# We define CANFrame here and data_loader imports from here
from dataclasses import dataclass


@dataclass
class CANFrame:
    """Represents a single CAN bus frame."""
    timestamp_us: int
    can_id: int
    dlc: int
    data: bytes
    label: str       # "normal" or "attack"
    attack_type: str  # "" for normal, "A1"-"A6" for attacks


CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)


def inject_tps_spoofing(frames: List[CANFrame], rng: np.random.Generator,
                        config: dict) -> Tuple[List[CANFrame], int, int]:
    """A1: Inject throttle position spoofing attack.

    Returns (injected_frames, attack_start_us, attack_end_us)
    """
    attack_config = config["attack_injection"]["attacks"]["A1_tps_spoofing"]
    target_id = int(attack_config["target_can_id"], 16)

    duration_s = rng.uniform(*config["attack_injection"]["attack_duration_range_s"])
    # Pick random start point in the first 80% of frames
    max_start = int(len(frames) * 0.8)
    start_idx = rng.integers(0, max(1, max_start))
    start_time = frames[start_idx].timestamp_us
    end_time = start_time + int(duration_s * 1e6)

    # Generate spoofed TPS value
    spoofed_tps = rng.uniform(*attack_config["spoofed_value_range"])
    rate_mult = rng.uniform(*attack_config["injection_rate_multiplier"])

    normal_period = config["can_reconstruction"]["message_periods_ms"]["VCU"] * 1000
    inject_period = int(normal_period / rate_mult)

    injected = []
    t = start_time
    while t < end_time:
        payload = bytearray(8)
        # TPS: unsigned 16-bit, 0.01%/LSB (same encoding as data_loader)
        tps_raw = int(spoofed_tps * 100 / 0.01)
        tps_raw = max(0, min(65535, tps_raw))
        struct.pack_into("<H", payload, 0, tps_raw)
        # Add some jitter to voltage channel too
        volt_raw = int(spoofed_tps / 100 * 5 / 0.001)
        volt_raw = max(0, min(65535, volt_raw))
        struct.pack_into("<H", payload, 2, volt_raw)

        frame = CANFrame(
            timestamp_us=t, can_id=target_id, dlc=8,
            data=bytes(payload), label="attack",
            attack_type="A1_tps_spoofing",
        )
        injected.append(frame)
        t += inject_period

    return injected, start_time, end_time


def inject_lean_angle(frames: List[CANFrame], rng: np.random.Generator,
                      config: dict) -> Tuple[List[CANFrame], int, int]:
    """A2: Inject false lean angle readings."""
    attack_config = config["attack_injection"]["attacks"]["A2_lean_injection"]
    target_id = int(attack_config["target_can_id"], 16)

    duration_s = rng.uniform(*config["attack_injection"]["attack_duration_range_s"])
    max_start = int(len(frames) * 0.8)
    start_idx = rng.integers(0, max(1, max_start))
    start_time = frames[start_idx].timestamp_us
    end_time = start_time + int(duration_s * 1e6)

    spoofed_lean = rng.uniform(*attack_config["spoofed_lean_range_deg"])
    angular_offset = rng.uniform(*attack_config["angular_rate_offset_range"])

    period = config["can_reconstruction"]["message_periods_ms"]["IMU"] * 1000
    injected = []
    t = start_time
    while t < end_time:
        payload = bytearray(8)
        # Lean angle: signed 16-bit, 0.01 deg/LSB
        lean_raw = int(spoofed_lean / 0.01)
        lean_raw = max(-32768, min(32767, lean_raw))
        struct.pack_into("<h", payload, 0, lean_raw)
        # Lean rate: signed 16-bit, 0.1 deg/s/LSB
        rate_raw = int(angular_offset / 0.1)
        rate_raw = max(-32768, min(32767, rate_raw))
        struct.pack_into("<h", payload, 2, rate_raw)

        frame = CANFrame(
            timestamp_us=t, can_id=target_id, dlc=8,
            data=bytes(payload), label="attack",
            attack_type="A2_lean_injection",
        )
        injected.append(frame)
        t += period

    return injected, start_time, end_time


def inject_bms_disappearance(frames: List[CANFrame], rng: np.random.Generator,
                             config: dict) -> Tuple[List[int], int, int]:
    """A3: Suppress BMS messages by removing them from the bus.

    Simulates a BMS node going offline: its CAN frames stop appearing
    entirely during the attack window. Returns indices to remove plus
    the attack time interval for labeling affected windows.

    Returns (indices_to_remove, start_time_us, end_time_us)
    """
    attack_config = config["attack_injection"]["attacks"]["A3_bms_disappearance"]
    target_ids = [int(x, 16) for x in attack_config["target_can_ids"]]

    duration_s = rng.uniform(*attack_config["suppression_duration_s"])
    max_start = int(len(frames) * 0.8)
    start_idx = rng.integers(0, max(1, max_start))
    start_time = frames[start_idx].timestamp_us
    end_time = start_time + int(duration_s * 1e6)

    indices_to_remove = []
    for i, frame in enumerate(frames):
        if (frame.can_id in target_ids and
                start_time <= frame.timestamp_us <= end_time and
                frame.label == "normal"):
            indices_to_remove.append(i)

    return indices_to_remove, start_time, end_time


def inject_replay(frames: List[CANFrame], rng: np.random.Generator,
                  config: dict) -> Tuple[List[CANFrame], int, int]:
    """A4: Replay previously captured traffic."""
    attack_config = config["attack_injection"]["attacks"]["A4_replay"]

    replay_offset_s = rng.uniform(*attack_config["replay_offset_s"])
    replay_duration_s = rng.uniform(*attack_config["replay_duration_s"])

    # Select insertion point in the second half of the recording
    total_time = frames[-1].timestamp_us - frames[0].timestamp_us
    insert_time = frames[0].timestamp_us + int(total_time * rng.uniform(0.5, 0.85))
    end_time = insert_time + int(replay_duration_s * 1e6)

    # Source segment (from earlier in the recording)
    source_start = insert_time - int(replay_offset_s * 1e6)
    source_end = source_start + int(replay_duration_s * 1e6)

    # Ensure source is within bounds
    source_start = max(frames[0].timestamp_us, source_start)
    source_end = min(frames[-1].timestamp_us, source_end)

    replayed = []
    for frame in frames:
        if source_start <= frame.timestamp_us <= source_end and frame.label == "normal":
            time_offset = frame.timestamp_us - source_start
            new_frame = CANFrame(
                timestamp_us=insert_time + time_offset,
                can_id=frame.can_id,
                dlc=frame.dlc,
                data=frame.data,  # exact same payload (replay)
                label="attack",
                attack_type="A4_replay",
            )
            replayed.append(new_frame)

    return replayed, insert_time, end_time


def inject_fuzzing(frames: List[CANFrame], rng: np.random.Generator,
                   config: dict) -> Tuple[List[CANFrame], int, int]:
    """A5: Inject frames with random payloads."""
    attack_config = config["attack_injection"]["attacks"]["A5_fuzzing"]

    duration_s = rng.uniform(*config["attack_injection"]["attack_duration_range_s"])
    injection_rate = rng.uniform(*attack_config["injection_rate_fps"])

    max_start = int(len(frames) * 0.8)
    start_idx = rng.integers(0, max(1, max_start))
    start_time = frames[start_idx].timestamp_us
    end_time = start_time + int(duration_s * 1e6)

    valid_ids = [int(v, 16) for v in config["can_reconstruction"]["can_ids"].values()]
    period = int(1e6 / max(injection_rate, 1))

    injected = []
    t = start_time
    while t < end_time:
        random_id = rng.choice(valid_ids)
        random_data = bytes(rng.integers(0, 256, size=8, dtype=np.uint8))

        frame = CANFrame(
            timestamp_us=t, can_id=random_id, dlc=8,
            data=random_data, label="attack",
            attack_type="A5_fuzzing",
        )
        injected.append(frame)
        t += period

    return injected, start_time, end_time


def inject_dos_flooding(frames: List[CANFrame], rng: np.random.Generator,
                        config: dict) -> Tuple[List[CANFrame], int, int]:
    """A6: Flood the bus with highest-priority frames."""
    attack_config = config["attack_injection"]["attacks"]["A6_dos_flooding"]
    flood_id = int(attack_config["frame_id"], 16)

    duration_s = rng.uniform(*config["attack_injection"]["attack_duration_range_s"])
    flooding_rate = rng.uniform(*attack_config["flooding_rate_fps"])

    max_start = int(len(frames) * 0.8)
    start_idx = rng.integers(0, max(1, max_start))
    start_time = frames[start_idx].timestamp_us
    end_time = start_time + int(duration_s * 1e6)

    period = int(1e6 / max(flooding_rate, 1))
    injected = []
    t = start_time
    while t < end_time:
        frame = CANFrame(
            timestamp_us=t, can_id=flood_id, dlc=8,
            data=b'\x00' * 8, label="attack",
            attack_type="A6_dos_flooding",
        )
        injected.append(frame)
        t += period

    return injected, start_time, end_time


ATTACK_INJECTORS = {
    "A1_tps_spoofing": inject_tps_spoofing,
    "A2_lean_injection": inject_lean_angle,
    "A4_replay": inject_replay,
    "A5_fuzzing": inject_fuzzing,
    "A6_dos_flooding": inject_dos_flooding,
}


def inject_all_attacks(frames: List[CANFrame], seed: int,
                       n_instances: int = 50) -> Tuple[List[CANFrame], Dict]:
    """Inject all attack types into a copy of the frame list.

    Returns (frames_with_attacks, injection_log)
    """
    rng = np.random.default_rng(seed)
    result = [CANFrame(f.timestamp_us, f.can_id, f.dlc, f.data, f.label, f.attack_type)
              for f in frames]

    injection_log = {}

    # Inject A1, A2, A4, A5, A6 (these add new frames)
    for attack_type, injector in ATTACK_INJECTORS.items():
        total_injected = 0
        for _ in range(n_instances):
            new_frames, _, _ = injector(result, rng, CONFIG)
            result.extend(new_frames)
            total_injected += len(new_frames)
        injection_log[attack_type] = {
            "instances": n_instances,
            "total_frames_injected": total_injected,
        }

    # Handle A3 (BMS disappearance) separately — suppress frames entirely
    # and insert invisible marker frames for window labeling
    a3_total_suppressed = 0
    a3_intervals = []
    indices_to_remove = set()
    for _ in range(n_instances):
        indices, start, end = inject_bms_disappearance(result, rng, CONFIG)
        indices_to_remove.update(indices)
        a3_total_suppressed += len(indices)
        a3_intervals.append((start, end))

    # Remove suppressed BMS frames (reverse order to preserve indices)
    for idx in sorted(indices_to_remove, reverse=True):
        if idx < len(result):
            del result[idx]

    # Insert marker frames with non-monitored CAN ID for window labeling.
    # Markers are placed across the WHOLE suppression interval (one every
    # ~stride) so that every sliding window overlapping the interval is
    # labelled A3.  Physically, the BMS node is absent for the entire
    # interval, so the entire interval is anomalous ground truth -- the
    # previous single mid-interval marker under-labelled A3.  The marker
    # ID (0xFFF, dlc=0) is non-monitored: it does not affect any extracted
    # feature and is ignored by the rule layers.
    fe = CONFIG["feature_extraction"]
    marker_step_us = max(1, int(fe["window_duration_ms"] * (1 - fe["window_overlap_ratio"]) * 1000))
    for start, end in a3_intervals:
        t = start
        while t <= end:
            result.append(CANFrame(
                timestamp_us=int(t),
                can_id=0xFFF,  # non-monitored ID, won't affect features
                dlc=0,
                data=b'\x00' * 8,
                label="attack",
                attack_type="A3_bms_disappearance",
            ))
            t += marker_step_us

    injection_log["A3_bms_disappearance"] = {
        "instances": n_instances,
        "total_frames_suppressed": a3_total_suppressed,
    }

    # Sort by timestamp
    result.sort(key=lambda f: f.timestamp_us)

    return result, injection_log


if __name__ == "__main__":
    print("[OK] Attack injection framework ready")
    print(f"  Attack types: {list(ATTACK_INJECTORS.keys()) + ['A3_bms_disappearance']}")
    print(f"  Instances per type: {CONFIG['attack_injection']['instances_per_attack_type']}")
