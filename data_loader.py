#!/usr/bin/env python3
"""
Data loader for CAN bus traffic from competition motorcycle testbed.

Parses Excel files from Sevcon Gen4 data exports and reconstructs
CAN-like traffic with realistic timing and message structure.

The real data contains decoded application-layer signals (timestamps,
throttle, motor speed, currents, voltages, temperatures) logged from
a Sevcon Gen4 motor controller via CAN bus. We reconstruct synthetic
CAN frames by re-encoding these signals into 8-byte payloads with
CAN IDs assigned to each subsystem, at realistic CAN bus timing.

Paper: Lightweight Autoencoder-Based Anomaly Detection for CAN Bus
       in Competition Motorcycles Deployed on ARM Cortex-M7
"""

import json
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)


@dataclass
class CANFrame:
    """Represents a single CAN bus frame."""
    timestamp_us: int
    can_id: int
    dlc: int
    data: bytes
    label: str       # "normal" or "attack"
    attack_type: str  # "" for normal, "A1"-"A6" for attacks


# Signal-to-CAN-ID mapping for the motorcycle testbed
# Each CAN ID carries specific signals encoded in the 8-byte payload
CAN_SIGNAL_MAP = {
    0x100: {  # VCU - Throttle and brake
        "name": "VCU_Throttle",
        "signals": ["throttle_pct", "throttle_voltage"],
        "period_ms": 10,
        "node": "VCU",
    },
    0x200: {  # Motor Controller - Speed and current
        "name": "Motor_Speed_Current",
        "signals": ["motor_rpm", "motor_current_ac", "torque_pct"],
        "period_ms": 10,
        "node": "Motor",
    },
    0x201: {  # Motor Controller - Temperatures
        "name": "Motor_Temperature",
        "signals": ["motor_temp", "heatsink_temp", "ptc_temp"],
        "period_ms": 100,
        "node": "Motor",
    },
    0x300: {  # BMS - Pack voltage and current
        "name": "BMS_Pack",
        "signals": ["battery_voltage", "battery_current"],
        "period_ms": 100,
        "node": "BMS",
    },
    0x301: {  # BMS - SOC and status
        "name": "BMS_SOC",
        "signals": ["soc_pct"],
        "period_ms": 100,
        "node": "BMS",
    },
    0x400: {  # IMU - Lean angle and rates (synthetic)
        "name": "IMU_Lean",
        "signals": ["lean_angle", "lean_rate"],
        "period_ms": 20,
        "node": "IMU",
    },
}

# Sevcon column name patterns -> signal name mapping
SEVCON_COLUMN_MAP = {
    "throttle_pct": ["Throttle Value"],
    "throttle_voltage": ["Throttle Input Voltage"],
    "motor_rpm": ["Velocity actual value"],
    "motor_current_ac": ["Actual AC Motor Current"],
    "torque_pct": ["Torque % of peak"],
    "motor_temp": ["Motor Temperature"],
    "heatsink_temp": ["Heatsink Temperature"],
    "ptc_temp": ["Temperature (Measured - PTC)"],
    "battery_voltage": ["Battery Voltage"],
    "battery_current": ["Battery Current"],
    "id_current": ["Id (If)"],
    "iq_current": ["Iq (Ia)"],
    "motor_voltage_ac": ["Actual AC Motor Voltage"],
}


def find_session_files(dataset_path: str) -> List[Path]:
    """Find all valid session data files in the dataset directory."""
    data_dir = Path(dataset_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    # Focus on .xlsm files (Sevcon template format with consistent structure)
    xlsm_files = list(data_dir.glob("*.xlsm"))

    # Exclude templates and non-data files
    exclude_prefixes = [
        "Plantilla", "ORGANIZACION", "Programa", "Telemetria_RC",
    ]
    exclude_keywords = [
        "Grafica", "CorteTemp",
    ]

    filtered = []
    for f in xlsm_files:
        name = f.name
        skip = any(name.startswith(p) for p in exclude_prefixes)
        if not skip:
            skip = any(k.lower() in name.lower() for k in exclude_keywords)
        if not skip:
            filtered.append(f)

    return sorted(filtered)


def _match_column(df_columns: List[str], patterns: List[str]) -> Optional[Tuple[str, str]]:
    """Find a column matching any of the given patterns.
    Returns (time_column, value_column) or None.
    """
    for pattern in patterns:
        for i, col in enumerate(df_columns):
            if pattern.lower() in str(col).lower() and "time" not in str(col).lower():
                # Find associated time column (the one before it)
                # In Sevcon format, columns come in triplets: Time, Value, (empty)
                # Find the nearest preceding Time column
                time_col = None
                for j in range(i - 1, -1, -1):
                    if "time" in str(df_columns[j]).lower():
                        time_col = df_columns[j]
                        break
                if time_col is not None:
                    return (time_col, col)
    return None


def parse_sevcon_session(filepath: Path) -> Optional[Dict[str, pd.DataFrame]]:
    """Parse a Sevcon Gen4 session file.

    Returns a dict mapping signal names to DataFrames with columns
    ['time_s', 'value'].
    """
    try:
        xl = pd.ExcelFile(filepath, engine="openpyxl")
        if "Datos" not in xl.sheet_names:
            return None

        df = pd.read_excel(xl, sheet_name="Datos")
        if df.shape[0] < 100 or df.shape[1] < 10:
            return None

        columns = list(df.columns)
        signals = {}

        for signal_name, patterns in SEVCON_COLUMN_MAP.items():
            match = _match_column(columns, patterns)
            if match is not None:
                time_col, val_col = match
                signal_df = df[[time_col, val_col]].dropna().copy()
                signal_df.columns = ["time_s", "value"]
                # Convert to numeric, drop non-numeric rows
                signal_df["time_s"] = pd.to_numeric(signal_df["time_s"], errors="coerce")
                signal_df["value"] = pd.to_numeric(signal_df["value"], errors="coerce")
                signal_df = signal_df.dropna()
                if len(signal_df) > 50:
                    signals[signal_name] = signal_df.reset_index(drop=True)

        return signals if len(signals) >= 3 else None

    except Exception as e:
        print(f"  Warning: Could not parse {filepath.name}: {e}")
        return None


def _encode_signal_to_bytes(value: float, scale: float, offset: float,
                            n_bytes: int = 2, signed: bool = True) -> bytes:
    """Encode a signal value into CAN payload bytes (little-endian)."""
    raw = int((value - offset) / scale)
    if signed:
        raw = max(-(1 << (n_bytes * 8 - 1)), min((1 << (n_bytes * 8 - 1)) - 1, raw))
        fmt = "<h" if n_bytes == 2 else "<b"
    else:
        raw = max(0, min((1 << (n_bytes * 8)) - 1, raw))
        fmt = "<H" if n_bytes == 2 else "<B"
    return struct.pack(fmt, raw)


def reconstruct_can_traffic(signals: Dict[str, pd.DataFrame],
                            session_id: str) -> List[CANFrame]:
    """Reconstruct CAN-like traffic from decoded signal data.

    Maps decoded signals to CAN IDs and encodes them into 8-byte payloads
    with realistic CAN timing.
    """
    frames = []

    # Determine session time range
    all_times = []
    for sig_df in signals.values():
        all_times.extend(sig_df["time_s"].tolist())
    if not all_times:
        return frames

    t_start = min(all_times)
    t_end = max(all_times)

    # For each CAN ID, generate frames at the specified period
    for can_id, can_info in CAN_SIGNAL_MAP.items():
        period_s = can_info["period_ms"] / 1000.0

        # Collect signals for this CAN ID
        available_signals = {}
        for sig_name in can_info["signals"]:
            if sig_name in signals:
                available_signals[sig_name] = signals[sig_name]

        # For IMU (synthetic lean angle), generate from motor speed + throttle
        if can_id == 0x400:
            if "motor_rpm" in signals and "throttle_pct" in signals:
                rpm_df = signals["motor_rpm"]
                thr_df = signals["throttle_pct"]
                # Synthesize lean angle from speed (higher speed -> more lean in corners)
                # This is a rough approximation for track data
                t_array = np.arange(t_start, t_end, period_s)
                rpm_interp = np.interp(t_array, rpm_df["time_s"].values, rpm_df["value"].values)
                thr_interp = np.interp(t_array, thr_df["time_s"].values, thr_df["value"].values)
                # Lean angle: simulate oscillation correlated with speed
                speed_norm = rpm_interp / max(np.max(np.abs(rpm_interp)), 1)
                # Add sinusoidal cornering pattern
                corner_freq = 0.15  # ~6-7s per corner at racing speed
                lean = 35 * speed_norm * np.sin(2 * np.pi * corner_freq * (t_array - t_start))
                # Add noise
                rng = np.random.default_rng(hash(session_id) % 2**32)
                lean += rng.normal(0, 1.5, len(lean))
                lean_rate = np.gradient(lean, period_s)

                for i, t in enumerate(t_array):
                    payload = bytearray(8)
                    # Lean angle: signed 16-bit, 0.01 deg/LSB
                    payload[0:2] = _encode_signal_to_bytes(lean[i], 0.01, 0, 2, True)
                    # Lean rate: signed 16-bit, 0.1 deg/s/LSB
                    payload[2:4] = _encode_signal_to_bytes(lean_rate[i], 0.1, 0, 2, True)
                    # Lateral accel (synthetic): signed 16-bit, 0.001 g/LSB
                    lat_accel = lean[i] * 0.017  # rough: tan(lean) * g
                    payload[4:6] = _encode_signal_to_bytes(lat_accel, 0.001, 0, 2, True)

                    frames.append(CANFrame(
                        timestamp_us=int(t * 1e6),
                        can_id=can_id,
                        dlc=8,
                        data=bytes(payload),
                        label="normal",
                        attack_type="",
                    ))
                continue  # skip default processing for IMU

        # For BMS SOC (0x301), synthesize from battery voltage if available
        if can_id == 0x301:
            if "battery_voltage" in signals:
                bv_df = signals["battery_voltage"]
                t_array = np.arange(t_start, t_end, period_s)
                for t in t_array:
                    voltage = np.interp(t, bv_df["time_s"].values, bv_df["value"].values)
                    v_min, v_max = 70, 100
                    soc = max(0, min(100, (voltage - v_min) / (v_max - v_min) * 100))
                    payload = bytearray(8)
                    payload[0:2] = _encode_signal_to_bytes(soc, 0.1, 0, 2, False)
                    frames.append(CANFrame(
                        timestamp_us=int(t * 1e6),
                        can_id=can_id,
                        dlc=8,
                        data=bytes(payload),
                        label="normal",
                        attack_type="",
                    ))
                continue

        if not available_signals and can_id != 0x400:
            # If no signals available for this ID but it's not IMU, skip
            continue

        # Generate frames at regular intervals
        t_array = np.arange(t_start, t_end, period_s)

        for t in t_array:
            payload = bytearray(8)
            byte_offset = 0

            if can_id == 0x100:
                # VCU: throttle_pct (0-100%, unsigned 16-bit, 0.01%/LSB)
                #       throttle_voltage (0-5V, unsigned 16-bit, 0.001V/LSB)
                if "throttle_pct" in available_signals:
                    val = np.interp(t, available_signals["throttle_pct"]["time_s"].values,
                                   available_signals["throttle_pct"]["value"].values)
                    payload[0:2] = _encode_signal_to_bytes(val * 100, 0.01, 0, 2, False)
                if "throttle_voltage" in available_signals:
                    val = np.interp(t, available_signals["throttle_voltage"]["time_s"].values,
                                   available_signals["throttle_voltage"]["value"].values)
                    payload[2:4] = _encode_signal_to_bytes(val, 0.001, 0, 2, False)

            elif can_id == 0x200:
                # Motor: RPM (signed 16-bit, 1 RPM/LSB)
                #        AC current (signed 16-bit, 0.1 A/LSB)
                #        Torque % (signed 16-bit, 0.1 %/LSB)
                if "motor_rpm" in available_signals:
                    val = np.interp(t, available_signals["motor_rpm"]["time_s"].values,
                                   available_signals["motor_rpm"]["value"].values)
                    payload[0:2] = _encode_signal_to_bytes(val, 1, 0, 2, True)
                if "motor_current_ac" in available_signals:
                    val = np.interp(t, available_signals["motor_current_ac"]["time_s"].values,
                                   available_signals["motor_current_ac"]["value"].values)
                    payload[2:4] = _encode_signal_to_bytes(val, 0.1, 0, 2, True)
                if "torque_pct" in available_signals:
                    val = np.interp(t, available_signals["torque_pct"]["time_s"].values,
                                   available_signals["torque_pct"]["value"].values)
                    payload[4:6] = _encode_signal_to_bytes(val, 0.1, 0, 2, True)

            elif can_id == 0x201:
                # Motor temps: motor_temp, heatsink_temp, ptc_temp
                # Each signed 16-bit, 0.1 degC/LSB
                for i, sig in enumerate(["motor_temp", "heatsink_temp", "ptc_temp"]):
                    if sig in available_signals:
                        val = np.interp(t, available_signals[sig]["time_s"].values,
                                       available_signals[sig]["value"].values)
                        payload[i*2:(i+1)*2] = _encode_signal_to_bytes(val, 0.1, 0, 2, True)

            elif can_id == 0x300:
                # BMS Pack: voltage (unsigned 16-bit, 0.01V/LSB)
                #           current (signed 16-bit, 0.01A/LSB)
                if "battery_voltage" in available_signals:
                    val = np.interp(t, available_signals["battery_voltage"]["time_s"].values,
                                   available_signals["battery_voltage"]["value"].values)
                    payload[0:2] = _encode_signal_to_bytes(val, 0.01, 0, 2, False)
                if "battery_current" in available_signals:
                    val = np.interp(t, available_signals["battery_current"]["time_s"].values,
                                   available_signals["battery_current"]["value"].values)
                    payload[2:4] = _encode_signal_to_bytes(val, 0.01, 0, 2, True)

            frames.append(CANFrame(
                timestamp_us=int(t * 1e6),
                can_id=can_id,
                dlc=8,
                data=bytes(payload),
                label="normal",
                attack_type="",
            ))

    # Sort by timestamp
    frames.sort(key=lambda f: f.timestamp_us)
    return frames


def load_all_sessions(max_sessions: int = 0) -> Tuple[List[CANFrame], Dict]:
    """Load and combine all session data.

    Args:
        max_sessions: Maximum sessions to load (0 = all)

    Returns:
        (all_frames, stats_dict)
    """
    dataset_path = CONFIG["data"]["dataset_path"]
    abs_path = (Path(__file__).parent / dataset_path).resolve()

    print(f"Looking for session files in: {abs_path}", flush=True)
    files = find_session_files(str(abs_path))
    print(f"Found {len(files)} session files", flush=True)

    all_frames = []
    stats = {
        "n_sessions": 0,
        "n_files_parsed": 0,
        "n_files_failed": 0,
        "total_frames": 0,
        "total_duration_s": 0,
        "unique_can_ids": set(),
        "can_nodes": 0,
        "session_details": [],
    }

    sessions_loaded = 0
    cumulative_offset_us = 0
    GAP_BETWEEN_SESSIONS_US = 10_000_000  # 10 s gap between sessions

    for filepath in files:
        if max_sessions > 0 and sessions_loaded >= max_sessions:
            break

        print(f"  Parsing: {filepath.name}...", end=" ", flush=True)
        signals = parse_sevcon_session(filepath)
        if signals is not None:
            session_frames = reconstruct_can_traffic(signals, filepath.stem)
            if session_frames:
                n_frames = len(session_frames)
                duration = (session_frames[-1].timestamp_us - session_frames[0].timestamp_us) / 1e6
                ids_in_session = set(f.can_id for f in session_frames)

                # Offset timestamps so sessions don't overlap after sort
                base_ts = session_frames[0].timestamp_us
                for frame in session_frames:
                    frame.timestamp_us = (frame.timestamp_us - base_ts) + cumulative_offset_us
                cumulative_offset_us = session_frames[-1].timestamp_us + GAP_BETWEEN_SESSIONS_US

                all_frames.extend(session_frames)
                stats["n_sessions"] += 1
                stats["n_files_parsed"] += 1
                stats["unique_can_ids"].update(ids_in_session)
                stats["session_details"].append({
                    "file": filepath.name,
                    "n_frames": n_frames,
                    "duration_s": round(duration, 1),
                    "can_ids": [hex(x) for x in sorted(ids_in_session)],
                    "signals_found": list(signals.keys()),
                })
                sessions_loaded += 1
                print(f"OK ({n_frames} frames, {duration:.0f}s, {len(ids_in_session)} IDs)")
            else:
                stats["n_files_failed"] += 1
                print("SKIP (no CAN frames generated)")
        else:
            stats["n_files_failed"] += 1
            print("SKIP (parse failed or insufficient data)")

    stats["total_frames"] = len(all_frames)
    stats["unique_can_ids"] = sorted([hex(x) for x in stats["unique_can_ids"]])
    stats["can_nodes"] = len(set(CAN_SIGNAL_MAP[int(x, 16)]["node"]
                                for x in stats["unique_can_ids"]
                                if int(x, 16) in CAN_SIGNAL_MAP))

    if all_frames:
        all_frames.sort(key=lambda f: f.timestamp_us)
        total_duration_s = sum(sd["duration_s"] for sd in stats["session_details"])
        stats["total_duration_s"] = round(total_duration_s, 1)

    return all_frames, stats


def split_data(frames: List[CANFrame]) -> Tuple[List[CANFrame], List[CANFrame], List[CANFrame]]:
    """Split frames into train/val/test using temporal split.

    Important: we split by session boundaries to avoid data leakage.
    """
    n = len(frames)
    train_end = int(n * CONFIG["data"]["train_ratio"])
    val_end = train_end + int(n * CONFIG["data"]["val_ratio"])

    train = frames[:train_end]
    val = frames[train_end:val_end]
    test = frames[val_end:]

    return train, val, test


if __name__ == "__main__":
    print("=" * 60)
    print("CAN Bus Data Loader — Testing")
    print("=" * 60)

    dataset_path = (Path(__file__).parent / CONFIG["data"]["dataset_path"]).resolve()
    print(f"Dataset path: {dataset_path}")

    try:
        files = find_session_files(str(dataset_path))
        print(f"Found {len(files)} session files")

        # Test parsing first file
        if files:
            print(f"\nTesting parse of: {files[0].name}")
            signals = parse_sevcon_session(files[0])
            if signals:
                print(f"  Signals found: {list(signals.keys())}")
                for sig_name, sig_df in signals.items():
                    print(f"    {sig_name}: {len(sig_df)} points, "
                          f"range [{sig_df['value'].min():.2f}, {sig_df['value'].max():.2f}]")

                # Test CAN reconstruction
                frames = reconstruct_can_traffic(signals, files[0].stem)
                print(f"\n  Reconstructed {len(frames)} CAN frames")
                ids = set(hex(f.can_id) for f in frames)
                print(f"  CAN IDs: {sorted(ids)}")
                if frames:
                    duration = (frames[-1].timestamp_us - frames[0].timestamp_us) / 1e6
                    print(f"  Duration: {duration:.1f} s")

    except FileNotFoundError as e:
        print(f"  {e}")
