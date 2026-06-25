#!/usr/bin/env python3
"""
Generate results/HYBRID_SUMMARY.md from aggregated_results.json.

Produces the before/after tables (single AE-INT8 vs HYBRID) for per-attack
F1 and detection latency, the exact rule thresholds used, and the embedded
memory cost of the rule layers.  Numbers are read verbatim from the
aggregated results -- nothing is recomputed or rounded by hand.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
AGG = RESULTS_DIR / "aggregated_results.json"

ATTACKS = [
    ("A1_tps_spoofing", "A1 TPS spoofing"),
    ("A2_lean_injection", "A2 Lean injection"),
    ("A3_bms_disappearance", "A3 BMS disappearance"),
    ("A4_replay", "A4 Replay"),
    ("A5_fuzzing", "A5 Fuzzing"),
    ("A6_dos_flooding", "A6 DoS flooding"),
]


def f1(d, method, atk):
    pa = d.get("per_attack", {}).get(method, {})
    if atk in pa:
        return pa[atk].get("f1_mean", 0.0), pa[atk].get("f1_std", 0.0)
    return None, None


def lat(d, key, atk):
    dl = d.get(key, {})
    if atk in dl:
        return dl[atk].get("mean_ms", 0.0), dl[atk].get("std_ms", 0.0)
    return None, None


def fmt(v, s):
    if v is None:
        return "--"
    return f"{v:.3f} ± {s:.3f}"


def fmt_ms(v, s):
    if v is None:
        return "--"
    return f"{v:.1f} ± {s:.1f}"


def main():
    with open(AGG) as fh:
        d = json.load(fh)

    overall = d["overall"]
    hc = d.get("hybrid_config", {})
    rb = d.get("rule_breakdown", {})
    res = d.get("embedded_resources", {})
    rule_res = res.get("rule_layers", {})
    hyb_res = res.get("hybrid", {})

    lines = []
    lines.append("# Hybrid Multi-Layer IDS -- Results Summary\n")
    lines.append(f"_Seeds_: {d.get('seeds')} (mean ± std over {d.get('n_seeds')} runs). "
                 f"All numbers are produced verbatim by `run_all.py`.\n")
    lines.append(f"_Timestamp_: {d.get('timestamp')}\n")

    # ---- Overall F1 leaderboard ----
    lines.append("\n## Overall detection performance\n")
    lines.append("| Method | Precision | Recall | F1 | FPR |")
    lines.append("|---|---|---|---|---|")
    order = ["THRESH", "OC-SVM", "IF", "LSTM-AE", "AE-FP32", "AE-INT8",
             "HYBRID-OCSVM", "HYBRID"]
    for m in order:
        if m not in overall:
            continue
        o = overall[m]
        def g(k):
            return o.get(k, {}).get("mean", 0.0), o.get(k, {}).get("std", 0.0)
        p, ps = g("precision"); r, rs = g("recall"); f, fs = g("f1_score"); fp, fps = g("fpr")
        star = " **" if m in ("HYBRID", "HYBRID-OCSVM") else " "
        name = f"**{m}**" if m in ("HYBRID", "HYBRID-OCSVM") else m
        lines.append(f"| {name} | {p:.3f}±{ps:.3f} | {r:.3f}±{rs:.3f} | "
                     f"{f:.3f}±{fs:.3f} | {fp:.3f}±{fps:.3f} |")

    thr_f1 = overall.get("THRESH", {}).get("f1_score", {}).get("mean", 0.0)
    oc_f1 = overall.get("OC-SVM", {}).get("f1_score", {}).get("mean", 0.0)
    ae_f1 = overall.get("AE-INT8", {}).get("f1_score", {}).get("mean", 0.0)
    hy_f1 = overall.get("HYBRID", {}).get("f1_score", {}).get("mean", 0.0)
    lines.append("")
    lines.append(f"- Single AE-INT8 overall F1 = **{ae_f1:.3f}**")
    lines.append(f"- THRESH baseline F1 = {thr_f1:.3f}; OC-SVM F1 = {oc_f1:.3f}")
    lines.append(f"- HYBRID (rules + AE-INT8) overall F1 = **{hy_f1:.3f}** "
                 f"(beats THRESH: {hy_f1 > thr_f1}; beats OC-SVM: {hy_f1 > oc_f1})")

    # ---- Per-attack F1 before/after ----
    lines.append("\n## Per-attack F1: single AE-INT8 vs HYBRID\n")
    lines.append("| Attack | AE-INT8 F1 | HYBRID F1 | Heartbeat recall | Freq recall |")
    lines.append("|---|---|---|---|---|")
    for key, label in ATTACKS:
        a_v, a_s = f1(d, "AE-INT8", key)
        h_v, h_s = f1(d, "HYBRID", key)
        hb = rb.get(key, {}).get("heartbeat_recall_mean", 0.0)
        fc = rb.get(key, {}).get("freq_recall_mean", 0.0)
        lines.append(f"| {label} | {fmt(a_v, a_s)} | {fmt(h_v, h_s)} | "
                     f"{hb:.3f} | {fc:.3f} |")

    # ---- Per-attack latency before/after ----
    lines.append("\n## Per-attack detection latency (ms): single AE-INT8 vs HYBRID\n")
    lines.append("| Attack | AE-INT8 latency | HYBRID latency |")
    lines.append("|---|---|---|")
    for key, label in ATTACKS:
        a_v, a_s = lat(d, "detection_latency_ae_int8", key)
        h_v, h_s = lat(d, "detection_latency", key)
        lines.append(f"| {label} | {fmt_ms(a_v, a_s)} | {fmt_ms(h_v, h_s)} |")

    # ---- Rule thresholds ----
    lines.append("\n## Rule thresholds used\n")
    lines.append(f"- **Heartbeat (liveness) rule** -> targets A3. "
                 f"Flag if a monitored CAN ID is absent for > k x expected_period, "
                 f"with **k = {hc.get('heartbeat_k')}**. "
                 f"Session-boundary reset gap = {hc.get('reset_gap_us')} us.")
    lines.append(f"  - Learned expected periods (ms): {hc.get('expected_periods_ms')}")
    lines.append(f"- **Frequency-counter rule** -> targets A6. "
                 f"Flag if per-window frame count > mu + n*sigma with "
                 f"**n = {hc.get('freq_n_sigma')}**, or a never-seen CAN ID "
                 f"(e.g. 0x000 flood) appears >= {hc.get('unknown_id_min_count')} "
                 f"times in a window.")
    lines.append(f"  - Normal per-window global frame count: "
                 f"mu = {hc.get('global_count_mean')}, "
                 f"sigma = {hc.get('global_count_std')}, "
                 f"global threshold = {hc.get('global_count_threshold')} frames/window.")
    lines.append(f"- **Window**: {hc.get('window_duration_ms')} ms duration, "
                 f"{hc.get('window_stride_ms')} ms stride. **Fusion**: {hc.get('fusion')}.")

    # ---- Embedded cost of rule layers ----
    lines.append("\n## Embedded cost of the rule layers (STM32H743, Cortex-M7 @ 480 MHz)\n")
    lines.append(f"- Heartbeat state: {rule_res.get('heartbeat_state_bytes')} bytes "
                 f"(last-seen u32 + period u32 per monitored ID).")
    lines.append(f"- Frequency-counter state: {rule_res.get('freq_counter_state_bytes')} bytes "
                 f"(per-ID + global counters/thresholds).")
    lines.append(f"- **Total rule state: {rule_res.get('total_rule_state_bytes')} bytes "
                 f"({rule_res.get('total_rule_state_kb')} KB).**")
    lines.append(f"- Per-frame update: {rule_res.get('per_frame_update_cycles')} cycles "
                 f"(~{rule_res.get('per_frame_update_latency_us')} us).")
    lines.append(f"- Per-window liveness check: {rule_res.get('per_window_check_cycles')} cycles "
                 f"(~{rule_res.get('per_window_check_latency_us')} us).")
    lines.append(f"- Rule RAM overhead vs INT8 AE runtime RAM: "
                 f"{hyb_res.get('rule_overhead_ram_pct')} %.")

    out = RESULTS_DIR / "HYBRID_SUMMARY.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
