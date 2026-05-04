#!/usr/bin/env python3
"""
Generate all paper figures programmatically.

Figures are publication-quality: PDF format, serif font, 10pt,
high resolution (300 dpi), accessible color palette.

Paper: Lightweight Autoencoder-Based Anomaly Detection for CAN Bus
       in Competition Motorcycles Deployed on ARM Cortex-M7
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.ticker as ticker
from pathlib import Path

# Publication-quality settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"

# Accessible color palette (avoid red-green for colorblind)
COLORS = {
    "THRESH": "#4477AA",     # Blue
    "OC-SVM": "#EE6677",     # Red/pink
    "IF": "#228833",         # Green
    "LSTM-AE": "#CCBB44",    # Yellow
    "AE-FP32": "#AA3377",    # Purple
    "AE-INT8": "#66CCEE",    # Cyan
}

ATTACK_SHORT = {
    "A1_tps_spoofing": "A1\nTPS",
    "A2_lean_injection": "A2\nLean",
    "A3_bms_disappearance": "A3\nBMS",
    "A4_replay": "A4\nReplay",
    "A5_fuzzing": "A5\nFuzz",
    "A6_dos_flooding": "A6\nDoS",
}

ATTACK_LABELS = {
    "A1_tps_spoofing": "A1: TPS Spoofing",
    "A2_lean_injection": "A2: Lean Angle Inj.",
    "A3_bms_disappearance": "A3: BMS Disappear.",
    "A4_replay": "A4: Replay",
    "A5_fuzzing": "A5: Fuzzing",
    "A6_dos_flooding": "A6: DoS Flooding",
}


def fig_system_architecture(results: dict):
    """Figure 1: System architecture block diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # Colors
    c_can = "#E8E8E8"
    c_proc = "#D4E6F1"
    c_model = "#D5F5E3"
    c_alert = "#FADBD8"

    # CAN Bus bar
    ax.add_patch(FancyBboxPatch((0.2, 1.2), 1.3, 0.6,
                                boxstyle="round,pad=0.1", facecolor=c_can, edgecolor="black", lw=1))
    ax.text(0.85, 1.5, "CAN Bus\n(250 kbps)", ha="center", va="center", fontsize=7, weight="bold")

    # Hardware FIFO
    ax.add_patch(FancyBboxPatch((2.0, 1.2), 1.2, 0.6,
                                boxstyle="round,pad=0.1", facecolor=c_can, edgecolor="black", lw=1))
    ax.text(2.6, 1.5, "FDCAN\nHW FIFO", ha="center", va="center", fontsize=7)

    # Circular buffer
    ax.add_patch(FancyBboxPatch((3.7, 1.2), 1.2, 0.6,
                                boxstyle="round,pad=0.1", facecolor=c_proc, edgecolor="black", lw=1))
    ax.text(4.3, 1.5, "Circular\nBuffer", ha="center", va="center", fontsize=7)

    # Feature extraction
    ax.add_patch(FancyBboxPatch((5.4, 1.2), 1.2, 0.6,
                                boxstyle="round,pad=0.1", facecolor=c_proc, edgecolor="black", lw=1))
    ax.text(6.0, 1.5, "Feature\nExtraction", ha="center", va="center", fontsize=7)

    # Autoencoder
    ax.add_patch(FancyBboxPatch((7.1, 1.0), 1.3, 1.0,
                                boxstyle="round,pad=0.1", facecolor=c_model, edgecolor="black", lw=1.5))
    ax.text(7.75, 1.65, "INT8\nAutoencoder", ha="center", va="center", fontsize=7, weight="bold")
    ax.text(7.75, 1.2, "80-40-20-10\n-20-40-80", ha="center", va="center", fontsize=6, style="italic")

    # Threshold
    ax.add_patch(FancyBboxPatch((8.8, 1.2), 1.0, 0.6,
                                boxstyle="round,pad=0.1", facecolor=c_alert, edgecolor="black", lw=1))
    ax.text(9.3, 1.5, "e > tau?\nAlert", ha="center", va="center", fontsize=7, weight="bold")

    # Arrows
    arrow_style = "Simple,tail_width=0.5,head_width=4,head_length=3"
    for x1, x2 in [(1.5, 2.0), (3.2, 3.7), (4.9, 5.4), (6.6, 7.1), (8.4, 8.8)]:
        ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Labels above
    ax.text(0.85, 2.2, "CAN\nFrames", ha="center", va="center", fontsize=7, color="gray")
    ax.text(4.3, 2.2, "100 ms\nwindow", ha="center", va="center", fontsize=7, color="gray")
    ax.text(6.0, 2.2, "80-dim\nvector", ha="center", va="center", fontsize=7, color="gray")

    # MCU label
    ax.text(5.0, 0.4, "STM32H743 (ARM Cortex-M7 @ 480 MHz)", ha="center", va="center",
            fontsize=8, style="italic", color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F8F8F8", edgecolor="#CCCCCC"))

    fig.savefig(FIGURES_DIR / "fig_architecture.pdf")
    plt.close(fig)
    print("  Generated: fig_architecture.pdf")


def fig_roc_curves(results: dict):
    """Figure 2: ROC curves for all methods."""
    roc_data = results.get("figure_data", {}).get("roc_curves", {})
    if not roc_data:
        print("  SKIP: No ROC data available for fig_roc_curves.pdf")
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0))

    # Plot random baseline
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.8, label="Random")

    methods_order = ["THRESH", "OC-SVM", "IF", "AE-FP32", "AE-INT8"]
    for method in methods_order:
        if method in roc_data:
            rd = roc_data[method]
            label = f"{method} (AUC={rd['auc']:.3f})"
            ax.plot(rd["fpr"], rd["tpr"], color=COLORS.get(method, "#333333"),
                    lw=1.5, label=label)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    fig.savefig(FIGURES_DIR / "fig_roc_curves.pdf")
    plt.close(fig)
    print("  Generated: fig_roc_curves.pdf")


def fig_reconstruction_error(results: dict):
    """Figure 3: Reconstruction error distribution (normal vs attack)."""
    recon_data = results.get("figure_data", {}).get("reconstruction_errors", {})
    if not recon_data or not recon_data.get("normal") or not recon_data.get("attack"):
        print("  SKIP: No reconstruction error data for fig_reconstruction_error.pdf")
        return

    normal_errors = np.array(recon_data["normal"])
    attack_errors = np.array(recon_data["attack"])
    threshold = recon_data.get("threshold", 0)

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.5))

    # Compute histogram range
    all_errors = np.concatenate([normal_errors, attack_errors])
    max_val = min(np.percentile(all_errors, 99.5), np.max(attack_errors))
    bins = np.linspace(0, max_val, 80)

    ax.hist(normal_errors, bins=bins, density=True, alpha=0.7,
            color="#4477AA", label="Normal traffic", edgecolor="none")
    ax.hist(attack_errors, bins=bins, density=True, alpha=0.7,
            color="#EE6677", label="Attack traffic", edgecolor="none")

    if threshold > 0:
        ax.axvline(threshold, color="#228833", linestyle="--", lw=1.5,
                   label=f"Threshold (tau={threshold:.4f})")

    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, max_val)

    fig.savefig(FIGURES_DIR / "fig_reconstruction_error.pdf")
    plt.close(fig)
    print("  Generated: fig_reconstruction_error.pdf")


def fig_per_attack_f1(results: dict):
    """Figure 4: Per-attack F1-score grouped bar chart."""
    per_attack = results.get("per_attack", {})
    if not per_attack:
        print("  SKIP: No per-attack data for fig_attack_comparison.pdf")
        return

    attack_types = list(ATTACK_SHORT.keys())
    methods = ["THRESH", "OC-SVM", "IF", "LSTM-AE", "AE-INT8"]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.5))

    n_attacks = len(attack_types)
    n_methods = len(methods)
    width = 0.14
    x = np.arange(n_attacks)

    for i, method in enumerate(methods):
        f1_means = []
        f1_stds = []
        for attack_type in attack_types:
            if method in per_attack and attack_type in per_attack[method]:
                f1_means.append(per_attack[method][attack_type]["f1_mean"])
                f1_stds.append(per_attack[method][attack_type].get("f1_std", 0))
            else:
                f1_means.append(0)
                f1_stds.append(0)

        bars = ax.bar(x + i * width - (n_methods - 1) * width / 2,
                      f1_means, width, yerr=f1_stds,
                      label=method, color=COLORS.get(method, "#888888"),
                      edgecolor="white", lw=0.5, capsize=2, error_kw={"lw": 0.8})

    ax.set_xlabel("Attack Type")
    ax.set_ylabel("F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_SHORT[a] for a in attack_types], fontsize=8)
    ax.legend(loc="lower left", ncol=3, fontsize=7, framealpha=0.9)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    fig.savefig(FIGURES_DIR / "fig_attack_comparison.pdf")
    plt.close(fig)
    print("  Generated: fig_attack_comparison.pdf")


def fig_quantization_impact(results: dict):
    """Figure 5: Quantization impact (model size vs F1)."""
    quant = results.get("quantization", {})
    overall = results.get("overall", {})

    if not quant or not overall:
        print("  SKIP: No quantization data for fig_quantization_impact.pdf")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Left: Model size vs F1
    fp32_f1 = overall.get("AE-FP32", {}).get("f1_score", {}).get("mean", 0)
    int8_f1 = overall.get("AE-INT8", {}).get("f1_score", {}).get("mean", 0)
    fp32_size = quant.get("fp32_size_kb", 0)
    int8_size = quant.get("int8_size_kb", 0)

    if fp32_size > 0 and int8_size > 0:
        ax1.scatter([fp32_size], [fp32_f1], s=120, c="#AA3377", marker="s",
                    label="FP32", zorder=5, edgecolors="black", lw=0.5)
        ax1.scatter([int8_size], [int8_f1], s=120, c="#66CCEE", marker="D",
                    label="INT8", zorder=5, edgecolors="black", lw=0.5)

        # Arrow from FP32 to INT8
        ax1.annotate("", xy=(int8_size, int8_f1), xytext=(fp32_size, fp32_f1),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5,
                                    connectionstyle="arc3,rad=-0.2"))
        ax1.annotate(f"{quant.get('size_reduction', 4):.0f}x smaller",
                     xy=((fp32_size + int8_size) / 2, (fp32_f1 + int8_f1) / 2),
                     fontsize=8, ha="center", va="bottom", color="gray")

    ax1.set_xlabel("Model Size (KB)")
    ax1.set_ylabel("F1-Score")
    ax1.legend(loc="lower right", fontsize=8)
    y_min = min(fp32_f1, int8_f1) - 0.05 if fp32_f1 > 0 else 0.85
    ax1.set_ylim(y_min, 1.01)

    # Right: Resource comparison bar chart
    categories = ["Model\nSize (KB)", "Peak\nRAM (KB)", "Latency\n(ms)"]
    embedded = results.get("embedded_resources", {})
    fp32_res = embedded.get("fp32", {})
    int8_res = embedded.get("int8", {})

    fp32_vals = [fp32_res.get("model_weights_flash_kb", 0),
                 fp32_res.get("runtime_ram_kb", 0),
                 fp32_res.get("total_detection_latency_ms", 0)]
    int8_vals = [int8_res.get("model_weights_flash_kb", 0),
                 int8_res.get("runtime_ram_kb", 0),
                 int8_res.get("total_detection_latency_ms", 0)]

    x = np.arange(len(categories))
    width = 0.3
    ax2.bar(x - width/2, fp32_vals, width, label="FP32", color="#AA3377", edgecolor="white")
    ax2.bar(x + width/2, int8_vals, width, label="INT8", color="#66CCEE", edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=8)
    ax2.set_ylabel("Value")
    ax2.legend(loc="upper right", fontsize=8)

    # Add 64 KB RAM budget line
    budget_kb = 64
    ax2.axhline(budget_kb, color="red", linestyle=":", lw=1, alpha=0.5)
    ax2.text(len(categories) - 0.5, budget_kb + 1, "64 KB budget",
             fontsize=7, color="red", alpha=0.7)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_quantization_impact.pdf")
    plt.close(fig)
    print("  Generated: fig_quantization_impact.pdf")


def generate_all_figures(results: dict):
    """Generate all paper figures from aggregated results."""
    print("\nGenerating figures...")

    fig_system_architecture(results)
    fig_roc_curves(results)
    fig_reconstruction_error(results)
    fig_per_attack_f1(results)
    fig_quantization_impact(results)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


def main():
    """Generate figures from saved results."""
    results_file = RESULTS_DIR / "aggregated_results.json"
    if not results_file.exists():
        print(f"ERROR: {results_file} not found. Run run_all.py first.")
        return

    with open(results_file) as f:
        results = json.load(f)

    generate_all_figures(results)


if __name__ == "__main__":
    main()
