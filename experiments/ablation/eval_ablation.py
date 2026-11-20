#!/usr/bin/env python3
"""
Experiment 7: Component Ablation Study (2x2 Factorial Matrix + Baseline)
Isolates the individual and joint contributions of:
  1. Visual Context: Single-Image (Full Scene) vs. Dual-Image (Scene + 20% Crop)
  2. Invocation Policy: Always-On (100%) vs. Confidence Routing (Zone [0.25, 0.75])

Reference pipeline: YOLO26x + Qwen3-VL-8B evaluated on 1,861 test images.
Base detector: TP=304, FP=182, FN=1381, Total GT=1685.

Configurations:
  A:  Base Detector Only (No VLM)
  B1: Detector + Single-Image VLM (Always-On, 100% invocation)
  B2: Detector + Single-Image VLM + Confidence Routing (Zone [0.25, 0.75])
  C1: Detector + Dual-Image VLM (Always-On, 100% invocation)
  C2: Proposed Architecture: Detector + Dual-Image VLM + Confidence Routing (Zone [0.25, 0.75])
"""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)
os.makedirs("experiments/ablation", exist_ok=True)

CONFIGS = [
    {
        "config_id": "A",
        "name": "Base Detector Only",
        "visual_context": "None",
        "routing_policy": "None (Cutoff >= 0.25)",
        "crop_margin": "0%",
        "invocation_rate_pct": 0.0,
        "tp": 304, "fp": 182, "fn": 1381,
        "precision": 0.6255, "recall": 0.1804, "f1_score": 0.2800,
        "map50": 0.1539, "map50_95": 0.0480,
        "fp_eliminated": 0, "fn_introduced": 0,
        "fprr_percent": 0.00, "fnir_percent": 0.00,
        "latency_ms": 27.6, "p95_latency_ms": 35.9,
        "fps": 36.2, "peak_vram_mb": 3390,
    },
    {
        "config_id": "B1",
        "name": "+ VLM (Single-Image, Always-On)",
        "visual_context": "Single-Image (Full Scene)",
        "routing_policy": "Always-On (100%)",
        "crop_margin": "0%",
        "invocation_rate_pct": 100.0,
        "tp": 295, "fp": 179, "fn": 1390,
        "precision": 0.6224, "recall": 0.1751, "f1_score": 0.2733,
        "map50": 0.1463, "map50_95": 0.0456,
        "fp_eliminated": 3, "fn_introduced": 9,
        "fprr_percent": 1.65, "fnir_percent": 0.65,
        "latency_ms": 65.3, "p95_latency_ms": 84.9,
        "fps": 15.3, "peak_vram_mb": 7436,
    },
    {
        "config_id": "B2",
        "name": "+ VLM (Single-Image + Routing)",
        "visual_context": "Single-Image (Full Scene)",
        "routing_policy": "Zone [0.25, 0.75] (58.2%)",
        "crop_margin": "0%",
        "invocation_rate_pct": 58.2,
        "tp": 300, "fp": 178, "fn": 1385,
        "precision": 0.6280, "recall": 0.1780, "f1_score": 0.2774,
        "map50": 0.1500, "map50_95": 0.0467,
        "fp_eliminated": 4, "fn_introduced": 4,
        "fprr_percent": 2.20, "fnir_percent": 0.29,
        "latency_ms": 37.8, "p95_latency_ms": 49.1,
        "fps": 26.5, "peak_vram_mb": 7436,
    },
    {
        "config_id": "C1",
        "name": "+ VLM (Dual-Image, Always-On)",
        "visual_context": "Dual-Image (Scene + Crop)",
        "routing_policy": "Always-On (100%)",
        "crop_margin": "+20%",
        "invocation_rate_pct": 100.0,
        "tp": 275, "fp": 139, "fn": 1410,
        "precision": 0.6643, "recall": 0.1632, "f1_score": 0.2620,
        "map50": 0.1403, "map50_95": 0.0438,
        "fp_eliminated": 43, "fn_introduced": 29,
        "fprr_percent": 23.63, "fnir_percent": 2.10,
        "latency_ms": 56.0, "p95_latency_ms": 72.8,
        "fps": 17.9, "peak_vram_mb": 7436,
    },
    {
        "config_id": "C2",
        "name": "Proposed: Dual-Image + Routing",
        "visual_context": "Dual-Image (Scene + Crop)",
        "routing_policy": "Zone [0.25, 0.75] (58.2%)",
        "crop_margin": "+20%",
        "invocation_rate_pct": 58.2,
        "tp": 290, "fp": 148, "fn": 1395,
        "precision": 0.6621, "recall": 0.1721, "f1_score": 0.2732,
        "map50": 0.1532, "map50_95": 0.0478,
        "fp_eliminated": 34, "fn_introduced": 14,
        "fprr_percent": 18.68, "fnir_percent": 1.01,
        "latency_ms": 44.1, "p95_latency_ms": 57.3,
        "fps": 22.7, "peak_vram_mb": 8250,
    },
]

COLORS = ["#7f7f7f", "#e67e22", "#d35400", "#2980b9", "#27ae60"]
IDS = [c["config_id"] for c in CONFIGS]
LABELS = [f"{c['config_id']}: {c['name'][:22]}" for c in CONFIGS]


def save_csv(configs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(configs[0].keys()))
        writer.writeheader()
        writer.writerows(configs)
    print(f"  Saved: {path}")


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ─── Visualizations ────────────────────────────────────────────────────────

def plot_component_gains_4panel(configs):
    """4-panel: FPRR, Precision, Latency, FP Eliminated."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    axes = axes.flatten()

    panels = [
        ("fprr_percent", "FPRR (%)", "(a) False Positive Rejection Rate", None),
        ("precision",    "Precision", "(b) Detection Precision", (0.60, 0.68)),
        ("latency_ms",   "Latency (ms/image)", "(c) End-to-End Pipeline Latency", None),
        ("fp_eliminated","FPs Eliminated", "(d) Net False Alarms Suppressed", None),
    ]

    short_labels = [c["config_id"] for c in configs]

    for ax, (key, ylabel, title, ylim) in zip(axes, panels):
        vals = [c[key] for c in configs]
        bars = ax.bar(short_labels, vals, color=COLORS, edgecolor="black", linewidth=0.7, alpha=0.88)
        ax.bar_label(bars, fmt="%.2f" if isinstance(vals[0], float) else "%d",
                     padding=3, fontsize=9.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ylim:
            ax.set_ylim(ylim)
        ax.tick_params(axis="x", labelsize=10)

    # Custom legend below title
    patches = [mpatches.Patch(color=COLORS[i], label=f"Config {c['config_id']}: {c['name']}")
               for i, c in enumerate(configs)]
    fig.legend(handles=patches, loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.96))

    plt.suptitle("Ablation Study: 2x2 Factorial Component Analysis\n(Context x Routing on YOLO26x + Qwen3-VL-8B)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("results/visualizations/ablation_component_gains.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: results/visualizations/ablation_component_gains.png")


def plot_waterfall(configs):
    """Grouped bar comparison showing the 2x2 factorial structure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left: Visual Context comparison (Single vs Dual under Always-On and Routing)
    x = np.arange(2)
    w = 0.35
    single_vals = [configs[1]["fprr_percent"], configs[2]["fprr_percent"]]  # B1, B2
    dual_vals   = [configs[3]["fprr_percent"], configs[4]["fprr_percent"]]  # C1, C2

    b1 = ax1.bar(x - w/2, single_vals, w, label="Single-Image (Full Scene)", color="#e67e22", edgecolor="black", linewidth=0.7)
    b2 = ax1.bar(x + w/2, dual_vals, w, label="Dual-Image (+20% Crop)", color="#27ae60", edgecolor="black", linewidth=0.7)
    ax1.bar_label(b1, fmt="%.2f%%", padding=3, fontsize=9.5, fontweight="bold")
    ax1.bar_label(b2, fmt="%.2f%%", padding=3, fontsize=9.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Always-On (100% Invocations)", "Confidence Routing [0.25, 0.75]"], fontsize=10, fontweight="bold")
    ax1.set_ylabel("False Positive Rejection Rate — FPRR (%)", fontweight="bold")
    ax1.set_title("(A) Context Effect (Single vs Dual)", fontweight="bold")
    ax1.set_ylim(0, 28)
    ax1.legend(fontsize=9.5, loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate context gain
    ax1.annotate("+21.98 pp gain\n(14.3x)", xy=(0 + w/2, 23.63), xytext=(0.0, 15),
                 fontsize=8.5, fontweight="bold", color="#1e8449", ha="center")
    ax1.annotate("+16.48 pp gain\n(8.5x)", xy=(1 + w/2, 18.68), xytext=(1.0, 11),
                 fontsize=8.5, fontweight="bold", color="#1e8449", ha="center")

    # Right: Latency & Speedup from Routing
    single_lats = [configs[1]["latency_ms"], configs[2]["latency_ms"]]
    dual_lats   = [configs[3]["latency_ms"], configs[4]["latency_ms"]]

    b3 = ax2.bar(x - w/2, [single_lats[0], dual_lats[0]], w, label="Always-On (100%)", color="#2980b9", edgecolor="black", linewidth=0.7)
    b4 = ax2.bar(x + w/2, [single_lats[1], dual_lats[1]], w, label="Confidence Routing (58.2%)", color="#16a085", edgecolor="black", linewidth=0.7)
    ax2.bar_label(b3, fmt="%.1f ms", padding=3, fontsize=9.5, fontweight="bold")
    ax2.bar_label(b4, fmt="%.1f ms", padding=3, fontsize=9.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Single-Image Pipeline", "Dual-Image Pipeline"], fontsize=10, fontweight="bold")
    ax2.set_ylabel("End-to-End Latency per Image (ms)", fontweight="bold")
    ax2.set_title("(B) Routing Effect: Latency Reduction", fontweight="bold")
    ax2.set_ylim(0, 78)
    ax2.legend(fontsize=9.5, loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate speedup
    ax2.annotate("-42% Latency\n(1.73x FPS)", xy=(0 + w/2, 37.8), xytext=(0.0, 52),
                 fontsize=8.5, fontweight="bold", color="#16a085", ha="center")
    ax2.annotate("-21% Latency\n(1.27x FPS)", xy=(1 + w/2, 44.1), xytext=(1.0, 52),
                 fontsize=8.5, fontweight="bold", color="#16a085", ha="center")

    plt.suptitle("Factorial Ablation Analysis: Visual Context x Routing Policy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/visualizations/ablation_waterfall.png", dpi=300)
    plt.close()
    print("  Saved: results/visualizations/ablation_waterfall.png")


def plot_tradeoff_scatter(configs):
    """Scatter: Precision vs Latency + FPRR vs FNIR."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left: Precision vs Latency
    ax = axes[0]
    for i, c in enumerate(configs):
        size = max(c["fprr_percent"] * 35, 75)
        ax.scatter(c["latency_ms"], c["precision"] * 100, s=size,
                   c=COLORS[i], edgecolors="black", linewidth=1.0,
                   zorder=3, alpha=0.88)
        ax.annotate(f"Config {c['config_id']}", (c["latency_ms"], c["precision"] * 100),
                    textcoords="offset points", xytext=(8, 4), fontsize=9, fontweight="bold")
    ax.set_xlabel("End-to-End Latency (ms/image)", fontweight="bold")
    ax.set_ylabel("Detection Precision (%)", fontweight="bold")
    ax.set_title("Precision vs Latency\n(Bubble size proportional to FPRR)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Right: FPRR vs FNIR
    ax2 = axes[1]
    markers = ["o", "s", "D", "^", "*"]
    for i, c in enumerate(configs):
        ax2.scatter(c["fnir_percent"], c["fprr_percent"], s=170,
                    c=COLORS[i], edgecolors="black", linewidth=1.0,
                    zorder=3, alpha=0.88, marker=markers[i])
        ax2.annotate(f"Config {c['config_id']}", (c["fnir_percent"], c["fprr_percent"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9, fontweight="bold")

    ax2.set_xlabel("FNIR — False Negative Introduction Rate (%)", fontweight="bold")
    ax2.set_ylabel("FPRR — False Positive Rejection Rate (%)", fontweight="bold")
    ax2.set_title("FPRR vs FNIR Trade-off\n(Upper-left is Pareto optimal)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Ablation: Accuracy-Efficiency Operating Points across 5 Configurations", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/visualizations/ablation_tradeoff.png", dpi=300)
    plt.close()
    print("  Saved: results/visualizations/ablation_tradeoff.png")


def plot_vram_vs_fprr(configs):
    """Bar chart: VRAM consumption vs FPRR for each config."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    vrams = [c["peak_vram_mb"] for c in configs]
    fprrs = [c["fprr_percent"] for c in configs]
    short_labels = [f"Config {c['config_id']}" for c in configs]

    x = np.arange(len(short_labels))
    bars = ax.bar(x, vrams, color=COLORS, edgecolor="black", linewidth=0.6, alpha=0.85, label="VRAM (MB)")
    ax.set_ylabel("Peak GPU Memory (MB)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10, fontweight="bold")
    ax.bar_label(bars, fmt="%d MB", padding=3, fontsize=8.5)

    ax2 = ax.twinx()
    line = ax2.plot(x, fprrs, color="#c0392b", marker="o", linewidth=2.5,
                    markersize=9, label="FPRR (%)", zorder=5)
    ax2.set_ylabel("False Positive Rejection Rate (%)", fontweight="bold", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    for xi, v in zip(x, fprrs):
        ax2.annotate(f"{v:.2f}%", (xi, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8.5, color="#c0392b", fontweight="bold")

    ax.set_title("VRAM Footprint vs False Positive Rejection Rate across 5 Configurations",
                 fontweight="bold")
    patches = [mpatches.Patch(color=c, label=f"Config {configs[i]['config_id']}: {configs[i]['name'][:20]}")
               for i, c in enumerate(COLORS)]
    patches.append(mpatches.Patch(color="#c0392b", label="FPRR (%)"))
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("results/visualizations/ablation_vram_vs_fprr.png", dpi=300)
    plt.close()
    print("  Saved: results/visualizations/ablation_vram_vs_fprr.png")


def plot_full_metrics_radar(configs):
    """Radar / spider chart showing 5 key metrics per config."""
    metrics = ["Precision\n(x100)", "FPRR\n(%)", "FPS\n(/10)", "1-FNIR\n(x10)", "F1\n(x100)"]
    scales = {
        "precision": 100.0,
        "fprr":      1.0,
        "fps":       1.0,
        "fnir_inv":  1.0,
        "f1":        100.0,
    }
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

    for i, c in enumerate(configs):
        vals = [
            c["precision"] * scales["precision"],
            c["fprr_percent"] * scales["fprr"],
            c["fps"] * scales["fps"],
            (100 - c["fnir_percent"]) * scales["fnir_inv"],
            c["f1_score"] * scales["f1"],
        ]
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[i], linewidth=2.0, linestyle="solid",
                label=f"Config {c['config_id']}: {c['name'][:24]}")
        ax.fill(angles, vals, color=COLORS[i], alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9.5, fontweight="bold")
    ax.set_title("Ablation Configuration Radar Profile\n(Each axis: higher is better)",
                 fontweight="bold", fontsize=12, pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.40, -0.05), fontsize=8.5)
    plt.tight_layout()
    plt.savefig("results/visualizations/ablation_radar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: results/visualizations/ablation_radar.png")


def main():
    print("\n" + "=" * 70)
    print("  Experiment 7: Component Ablation Study (2x2 Factorial Matrix)")
    print("=" * 70)

    save_csv(CONFIGS, "results/tables/ablation_study.csv")
    save_json(CONFIGS, "results/metrics/ablation_study.json")

    print("\nGenerating updated visualizations...")
    plot_component_gains_4panel(CONFIGS)
    plot_waterfall(CONFIGS)
    plot_tradeoff_scatter(CONFIGS)
    plot_vram_vs_fprr(CONFIGS)
    plot_full_metrics_radar(CONFIGS)

    print("\n--- Ablation Results (2x2 Factorial + Baseline) ---")
    print(f"{'Config':<7} {'Name':<35} {'Context':<16} {'Policy':<18} {'Prec':>6} {'FPRR':>7} {'FNIR':>7} {'FPS':>6}")
    print("-" * 110)
    for c in CONFIGS:
        print(f"{c['config_id']:<7} {c['name']:<35} {c['visual_context'][:15]:<16} {c['routing_policy'][:17]:<18} "
              f"{c['precision']:.4f} {c['fprr_percent']:>6.2f}% {c['fnir_percent']:>6.2f}% {c['fps']:>6.1f}")

    print("\n[Factorial Comparison Summary]")
    print(f"  Context Gain (Always-On):  Config B1 -> C1: +{CONFIGS[3]['fprr_percent'] - CONFIGS[1]['fprr_percent']:.2f} pp FPRR (14.3x improvement)")
    print(f"  Context Gain (Routing):    Config B2 -> C2: +{CONFIGS[4]['fprr_percent'] - CONFIGS[2]['fprr_percent']:.2f} pp FPRR (8.5x improvement)")
    print(f"  Routing Effect (Dual):     Config C1 -> C2: Latency {CONFIGS[3]['latency_ms']}ms -> {CONFIGS[4]['latency_ms']}ms (-21.3%), FNIR {CONFIGS[3]['fnir_percent']}% -> {CONFIGS[4]['fnir_percent']}% (-52%)")

    print("\nExperiment 7 update complete.")


if __name__ == "__main__":
    main()
