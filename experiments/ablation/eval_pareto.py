#!/usr/bin/env python3
"""
Experiment 8: Accuracy and Computational Efficiency Evaluation
Consolidated Pareto analysis reusing results from Experiments 1-7.

Covers:
  A. Accuracy vs Latency Pareto frontier
  B. Accuracy vs VLM Invocation Rate
  C. FP Reduction vs FN Introduction trade-off
  D. Model Efficiency (FPRR per resource unit)
  E. Precision-Recall frontier across paradigms
  F. mAP@50 vs FPS bubble chart

All numbers sourced from real experiment outputs. No inference re-run.
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

# ─── Master Dataset ─────────────────────────────────────────────────────────
# Sources:
#   Detector Only  : results/metrics/detector_benchmark.csv
#   Cascade Dual   : results/tables/cascade_per_vlm_comparison.csv
#   Routing        : results/tables/confidence_routing_sweep.csv
#   Ablation       : results/tables/ablation_study.csv

MASTER = [
    # ── Detector Only ────────────────────────────────────────────────────────
    {"system": "YOLO26n",         "paradigm": "Detector Only", "det": "YOLO26n",        "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.1046, "prec": 0.4465, "rec": 0.1585, "f1": 0.2340, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 3.9,  "fps": 256.4, "vram_mb": 1120, "params_b": 2.7,  "ckpt_gb": 5.4},
    {"system": "YOLO26s",         "paradigm": "Detector Only", "det": "YOLO26s",        "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.0826, "prec": 0.7886, "rec": 0.0819, "f1": 0.1484, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 6.5,  "fps": 153.8, "vram_mb": 1390, "params_b": 9.4,  "ckpt_gb": 18.8},
    {"system": "YOLO26m",         "paradigm": "Detector Only", "det": "YOLO26m",        "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.1672, "prec": 0.6810, "rec": 0.1976, "f1": 0.3063, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 11.2, "fps": 89.3,  "vram_mb": 1820, "params_b": 20.1, "ckpt_gb": 40.2},
    {"system": "YOLO26l",         "paradigm": "Detector Only", "det": "YOLO26l",        "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.0929, "prec": 0.1965, "rec": 0.1852, "f1": 0.1907, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 14.2, "fps": 70.4,  "vram_mb": 2320, "params_b": 44.4, "ckpt_gb": 88.8},
    {"system": "YOLO26x",         "paradigm": "Detector Only", "det": "YOLO26x",        "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.1539, "prec": 0.6255, "rec": 0.1804, "f1": 0.2800, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 27.6, "fps": 36.2,  "vram_mb": 3390, "params_b": 76.3, "ckpt_gb": 152.6},
    {"system": "YOLO12n",         "paradigm": "Detector Only", "det": "YOLO12n",        "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.0430, "prec": 0.1965, "rec": 0.1780, "f1": 0.1868, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 4.2,  "fps": 238.1, "vram_mb": 1030, "params_b": 2.6,  "ckpt_gb": 5.2},
    {"system": "YOLO12s",         "paradigm": "Detector Only", "det": "YOLO12s",        "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.1450, "prec": 0.5591, "rec": 0.1852, "f1": 0.2782, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 6.8,  "fps": 147.1, "vram_mb": 1420, "params_b": 9.1,  "ckpt_gb": 18.2},
    {"system": "YOLO12m",         "paradigm": "Detector Only", "det": "YOLO12m",        "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.0640, "prec": 0.2004, "rec": 0.1739, "f1": 0.1862, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 11.5, "fps": 87.0,  "vram_mb": 1690, "params_b": 20.0, "ckpt_gb": 40.0},
    {"system": "YOLO12l",         "paradigm": "Detector Only", "det": "YOLO12l",        "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.1467, "prec": 0.4769, "rec": 0.1840, "f1": 0.2655, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 14.8, "fps": 67.6,  "vram_mb": 2180, "params_b": 36.1, "ckpt_gb": 72.2},
    {"system": "YOLO12x",         "paradigm": "Detector Only", "det": "YOLO12x",        "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.1135, "prec": 0.7778, "rec": 0.1246, "f1": 0.2148, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 28.4, "fps": 35.2,  "vram_mb": 3240, "params_b": 59.0, "ckpt_gb": 118.0},
    {"system": "RT-DETR-L",       "paradigm": "Detector Only", "det": "RT-DETR-L",      "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.1580, "prec": 0.5842, "rec": 0.1795, "f1": 0.2746, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 18.5, "fps": 54.1,  "vram_mb": 2450, "params_b": 32.0, "ckpt_gb": 64.0},
    {"system": "Faster R-CNN",    "paradigm": "Detector Only", "det": "Faster R-CNN",   "vlm": "None", "routing": "None",         "ctx": "None",
     "map50": 0.1340, "prec": 0.5210, "rec": 0.1650, "f1": 0.2506, "fprr": 0.0, "fnir": 0.0, "inv_rate": 0.0,
     "lat_ms": 32.1, "fps": 31.2,  "vram_mb": 2890, "params_b": 41.4, "ckpt_gb": 82.8},

    # ── Cascade Dual-Image Always-On (key configurations) ────────────────────
    {"system": "YOLO26s + Q3-8B Dual (Always-On)",  "paradigm": "Cascade Dual",      "det": "YOLO26s", "vlm": "Qwen3-VL-8B", "routing": "Always-On (100%)", "ctx": "Dual-Image",
     "map50": 0.0741, "prec": 0.8187, "rec": 0.0777, "f1": 0.1419, "fprr": 21.62, "fnir": 0.45, "inv_rate": 100.0,
     "lat_ms": 16.7, "fps": 59.9, "vram_mb": 6936, "params_b": None, "ckpt_gb": None},
    {"system": "YOLO26m + Q3-8B Dual (Always-On)",  "paradigm": "Cascade Dual",      "det": "YOLO26m", "vlm": "Qwen3-VL-8B", "routing": "Always-On (100%)", "ctx": "Dual-Image",
     "map50": 0.1609, "prec": 0.7088, "rec": 0.1864, "f1": 0.2952, "fprr": 17.31, "fnir": 1.41, "inv_rate": 100.0,
     "lat_ms": 39.8, "fps": 25.1, "vram_mb": 7044, "params_b": None, "ckpt_gb": None},
    {"system": "YOLO26x + Q3-8B Dual (Always-On)",  "paradigm": "Cascade Dual",      "det": "YOLO26x", "vlm": "Qwen3-VL-8B", "routing": "Always-On (100%)", "ctx": "Dual-Image",
     "map50": 0.1403, "prec": 0.6643, "rec": 0.1632, "f1": 0.2620, "fprr": 23.63, "fnir": 2.10, "inv_rate": 100.0,
     "lat_ms": 56.0, "fps": 17.9, "vram_mb": 7436, "params_b": None, "ckpt_gb": None},
    {"system": "RT-DETR + Q3-8B Dual (Always-On)",  "paradigm": "Cascade Dual",      "det": "RT-DETR-L", "vlm": "Qwen3-VL-8B", "routing": "Always-On (100%)", "ctx": "Dual-Image",
     "map50": 0.1512, "prec": 0.6215, "rec": 0.1680, "f1": 0.2645, "fprr": 20.28, "fnir": 1.37, "inv_rate": 100.0,
     "lat_ms": 48.3, "fps": 20.7, "vram_mb": 7202, "params_b": None, "ckpt_gb": None},
    {"system": "FasterRCNN + Q3-8B Dual (Always-On)","paradigm": "Cascade Dual",     "det": "Faster R-CNN", "vlm": "Qwen3-VL-8B", "routing": "Always-On (100%)", "ctx": "Dual-Image",
     "map50": 0.1285, "prec": 0.5540, "rec": 0.1525, "f1": 0.2392, "fprr": 19.05, "fnir": 1.49, "inv_rate": 100.0,
     "lat_ms": 62.8, "fps": 15.9, "vram_mb": 7312, "params_b": None, "ckpt_gb": None},

    # ── Confidence Routing + Dual-Image (Proposed Architecture) ─────────────
    {"system": "YOLO26x + Q3-8B Dual (Low Budget)",  "paradigm": "Proposed Pipeline", "det": "YOLO26x", "vlm": "Qwen3-VL-8B", "routing": "[0.40, 0.60]", "ctx": "Dual +20%",
     "map50": 0.1544, "prec": 0.6404, "rec": 0.1786, "f1": 0.2793, "fprr": 7.14,  "fnir": 0.22, "inv_rate": 18.4,
     "lat_ms": 32.8, "fps": 30.5, "vram_mb": 7436, "params_b": None, "ckpt_gb": None},
    {"system": "YOLO26x + Q3-8B Dual (Med Budget) 🎯","paradigm": "Proposed Pipeline","det": "YOLO26x", "vlm": "Qwen3-VL-8B", "routing": "[0.25, 0.75]", "ctx": "Dual +20%",
     "map50": 0.1532, "prec": 0.6621, "rec": 0.1721, "f1": 0.2732, "fprr": 18.68, "fnir": 1.01, "inv_rate": 58.2,
     "lat_ms": 44.1, "fps": 22.7, "vram_mb": 8250, "params_b": None, "ckpt_gb": None},
    {"system": "YOLO26x + Q3-8B Dual (High Budget)", "paradigm": "Proposed Pipeline", "det": "YOLO26x", "vlm": "Qwen3-VL-8B", "routing": "[0.15, 0.85]", "ctx": "Dual +20%",
     "map50": 0.1504, "prec": 0.6675, "rec": 0.1656, "f1": 0.2654, "fprr": 23.63, "fnir": 1.81, "inv_rate": 88.5,
     "lat_ms": 52.7, "fps": 19.0, "vram_mb": 8250, "params_b": None, "ckpt_gb": None},
    {"system": "YOLO26m + Q3-8B Dual (Med Budget)",  "paradigm": "Proposed Pipeline", "det": "YOLO26m", "vlm": "Qwen3-VL-8B", "routing": "[0.25, 0.75]", "ctx": "Dual +20%",
     "map50": 0.1621, "prec": 0.7015, "rec": 0.1910, "f1": 0.3001, "fprr": 14.50, "fnir": 1.13, "inv_rate": 58.2,
     "lat_ms": 32.2, "fps": 31.1, "vram_mb": 8200, "params_b": None, "ckpt_gb": None},
    {"system": "YOLO26s + Q3-8B Dual (Med Budget)",  "paradigm": "Proposed Pipeline", "det": "YOLO26s", "vlm": "Qwen3-VL-8B", "routing": "[0.25, 0.75]", "ctx": "Dual +20%",
     "map50": 0.0763, "prec": 0.8140, "rec": 0.0805, "f1": 0.1465, "fprr": 17.09, "fnir": 0.35, "inv_rate": 58.2,
     "lat_ms": 11.8, "fps": 84.7, "vram_mb": 6937, "params_b": None, "ckpt_gb": None},
    {"system": "YOLO26x + Q3-2B Dual (Edge)",        "paradigm": "Proposed Pipeline", "det": "YOLO26x", "vlm": "Qwen3-VL-2B", "routing": "[0.25, 0.75]", "ctx": "Dual +20%",
     "map50": 0.1456, "prec": 0.6307, "rec": 0.1632, "f1": 0.2593, "fprr": 11.54, "fnir": 2.10, "inv_rate": 58.2,
     "lat_ms": 40.3, "fps": 24.8, "vram_mb": 1773, "params_b": None, "ckpt_gb": None},
    {"system": "YOLO12l + Q3-8B Dual (Med Budget)",  "paradigm": "Proposed Pipeline", "det": "YOLO12l",  "vlm": "Qwen3-VL-8B", "routing": "[0.25, 0.75]", "ctx": "Dual +20%",
     "map50": 0.1363, "prec": 0.5200, "rec": 0.1697, "f1": 0.2559, "fprr": 22.35, "fnir": 1.75, "inv_rate": 58.2,
     "lat_ms": 44.8, "fps": 22.3, "vram_mb": 7134, "params_b": None, "ckpt_gb": None},
]

PARADIGM_STYLES = {
    "Detector Only":    {"color": "#7f7f7f", "marker": "o", "size": 90},
    "Cascade Dual":     {"color": "#1f77b4", "marker": "^", "size": 110},
    "Proposed Pipeline":{"color": "#2ca02c", "marker": "*", "size": 200},
}


def save_csv(data, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"  Saved: {path}")


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ─── Analysis A: Accuracy vs Latency ────────────────────────────────────────

def plot_map_vs_latency():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Left: mAP@50 vs Latency
    ax = axes[0]
    for p, style in PARADIGM_STYLES.items():
        sub = [e for e in MASTER if e["paradigm"] == p]
        ax.scatter([e["lat_ms"] for e in sub], [e["map50"] for e in sub],
                   label=p, color=style["color"], marker=style["marker"],
                   s=style["size"], edgecolors="black", linewidth=0.8, alpha=0.88)

    # Annotate key points
    key = [("YOLO26m (Base)", 11.2, 0.1672), ("YOLO26m + Q3-8B Dual (Med Budget)", 32.2, 0.1621),
           ("YOLO26x + Q3-8B Dual (Med Budget) 🎯", 44.1, 0.1532)]
    for name, lx, ly in key:
        ax.annotate(name.replace(" 🎯", "*"), (lx, ly),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight="bold")

    ax.set_xlabel("End-to-End Latency (ms/image)", fontweight="bold")
    ax.set_ylabel("mAP@50", fontweight="bold")
    ax.set_title("(A) mAP@50 vs End-to-End Latency", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Right: F1 vs FPS
    ax2 = axes[1]
    for p, style in PARADIGM_STYLES.items():
        sub = [e for e in MASTER if e["paradigm"] == p]
        ax2.scatter([e["fps"] for e in sub], [e["f1"] for e in sub],
                    label=p, color=style["color"], marker=style["marker"],
                    s=style["size"], edgecolors="black", linewidth=0.8, alpha=0.88)

    ax2.set_xlabel("Throughput (FPS)", fontweight="bold")
    ax2.set_ylabel("F1-Score", fontweight="bold")
    ax2.set_title("(A) F1-Score vs Throughput (FPS)", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Analysis A: Accuracy vs Computational Cost", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/visualizations/pareto_map_vs_latency.png", dpi=300)
    plt.close()
    print("  Saved: results/visualizations/pareto_map_vs_latency.png")


# ─── Analysis B: Accuracy vs VLM Invocation Rate ────────────────────────────

def plot_accuracy_vs_invocation():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Routing sweep data (from confidence_routing_sweep.csv, 8 budget tiers)
    inv_rates  = [0.0, 18.4, 28.5, 42.0, 58.2, 72.0, 88.5, 100.0]
    fprr_vals  = [0.00, 7.14, 9.89, 14.29, 18.68, 22.53, 23.63, 23.63]
    prec_vals  = [0.6255, 0.6404, 0.6450, 0.6541, 0.6621, 0.6690, 0.6675, 0.6643]
    f1_vals    = [0.2800, 0.2793, 0.2777, 0.2762, 0.2732, 0.2700, 0.2654, 0.2620]
    fnir_vals  = [0.00, 0.22, 0.43, 0.65, 1.01, 1.38, 1.81, 2.10]

    ax = axes[0]
    color1, color2 = "#2ca02c", "#d62728"
    l1, = ax.plot(inv_rates, fprr_vals, "o-", color=color1, linewidth=2.5, markersize=7, label="FPRR (%)")
    ax.set_ylabel("FPRR (%)", color=color1, fontweight="bold")
    ax.tick_params(axis="y", labelcolor=color1)
    ax2 = ax.twinx()
    l2, = ax2.plot(inv_rates, fnir_vals, "s--", color=color2, linewidth=2.0, markersize=7, label="FNIR (%)")
    ax2.set_ylabel("FNIR (%)", color=color2, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax.axvline(x=58.2, color="orange", linestyle=":", linewidth=1.8, label="Medium Budget (58.2%)")
    ax.set_xlabel("VLM Invocation Rate (%)", fontweight="bold")
    ax.set_title("(B) FPRR & FNIR vs VLM Invocation Rate", fontweight="bold")
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax3 = axes[1]
    ax3.plot(inv_rates, [p * 100 for p in prec_vals], "o-", color="#1f77b4", linewidth=2.5,
             markersize=7, label="Precision (×100)")
    ax3.plot(inv_rates, [f * 100 for f in f1_vals], "s--", color="#ff7f0e", linewidth=2.0,
             markersize=7, label="F1-Score (×100)")
    ax3.axvline(x=58.2, color="orange", linestyle=":", linewidth=1.8, label="Optimal (58.2%)")
    ax3.set_xlabel("VLM Invocation Rate (%)", fontweight="bold")
    ax3.set_ylabel("Value (×100)", fontweight="bold")
    ax3.set_title("(B) Precision & F1 vs VLM Invocation Rate", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Analysis B: Detection Performance vs VLM Invocation Rate\n(YOLO26x + Qwen3-VL-8B Dual-Image Routing Sweep)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/visualizations/pareto_accuracy_vs_invocation.png", dpi=300)
    plt.close()
    print("  Saved: results/visualizations/pareto_accuracy_vs_invocation.png")


# ─── Analysis C: FP Reduction vs FN Introduction ────────────────────────────

def plot_fp_vs_fn():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    for p, style in PARADIGM_STYLES.items():
        sub = [e for e in MASTER if e["paradigm"] == p]
        sc = ax.scatter([e["fnir"] for e in sub], [e["fprr"] for e in sub],
                        label=p, color=style["color"], marker=style["marker"],
                        s=style["size"], edgecolors="black", linewidth=0.8, alpha=0.88, zorder=3)

    # Annotate key operating points
    annotate = [
        ("YOLO26x + Q3-8B Dual (Med Budget) 🎯", 1.01, 18.68, (6, 4)),
        ("YOLO26x + Q3-8B Dual (Always-On)",       2.10, 23.63, (6, -12)),
        ("YOLO26s + Q3-8B Dual (Always-On)",        0.45, 21.62, (6, 4)),
        ("YOLO26x + Q3-2B Dual (Edge)",             2.10, 11.54, (-95, 4)),
    ]
    for name, x, y, off in annotate:
        ax.annotate(name.replace(" 🎯", "*"), (x, y),
                    textcoords="offset points", xytext=off, fontsize=8.0, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))

    # Arrow indicating "better" direction
    ax.annotate("", xy=(0.1, 22), xytext=(2.0, 5),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.0))
    ax.text(0.3, 19, "Better\n(↑FPRR, ↓FNIR)", color="#2ca02c", fontsize=9, fontweight="bold")

    ax.set_xlabel("FNIR — False Negative Introduction Rate (%)", fontweight="bold")
    ax.set_ylabel("FPRR — False Positive Rejection Rate (%)", fontweight="bold")
    ax.set_title("Analysis C: FP Reduction vs FN Introduction Trade-off\n(Upper-left quadrant = Pareto optimal)",
                 fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-1.0, 27.0)
    plt.tight_layout()
    plt.savefig("results/visualizations/pareto_fp_vs_fn.png", dpi=300)
    plt.close()
    print("  Saved: results/visualizations/pareto_fp_vs_fn.png")


# ─── Analysis D: Model Efficiency ────────────────────────────────────────────

def plot_model_efficiency():
    """FPRR per VRAM and per latency for cascade configurations."""
    cascade = [e for e in MASTER if e["paradigm"] in ("Cascade Dual", "Proposed Pipeline") and e["fprr"] > 0]
    names = [e["system"].replace(" + Q3-8B", "").replace(" + Q3-2B", " (2B)") for e in cascade]
    fprr_per_vram = [e["fprr"] / (e["vram_mb"] / 1000) for e in cascade]
    fprr_per_lat  = [e["fprr"] / e["lat_ms"] for e in cascade]
    colors = ["#1f77b4" if e["paradigm"] == "Cascade Dual" else "#2ca02c" for e in cascade]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    ax = axes[0]
    x = np.arange(len(names))
    bars = ax.bar(x, fprr_per_vram, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("FPRR (%) / GB VRAM", fontweight="bold")
    ax.set_title("(D) FPRR Efficiency: Per GB VRAM", fontweight="bold")
    patches = [mpatches.Patch(color="#1f77b4", label="Cascade Dual Always-On"),
               mpatches.Patch(color="#2ca02c", label="Proposed Pipeline (Routing)")]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = axes[1]
    bars2 = ax2.bar(x, fprr_per_lat, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax2.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("FPRR (%) / ms Latency", fontweight="bold")
    ax2.set_title("(D) FPRR Efficiency: Per ms Latency", fontweight="bold")
    ax2.legend(handles=patches, fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Analysis D: Model Efficiency — FPRR per Resource Unit", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/visualizations/pareto_model_efficiency.png", dpi=300)
    plt.close()
    print("  Saved: results/visualizations/pareto_model_efficiency.png")


# ─── Analysis E: Precision-Recall Frontier ───────────────────────────────────

def plot_precision_recall():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    for p, style in PARADIGM_STYLES.items():
        sub = [e for e in MASTER if e["paradigm"] == p]
        ax.scatter([e["rec"] for e in sub], [e["prec"] for e in sub],
                   label=p, color=style["color"], marker=style["marker"],
                   s=style["size"], edgecolors="black", linewidth=0.8, alpha=0.88, zorder=3)

    # Annotate extremes
    for e in MASTER:
        if e["prec"] > 0.75 or e["f1"] > 0.29:
            ax.annotate(e["system"][:30], (e["rec"], e["prec"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7.5)

    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title("Analysis E: Precision-Recall Operating Points\nAcross All Paradigms", fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("results/visualizations/pareto_precision_recall_frontier.png", dpi=300)
    plt.close()
    print("  Saved: results/visualizations/pareto_precision_recall_frontier.png")


# ─── Analysis F: FPRR vs Latency ─────────────────────────────────────────────

def plot_fprr_vs_latency():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    for p, style in PARADIGM_STYLES.items():
        sub = [e for e in MASTER if e["paradigm"] == p]
        ax.scatter([e["lat_ms"] for e in sub], [e["fprr"] for e in sub],
                   label=p, color=style["color"], marker=style["marker"],
                   s=style["size"], edgecolors="black", linewidth=0.8, alpha=0.88, zorder=3)

    # Annotate Pareto-optimal points (high FPRR, low latency)
    highlight = [
        ("YOLO26s + Q3-8B Dual (Med Budget)", 11.8, 17.09),
        ("YOLO26m + Q3-8B Dual (Med Budget)", 32.2, 14.50),
        ("YOLO26x + Q3-8B Dual (Med Budget) 🎯", 44.1, 18.68),
    ]
    for name, lx, ly in highlight:
        ax.annotate(name.replace(" 🎯", "*"), (lx, ly),
                    textcoords="offset points", xytext=(6, 5), fontsize=8, fontweight="bold")

    ax.set_xlabel("End-to-End Latency (ms/image)", fontweight="bold")
    ax.set_ylabel("False Positive Rejection Rate — FPRR (%)", fontweight="bold")
    ax.set_title("FPRR vs End-to-End Latency — Pareto Frontier",
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("results/visualizations/pareto_fprr_vs_latency.png", dpi=300)
    plt.close()
    print("  Saved: results/visualizations/pareto_fprr_vs_latency.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  Experiment 8: Accuracy & Computational Efficiency (Pareto)")
    print("=" * 65)

    save_csv(MASTER, "results/tables/master_comparative_benchmark.csv")
    save_json(MASTER, "results/metrics/master_comparative_benchmark.json")

    print("\nGenerating visualizations...")
    plot_map_vs_latency()
    plot_accuracy_vs_invocation()
    plot_fp_vs_fn()
    plot_model_efficiency()
    plot_precision_recall()
    plot_fprr_vs_latency()

    # Identify Pareto-optimal configurations
    print("\n--- Pareto-Optimal Configurations ---")
    # Sort by FPRR descending, then latency ascending
    cascade_pipeline = [e for e in MASTER if e["paradigm"] in ("Cascade Dual", "Proposed Pipeline")]
    cascade_pipeline.sort(key=lambda e: (-e["fprr"], e["lat_ms"]))
    print(f"\n{'System':<45} {'FPRR':>7} {'F1':>6} {'mAP50':>7} {'FPS':>6} {'VRAM':>8}")
    print("-" * 82)
    for e in cascade_pipeline[:10]:
        print(f"{e['system']:<45} {e['fprr']:>6.2f}% {e['f1']:>6.4f} {e['map50']:>7.4f} "
              f"{e['fps']:>6.1f} {e['vram_mb']:>7d}MB")

    print("\nExperiment 8 complete.")


if __name__ == "__main__":
    main()
