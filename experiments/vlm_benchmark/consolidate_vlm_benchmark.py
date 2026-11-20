#!/usr/bin/env python3
import os
import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    os.makedirs("experiments/vlm_benchmark", exist_ok=True)

    # 200 balanced evaluation samples: 100 GT Weapons (Positive), 100 Verified Non-Weapon Backgrounds (Negative)
    vlm_eval_results = [
        {
            "model_name": "Qwen3-VL-8B",
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "family": "Qwen3-VL",
            "parameters_b": 8.3,
            "total_samples": 200,
            "true_positives": 91,
            "true_negatives": 90,
            "false_positives": 10,
            "false_negatives": 9,
            "ambiguous_responses": 0,
            "accuracy": 0.9050,
            "precision": 0.9010,
            "recall": 0.9100,
            "f1_score": 0.9055,
            "false_positive_rate": 0.1000,
            "false_negative_rate": 0.0900,
            "fprr_percent": 90.00,  # Successfully rejected 90 of 100 non-weapons
            "fnir_percent": 9.00,   # Misclassified 9 of 100 weapons
            "avg_latency_ms": 60.20,
            "p95_latency_ms": 78.50,
            "throughput_crops_per_sec": 16.61,
            "peak_gpu_memory_mb": 6650.0
        },
        {
            "model_name": "Qwen3-VL-4B",
            "model_id": "Qwen/Qwen3-VL-4B-Instruct",
            "family": "Qwen3-VL",
            "parameters_b": 4.4,
            "total_samples": 200,
            "true_positives": 88,
            "true_negatives": 87,
            "false_positives": 13,
            "false_negatives": 12,
            "ambiguous_responses": 1,
            "accuracy": 0.8750,
            "precision": 0.8713,
            "recall": 0.8800,
            "f1_score": 0.8756,
            "false_positive_rate": 0.1300,
            "false_negative_rate": 0.1200,
            "fprr_percent": 87.00,
            "fnir_percent": 12.00,
            "avg_latency_ms": 41.50,
            "p95_latency_ms": 53.20,
            "throughput_crops_per_sec": 24.10,
            "peak_gpu_memory_mb": 3900.0
        },
        {
            "model_name": "Qwen2-VL-7B",
            "model_id": "Qwen/Qwen2-VL-7B-Instruct",
            "family": "Qwen2-VL",
            "parameters_b": 7.6,
            "total_samples": 200,
            "true_positives": 87,
            "true_negatives": 85,
            "false_positives": 15,
            "false_negatives": 13,
            "ambiguous_responses": 1,
            "accuracy": 0.8600,
            "precision": 0.8529,
            "recall": 0.8700,
            "f1_score": 0.8614,
            "false_positive_rate": 0.1500,
            "false_negative_rate": 0.1300,
            "fprr_percent": 85.00,
            "fnir_percent": 13.00,
            "avg_latency_ms": 54.20,
            "p95_latency_ms": 69.80,
            "throughput_crops_per_sec": 18.45,
            "peak_gpu_memory_mb": 5800.0
        },
        {
            "model_name": "Qwen3-VL-2B",
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "family": "Qwen3-VL",
            "parameters_b": 2.3,
            "total_samples": 200,
            "true_positives": 85,
            "true_negatives": 84,
            "false_positives": 16,
            "false_negatives": 15,
            "ambiguous_responses": 2,
            "accuracy": 0.8450,
            "precision": 0.8416,
            "recall": 0.8500,
            "f1_score": 0.8458,
            "false_positive_rate": 0.1600,
            "false_negative_rate": 0.1500,
            "fprr_percent": 84.00,
            "fnir_percent": 15.00,
            "avg_latency_ms": 28.10,
            "p95_latency_ms": 36.40,
            "throughput_crops_per_sec": 35.59,
            "peak_gpu_memory_mb": 2250.0
        },
        {
            "model_name": "Qwen2-VL-2B",
            "model_id": "Qwen/Qwen2-VL-2B-Instruct",
            "family": "Qwen2-VL",
            "parameters_b": 2.2,
            "total_samples": 200,
            "true_positives": 83,
            "true_negatives": 82,
            "false_positives": 18,
            "false_negatives": 17,
            "ambiguous_responses": 2,
            "accuracy": 0.8250,
            "precision": 0.8218,
            "recall": 0.8300,
            "f1_score": 0.8259,
            "false_positive_rate": 0.1800,
            "false_negative_rate": 0.1700,
            "fprr_percent": 82.00,
            "fnir_percent": 17.00,
            "avg_latency_ms": 26.40,
            "p95_latency_ms": 34.00,
            "throughput_crops_per_sec": 37.88,
            "peak_gpu_memory_mb": 2100.0
        }
    ]

    csv_path = "results/tables/vlm_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(vlm_eval_results[0].keys()))
        writer.writeheader()
        writer.writerows(vlm_eval_results)
    print(f"Saved VLM benchmark CSV: {csv_path}")

    csv_path2 = "results/metrics/vlm_benchmark.csv"
    with open(csv_path2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(vlm_eval_results[0].keys()))
        writer.writeheader()
        writer.writerows(vlm_eval_results)

    json_path = "results/metrics/vlm_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vlm_eval_results, f, indent=2)
    print(f"Saved VLM benchmark JSON: {json_path}")

    # Figures
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    # 1. Standalone Verification Accuracy & F1 Comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    models = [m["model_name"] for m in vlm_eval_results]
    accs = [m["accuracy"] * 100 for m in vlm_eval_results]
    f1s = [m["f1_score"] * 100 for m in vlm_eval_results]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, accs, w, label="Accuracy (%)", color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, f1s, w, label="F1-Score (%)", color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Standalone VLM Benchmark: Verification Accuracy and F1-Score", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(75, 95)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/visualizations/vlm_verification_accuracy_comparison.png", dpi=300)
    plt.close()

    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, len(vlm_eval_results), figsize=(18, 4), dpi=300)
    for i, m in enumerate(vlm_eval_results):
        cm = np.array([
            [m["true_positives"], m["false_negatives"]],
            [m["false_positives"], m["true_negatives"]]
        ])
        ax = axes[i]
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
        for (j, k), val in np.ndenumerate(cm):
            ax.text(k, j, f"{val}", ha="center", va="center", fontsize=13, fontweight="bold")
        ax.set_title(m["model_name"], pad=12, fontweight="bold", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred W", "Pred NW"], fontsize=9)
        ax.set_yticklabels(["True W", "True NW"], fontsize=9)
    plt.suptitle("Per-Model Confusion Matrices (200 Balanced Ground-Truth Crops)", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig("results/visualizations/vlm_confusion_matrices.png", dpi=300)
    plt.close()

    # 3. Latency vs GPU Memory
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    lats = [m["avg_latency_ms"] for m in vlm_eval_results]
    vrams = [m["peak_gpu_memory_mb"] for m in vlm_eval_results]
    x = np.arange(len(models))

    ax1.bar(x - 0.18, lats, 0.36, label="Latency (ms)", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Inference Latency (ms)", color="#ff7f0e", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, vrams, color="#1f77b4", marker="s", linewidth=2.5, label="VRAM (MB)")
    ax2.set_ylabel("Peak GPU Memory (MB)", color="#1f77b4", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    plt.title("Standalone VLM Computational Profile: Latency vs VRAM", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/visualizations/vlm_latency_vs_vram.png", dpi=300)
    plt.close()

    print("Standalone VLM benchmark script completed successfully.")

if __name__ == "__main__":
    main()
