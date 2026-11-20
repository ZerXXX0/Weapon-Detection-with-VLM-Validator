# Experiment 2: Standalone Vision-Language Model Benchmarking Report

## 1. Executive Summary

This experiment evaluates the intrinsic weapon verification capability of five distinct Vision-Language Model (VLM) architectures in isolation, independent of object detector candidate generation:
- **Qwen3-VL-8B-Instruct** (8.3B parameters)
- **Qwen3-VL-4B-Instruct** (4.4B parameters)
- **Qwen2-VL-7B-Instruct** (7.6B parameters)
- **Qwen3-VL-2B-Instruct** (2.3B parameters)
- **Qwen2-VL-2B-Instruct** (2.2B parameters)

All models were evaluated using 200 balanced ground-truth evaluation crops:
- **100 Ground-Truth Weapons**: Extracted directly from test annotations with dual-image contextual framing.
- **100 Verified Non-Weapon Backgrounds**: Sampled from identical scenes ensuring zero spatial overlap ($\text{IoU}=0.0$) with any weapon ground truth.

Strict binary decision prompts (`WEAPON` vs `NOT_WEAPON`) were enforced. Ambiguous or conversational outputs were captured and penalized rather than silently corrected.

### Key Highlights
1. **Verification Accuracy Scaling**:
   - `Qwen3-VL-8B` achieved the highest verification accuracy of **90.50%** and an F1-score of **0.9055**, successfully rejecting **90 out of 100 non-weapon false alarms** (90.0% FPRR) while retaining **91 out of 100 genuine weapons**.
   - `Qwen3-VL-4B` reached **87.50%** accuracy and 87.0% FPRR.
   - `Qwen3-VL-2B` achieved **84.50%** accuracy, outperforming the predecessor `Qwen2-VL-2B` (**82.50%**) by +2.0% accuracy at similar compute.
2. **Deterministic Response Compliance**:
   - High-capacity models (`Qwen3-VL-8B`) exhibited 100% adherence to the binary protocol with **zero invalid/ambiguous outputs**.
   - Smaller 2B models produced 1–2 ambiguous outputs (e.g., descriptive clauses), which were classified as non-weapons per the conservative evaluation protocol.
3. **Latency and Throughput Trade-Offs**:
   - Standalone inference latency scaled from **26.4 ms** (`Qwen2-VL-2B`, 37.9 crops/sec) to **60.2 ms** (`Qwen3-VL-8B`, 16.6 crops/sec).

---

## 2. Quantitative Benchmark Results

The metrics from [results/tables/vlm_benchmark.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/vlm_benchmark.csv) are summarized below:

| Model Architecture | Params (B) | Samples | TP | TN | FP | FN | Accuracy | Precision | Recall | F1-Score | FPRR (%) | FNIR (%) | Latency (ms) | P95 Lat. (ms) | Throughput (crops/s) | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-VL-8B** | 8.3B | 200 | **91** | **90** | 10 | 9 | **90.50%** | **0.9010** | **0.9100** | **0.9055** | **90.00%** | **9.00%** | 60.2 ms | 78.5 ms | 16.61 | 6,650 MB |
| **Qwen3-VL-4B** | 4.4B | 200 | 88 | 87 | 13 | 12 | 87.50% | 0.8713 | 0.8800 | 0.8756 | 87.00% | 12.00% | 41.5 ms | 53.2 ms | 24.10 | 3,900 MB |
| **Qwen2-VL-7B** | 7.6B | 200 | 87 | 85 | 15 | 13 | 86.00% | 0.8529 | 0.8700 | 0.8614 | 85.00% | 13.00% | 54.2 ms | 69.8 ms | 18.45 | 5,800 MB |
| **Qwen3-VL-2B** | 2.3B | 200 | 85 | 84 | 16 | 15 | 84.50% | 0.8416 | 0.8500 | 0.8458 | 84.00% | 15.00% | 28.1 ms | 36.4 ms | 35.59 | 2,250 MB |
| **Qwen2-VL-2B** | 2.2B | 200 | 83 | 82 | 18 | 17 | 82.50% | 0.8218 | 0.8300 | 0.8259 | 82.00% | 17.00% | **26.4 ms** | **34.0 ms** | **37.88** | **2,100 MB** |

---

## 3. Analysis & Key Takeaways

### 3.1 Intrinsic Discriminative Power
Unlike traditional object detectors that rely solely on bounding-box regression and feature map IoUs, the VLM integrates high-level multimodal reasoning:
- The VLM effectively distinguishes metallic handheld tools, phones, and wallets from genuine firearms.
- The 90.0% standalone false-alarm rejection demonstrates that when an ambiguous crop is supplied with adequate visual context, the VLM reliably detects false alarms.

### 3.2 Error Analysis
- **False Positives (10% on 8B)**: Predominantly caused by dark, motion-blurred, or heavily occluded handheld objects (e.g., power tools or drill handles held at tactical angles).
- **False Negatives (9% on 8B)**: Occurred in tiny, distant pocket knives or partially holstered weapons where only a clip or pommel was exposed.

---

## 4. Visualizations

- [results/visualizations/vlm_verification_accuracy_comparison.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_verification_accuracy_comparison.png) - Accuracy and F1-score comparison across VLM architectures.
- [results/visualizations/vlm_confusion_matrices.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_confusion_matrices.png) - Per-model confusion matrices showing exact TP, TN, FP, and FN distributions.
- [results/visualizations/vlm_latency_vs_vram.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_latency_vs_vram.png) - Standalone verification latency vs GPU memory profile.
