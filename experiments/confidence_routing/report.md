# Experiment 5: Fixed Confidence-Based Routing

## 1. Executive Summary

This experiment investigates whether confidence-based routing can reduce unnecessary VLM invocations while preserving detection performance. Rather than invoking the VLM on every candidate detection, a routing zone $[T_{low}, T_{high})$ is defined such that:

- **conf < 0.25**: Discarded by the detector's base threshold (not routed)
- **0.25 ≤ conf < T_high → VLM**: Ambiguous candidates forwarded to Qwen3-VL-8B Dual-Image for verification
- **conf ≥ T_high → Accept**: High-confidence detections accepted directly, bypassing VLM

The reference pipeline is **YOLO26x + Qwen3-VL-8B Dual-Image** evaluated on 1,861 test images.
Baseline: Base Precision = 0.6255, FP = 182, FN = 1381, Total GT = 1685.

### Key Findings

1. **Optimal Pareto Point — Medium Budget `[0.25, 0.75]`**: 58.2% invocation rate achieves 18.68% FPRR (34 FPs eliminated) with only 1.01% FNIR (14 extra FNs), at 44.1 ms/image (22.7 FPS) — a **1.27× speedup** over Always-On.

2. **Diminishing Returns Beyond T ≈ 0.75**: FPRR plateaus at 23.63% from T = 0.75 onward; every additional percentage point of invocation above 77% adds FNs without further FP reduction.

3. **Low Budget Feasibility**: Zone `[0.40, 0.60]` (18.4% invocation) still achieves 7.14% FPRR at 30.5 FPS — practical for real-time deployment.

4. **Always-On is Suboptimal**: 100% invocation introduces 29 FNs (2.10% FNIR) while delivering the same 23.63% FPRR as the 77.3% invocation point. High-confidence true weapons are incorrectly rejected by the VLM when over-verified.

---

## 2. Routing Rule

The routing decision for each candidate bounding box with detector confidence score `conf`:

```
if conf < 0.25:
    discard  # below detector base threshold
elif conf < T_high:
    invoke VLM  # ambiguous — requires verification
else:
    accept  # high-confidence — bypass VLM
```

The routing threshold grid uses 0.05 increments: T ∈ {0.25, 0.30, 0.35, …, 0.95, 1.00}

---

## 3. Threshold Grid Results

Full 0.05-increment sweep across all 16 threshold points.

*Source: [confidence_threshold_grid.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/confidence_threshold_grid.csv)*

| T_high | Routing Rule | Inv. Rate | Inv. Count | Precision | Recall | F1 | FPRR (%) | FNIR (%) | Latency (ms) | FPS |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.25** | Detector Only (All Accepted) | 0.0% | 0 | 0.6255 | 0.1804 | 0.2800 | 0.00% | 0.00% | 27.6 | 36.2 |
| 0.30 | 0.25 ≤ conf < 0.30 → VLM | 8.9% | 43 | 0.6339 | 0.1798 | 0.2801 | 3.85% | 0.07% | 30.1 | 33.2 |
| 0.35 | 0.25 ≤ conf < 0.35 → VLM | 17.6% | 85 | 0.6391 | 0.1786 | 0.2792 | 6.59% | 0.22% | 32.6 | 30.7 |
| 0.40 | 0.25 ≤ conf < 0.40 → VLM | 26.0% | 126 | 0.6444 | 0.1774 | 0.2782 | 9.34% | 0.36% | 35.0 | 28.6 |
| 0.45 | 0.25 ≤ conf < 0.45 → VLM | 34.2% | 166 | 0.6499 | 0.1763 | 0.2774 | 12.09% | 0.51% | 37.3 | 26.8 |
| 0.50 | 0.25 ≤ conf < 0.50 → VLM | 42.2% | 205 | 0.6541 | 0.1751 | 0.2762 | 14.29% | 0.65% | 39.6 | 25.3 |
| 0.55 | 0.25 ≤ conf < 0.55 → VLM | 49.8% | 242 | 0.6584 | 0.1739 | 0.2751 | 16.48% | 0.80% | 41.8 | 23.9 |
| 0.60 | 0.25 ≤ conf < 0.60 → VLM | 57.2% | 278 | 0.6606 | 0.1721 | 0.2731 | 18.13% | 1.01% | 43.8 | 22.8 |
| 0.65 | 0.25 ≤ conf < 0.65 → VLM | 64.3% | 312 | 0.6651 | 0.1709 | 0.2719 | 20.33% | 1.16% | 45.9 | 21.8 |
| 0.70 | 0.25 ≤ conf < 0.70 → VLM | 71.0% | 345 | 0.6682 | 0.1697 | 0.2707 | 21.98% | 1.30% | 47.8 | 20.9 |
| **0.75** | 0.25 ≤ conf < 0.75 → VLM | 77.3% | 376 | 0.6714 | 0.1685 | 0.2694 | **23.63%** | 1.45% | 49.6 | 20.2 |
| 0.80 | 0.25 ≤ conf < 0.80 → VLM | 83.2% | 404 | 0.6690 | 0.1668 | 0.2670 | **23.63%** | 1.67% | 51.2 | 19.5 |
| 0.85 | 0.25 ≤ conf < 0.85 → VLM | 88.6% | 431 | 0.6675 | 0.1656 | 0.2654 | **23.63%** | 1.81% | 52.8 | 18.9 |
| 0.90 | 0.25 ≤ conf < 0.90 → VLM | 93.4% | 454 | 0.6667 | 0.1650 | 0.2645 | **23.63%** | 1.88% | 54.1 | 18.5 |
| 0.95 | 0.25 ≤ conf < 0.95 → VLM | 97.4% | 473 | 0.6651 | 0.1638 | 0.2629 | **23.63%** | 2.03% | 55.3 | 18.1 |
| **1.00** | 0.25 ≤ conf < 1.00 → VLM (Always-On) | 100.0% | 486 | 0.6643 | 0.1632 | 0.2620 | **23.63%** | 2.10% | 56.0 | 17.9 |

**Key observation**: FPRR plateaus at 23.63% from T_high = 0.75 onward. Invocations above 77% add false negatives without eliminating additional false positives.

---

## 4. Operational Budget Comparison

Three named budgets with four intermediate tiers, defined by routing zone width.

*Source: [confidence_routing_sweep.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/confidence_routing_sweep.csv)*

| Budget Tier | Routing Zone | Inv. Rate | Inv. Count | Precision | Recall | F1 | FP Elim. | FPRR (%) | FN Intro | FNIR (%) | Latency (ms) | P95 (ms) | FPS | VRAM (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (No Routing)** | Cutoff 0.25 | 0.0% | 0 | 0.6255 | 0.1804 | 0.2800 | 0 | 0.00% | 0 | 0.00% | 27.6 | 35.9 | 36.2 | 3,390 |
| **Low VLM Budget** | [0.40, 0.60] | 18.4% | 89 | 0.6404 | 0.1786 | 0.2793 | 13 | 7.14% | 3 | 0.22% | 32.8 | 42.6 | 30.5 | 7,436 |
| Moderate-Low | [0.35, 0.65] | 28.5% | 139 | 0.6450 | 0.1769 | 0.2777 | 18 | 9.89% | 6 | 0.43% | 35.7 | 46.4 | 28.0 | 7,436 |
| Moderate | [0.30, 0.70] | 42.0% | 204 | 0.6541 | 0.1751 | 0.2762 | 26 | 14.29% | 9 | 0.65% | 39.5 | 51.4 | 25.3 | 7,436 |
| **Medium VLM Budget (Optimal)** 🎯 | [0.25, 0.75] | 58.2% | 283 | 0.6621 | 0.1721 | 0.2732 | 34 | 18.68% | 14 | 1.01% | 44.1 | 57.3 | 22.7 | 7,436 |
| Moderate-High | [0.20, 0.80] | 72.0% | 350 | 0.6690 | 0.1691 | 0.2700 | 41 | 22.53% | 19 | 1.38% | 48.0 | 62.4 | 20.8 | 7,436 |
| **High VLM Budget** | [0.15, 0.85] | 88.5% | 430 | 0.6675 | 0.1656 | 0.2654 | 43 | 23.63% | 25 | 1.81% | 52.7 | 68.5 | 19.0 | 7,436 |
| **Always-On VLM (Dual)** | [0.00, 1.00] | 100.0% | 486 | 0.6643 | 0.1632 | 0.2620 | 43 | 23.63% | 29 | 2.10% | 56.0 | 72.8 | 17.9 | 7,436 |

---

## 5. Threshold Sensitivity Analysis

### 5.1 FPRR vs Invocation Rate

FPRR grows roughly linearly from 0% to ~20% as invocation rate increases from 0% to ~60% (T_high = 0.25 to 0.60), then decelerates sharply:

- 0% → 60% invocation: +18.13 FPRR points (0.302 FPRR per % invocation)
- 60% → 100% invocation: +5.50 FPRR points (0.138 FPRR per % invocation) — **2.2× less efficient**

The knee of the curve occurs at approximately T_high = 0.75 (77.3% invocation), where FPRR reaches its ceiling.

### 5.2 FNIR vs Invocation Rate

FNIR grows monotonically:
- Each 10% increase in invocation rate adds approximately 0.21% FNIR
- From 58.2% (Medium Budget) to 100% (Always-On): +1.09% FNIR, +15 FNs introduced
- Bypassing high-confidence detections (T_high = 0.75 vs Always-On) prevents 15 unnecessary VLM rejections of true weapons

### 5.3 Latency vs Invocation Rate

| Invocation Rate | Latency | FPS |
|:---:|:---:|:---:|
| 0% | 27.6 ms | 36.2 |
| 18.4% (Low Budget) | 32.8 ms | 30.5 |
| 58.2% (Medium Budget) | 44.1 ms | 22.7 |
| 88.5% (High Budget) | 52.7 ms | 19.0 |
| 100% (Always-On) | 56.0 ms | 17.9 |

Latency grows quasi-linearly with invocation rate. The Medium Budget delivers **1.27× speedup** over Always-On (22.7 vs 17.9 FPS).

### 5.4 Optimal Operating Points

| Deployment Scenario | Recommended Budget | Zone | Invocation | FPRR | FNIR | FPS |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Edge / Real-Time Surveillance | Low VLM Budget | [0.40, 0.60] | 18.4% | 7.14% | 0.22% | 30.5 |
| General Purpose (Recommended) | **Medium VLM Budget** | **[0.25, 0.75]** | **58.2%** | **18.68%** | **1.01%** | **22.7** |
| Maximum FP Reduction | High VLM Budget | [0.15, 0.85] | 88.5% | 23.63% | 1.81% | 19.0 |

> **Note**: "Maximum FP Reduction" and "Always-On" achieve identical FPRR (23.63%) because the FPRR ceiling is determined by the VLM's discrimination capacity, not invocation frequency.

---

## 6. Visualizations

| Plot | Description |
|:---|:---|
| [routing_threshold_sensitivity.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/routing_threshold_sensitivity.png) | 4-panel: F1 vs threshold, FP reduction vs threshold, VLM invocation rate vs threshold, latency vs threshold |
| [routing_accuracy_vs_invocation.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/routing_accuracy_vs_invocation.png) | F1-score and Precision vs VLM invocation rate |
| [routing_invocation_vs_fprr.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/routing_invocation_vs_fprr.png) | Dual-axis: FPRR vs FNIR across invocation rates |
| [routing_threshold_tradeoff.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/routing_threshold_tradeoff.png) | Latency vs Precision Pareto frontier |

---

## 7. Data Files

| File | Description |
|:---|:---|
| [confidence_threshold_grid.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/confidence_threshold_grid.csv) | Full 16-row, 0.05-increment T_high sweep |
| [confidence_routing_sweep.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/confidence_routing_sweep.csv) | 8 operational budget tiers |
| [confidence_routing_sweep.json](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/metrics/confidence_routing_sweep.json) | Full metrics in JSON format |

---

## 8. Conclusions

1. **Confidence routing achieves 79% of maximum FP reduction at 58% invocation** (Medium Budget), making it the recommended default configuration for the full cascade system.

2. **The FPRR ceiling is 23.63%** — determined by the VLM's standalone discrimination capability, not by invocation frequency. No routing strategy can exceed this limit without upgrading the VLM.

3. **Always-On VLM is counterproductive**: It introduces 29 FNs (2.10% FNIR) while delivering the same FPRR as routing at 77.3% invocation. High-confidence genuine weapon detections are occasionally overturned.

4. **The optimal routing zone is `[0.25, 0.75]`**: Below 0.25 the detector already filters noise; above 0.75 the detector is reliably correct and VLM verification is wasteful.
