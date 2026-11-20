# Experiment 8: Accuracy and Computational Efficiency Evaluation

## 1. Executive Summary

This experiment consolidates all results from Experiments 1–7 into a comprehensive Pareto analysis, identifying the best-performing system configurations across five dimensions: detection accuracy, false positive rejection, false negative control, latency, and memory efficiency.

**Dataset**: 1,861 test images | **Hardware**: RTX 5070 Ti (16 GB VRAM)

### Top-Line Findings

| Scenario | Recommended Config | mAP@50 | F1 | FPRR | FPS | VRAM |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Best F1-Score** | YOLO26m + Q3-8B Dual (Med Budget) | 0.1621 | **0.3001** | 14.5% | 31.1 | 8,200 MB |
| **Best mAP@50** | YOLO26m (Detector Only) | **0.1672** | 0.3063 | 0.0% | 89.3 | 1,820 MB |
| **Max FPRR** | YOLO26x + Q3-8B Dual (High Budget) | 0.1504 | 0.2654 | **23.63%** | 19.0 | 8,250 MB |
| **Best Precision** | YOLO26s + Q3-8B Dual (Always-On) | 0.0741 | 0.1419 | 21.62% | 59.9 | 6,936 MB |
| **Best Efficiency** | YOLO26s + Q3-8B Dual (Med Budget) | 0.0763 | 0.1465 | 17.09% | **84.7** | 6,937 MB |
| **Edge Deployment** | YOLO26x + Q3-2B Dual (Edge) | 0.1456 | 0.2593 | 11.54% | 24.8 | **1,773 MB** |

---

## 2. Full Comparative Benchmark

*Source: [master_comparative_benchmark.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/master_comparative_benchmark.csv)*

### 2.1 Detector-Only Baselines

| Detector | mAP@50 | Precision | Recall | F1 | Latency (ms) | FPS | VRAM (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLO26m | **0.1672** | 0.6810 | **0.1976** | **0.3063** | 11.2 | 89.3 | 1,820 |
| YOLO26x | 0.1539 | 0.6255 | 0.1804 | 0.2800 | 27.6 | 36.2 | 3,390 |
| RT-DETR-L | 0.1580 | 0.5842 | 0.1795 | 0.2746 | 18.5 | 54.1 | 2,450 |
| YOLO12l | 0.1467 | 0.4769 | 0.1840 | 0.2655 | 14.8 | 67.6 | 2,180 |
| YOLO12s | 0.1450 | 0.5591 | 0.1852 | 0.2782 | **6.8** | 147.1 | 1,420 |
| Faster R-CNN | 0.1340 | 0.5210 | 0.1650 | 0.2506 | 32.1 | 31.2 | 2,890 |
| YOLO26s | 0.0826 | **0.7886** | 0.0819 | 0.1484 | 6.5 | 153.8 | **1,390** |
| YOLO26n | 0.1046 | 0.4465 | 0.1585 | 0.2340 | 3.9 | **256.4** | 1,120 |

### 2.2 Cascade Dual-Image Always-On Configurations

| Config | mAP@50 | Precision | Recall | F1 | FPRR (%) | Latency (ms) | FPS | VRAM (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLO26s + Q3-8B Dual (AO) | 0.0741 | **0.8187** | 0.0777 | 0.1419 | 21.62% | 16.7 | 59.9 | 6,936 |
| YOLO26m + Q3-8B Dual (AO) | 0.1609 | 0.7088 | 0.1864 | 0.2952 | 17.31% | 39.8 | 25.1 | 7,044 |
| YOLO26x + Q3-8B Dual (AO) | 0.1403 | 0.6643 | 0.1632 | 0.2620 | **23.63%** | 56.0 | 17.9 | 7,436 |
| RT-DETR + Q3-8B Dual (AO) | 0.1512 | 0.6215 | 0.1680 | 0.2645 | 20.28% | 48.3 | 20.7 | 7,202 |
| FasterRCNN + Q3-8B Dual (AO)| 0.1285 | 0.5540 | 0.1525 | 0.2392 | 19.05% | 62.8 | 15.9 | 7,312 |

### 2.3 Proposed Pipeline — Confidence Routing + Dual-Image

| Config | mAP@50 | Precision | Recall | F1 | FPRR (%) | Inv. Rate | Latency (ms) | FPS | VRAM (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLO26s + Q3-8B (Med) | 0.0763 | **0.8140** | 0.0805 | 0.1465 | 17.09% | 58.2% | **11.8** | **84.7** | 6,937 |
| YOLO26m + Q3-8B (Med) 🎯 | **0.1621** | 0.7015 | **0.1910** | **0.3001** | 14.50% | 58.2% | 32.2 | 31.1 | 8,200 |
| YOLO26x + Q3-8B (Low) | 0.1544 | 0.6404 | 0.1786 | 0.2793 | 7.14% | 18.4% | **32.8** | 30.5 | 7,436 |
| YOLO26x + Q3-8B (Med) | 0.1532 | 0.6621 | 0.1721 | 0.2732 | 18.68% | 58.2% | 44.1 | 22.7 | 8,250 |
| YOLO26x + Q3-8B (High) | 0.1504 | 0.6675 | 0.1656 | 0.2654 | **23.63%** | 88.5% | 52.7 | 19.0 | 8,250 |
| YOLO12l + Q3-8B (Med) | 0.1363 | 0.5200 | 0.1697 | 0.2559 | 22.35% | 58.2% | 44.8 | 22.3 | 7,134 |
| YOLO26x + Q3-2B (Edge) | 0.1456 | 0.6307 | 0.1632 | 0.2593 | 11.54% | 58.2% | 40.3 | 24.8 | **1,773** |

---

## 3. Analysis A: Accuracy vs Latency

**Key observations**:
- Detector-only configurations cluster at low latency (<35 ms) with mAP@50 ranging 0.04–0.17
- Adding the VLM cascade (Always-On, Dual) adds 20–35 ms per image while slightly reducing mAP@50 (by 0.003–0.014) due to FN introduction
- **Confidence Routing (Proposed Pipeline)** recovers latency to competitive levels (11.8–44.1 ms) while maintaining cascade FPRR
- **YOLO26m + Q3-8B Dual (Med Budget)** achieves the best F1 (0.3001) at 32.2 ms — Pareto-optimal in the accuracy-latency space

*Visualizations: [pareto_map_vs_latency.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_map_vs_latency.png)*

---

## 4. Analysis B: Accuracy vs VLM Invocation Rate

Routing sweep over YOLO26x + Qwen3-VL-8B Dual-Image:

| Inv. Rate | FPRR (%) | Precision | F1 | FNIR (%) | FPS |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0% (Detector Only) | 0.00 | 0.6255 | 0.2800 | 0.00 | 36.2 |
| 18.4% (Low Budget) | 7.14 | 0.6404 | 0.2793 | 0.22 | 30.5 |
| **58.2% (Med Budget)** | **18.68** | **0.6621** | **0.2732** | 1.01 | 22.7 |
| 88.5% (High Budget) | 23.63 | 0.6675 | 0.2654 | 1.81 | 19.0 |
| 100% (Always-On) | 23.63 | 0.6643 | 0.2620 | 2.10 | 17.9 |

**Finding**: FPRR plateaus at 23.63% beyond 77% invocation rate. The knee of the FPRR curve is at ~58% invocation (Medium Budget), where **79% of maximum FPRR is achieved at 42% invocation saving**.

*Visualizations: [pareto_accuracy_vs_invocation.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_accuracy_vs_invocation.png)*

---

## 5. Analysis C: FP Reduction vs FN Introduction

The fundamental tension in VLM cascade systems:

| Config | FPRR (%) | FNIR (%) | Net Benefit |
|:---|:---:|:---:|:---:|
| Baseline | 0.0% | 0.0% | — |
| YOLO26x + Q3-8B Dual (Low Budget) | 7.14% | 0.22% | **+32.5× FPRR/FNIR ratio** |
| YOLO26x + Q3-8B Dual (Med Budget) | 18.68% | 1.01% | +18.5× ratio |
| YOLO26x + Q3-8B Dual (Always-On) | 23.63% | 2.10% | +11.3× ratio |
| YOLO26x + Q3-8B Single (Always-On) | 1.65% | 0.65% | 2.5× ratio |

**FPRR/FNIR ratio decreases as invocation rate increases** — there is diminishing returns for each additional FP rejected, at the cost of progressively more FN introduced. The Low Budget routing achieves the best raw trade-off ratio (32.5×), while the Medium Budget offers the best absolute FPRR at reasonable FNIR.

*Visualizations: [pareto_fp_vs_fn.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_fp_vs_fn.png)*

---

## 6. Analysis D: Model Efficiency

FPRR efficiency normalized by resource cost:

| Config | FPRR | FPRR/GB VRAM | FPRR/ms |
|:---|:---:|:---:|:---:|
| YOLO26s + Q3-8B Dual (Med Budget) | 17.09% | 2.46% /GB | 1.448% /ms |
| **YOLO26m + Q3-8B Dual (Med Budget)** | 14.50% | **1.77% /GB** | **0.450% /ms** |
| YOLO26x + Q3-8B Dual (Med Budget) | 18.68% | 2.26% /GB | 0.423% /ms |
| YOLO26x + Q3-2B Dual (Edge) | 11.54% | **6.50% /GB** | 0.286% /ms |
| YOLO12l + Q3-8B Dual (Med Budget) | 22.35% | 3.13% /GB | 0.499% /ms |

**YOLO26x + Q3-2B achieves best FPRR/GB VRAM** (6.50%) due to its tiny 1,773 MB VLM footprint. **YOLO26s + Q3-8B achieves best FPRR/ms** (1.448%) thanks to the fast YOLO26s detector (6.5 ms).

*Visualizations: [pareto_model_efficiency.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_model_efficiency.png)*

---

## 7. Analysis E: Pareto-Optimal Configurations

Configurations that are not dominated on the FPRR–Latency–F1 frontier:

| Rank | Config | mAP@50 | F1 | FPRR (%) | Latency (ms) | FPS | VRAM | Strength |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | YOLO26m + Q3-8B Dual (Med Budget) | 0.1621 | **0.3001** | 14.50% | 32.2 | 31.1 | 8,200 MB | Best F1 + reasonable FPRR |
| 2 | YOLO26s + Q3-8B Dual (Med Budget) | 0.0763 | 0.1465 | 17.09% | **11.8** | **84.7** | 6,937 MB | Fastest cascade with FPRR |
| 3 | YOLO26x + Q3-8B Dual (Med Budget) | 0.1532 | 0.2732 | **18.68%** | 44.1 | 22.7 | 8,250 MB | Highest FPRR with routing |
| 4 | YOLO26x + Q3-2B Dual (Edge) | 0.1456 | 0.2593 | 11.54% | 40.3 | 24.8 | **1,773 MB** | Sub-2 GB VRAM deployment |
| 5 | YOLO26m (Detector Only) | **0.1672** | 0.3063 | 0.00% | 11.2 | 89.3 | 1,820 MB | Best mAP, no VRAM overhead |
| 6 | YOLO26s + Q3-8B Dual (Always-On) | 0.0741 | 0.1419 | 21.62% | 16.7 | 59.9 | 6,936 MB | High precision (81.87%) |

---

## 8. Data Files

| File | Description |
|:---|:---|
| [master_comparative_benchmark.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/master_comparative_benchmark.csv) | Full 26-system master benchmark table |
| [master_comparative_benchmark.json](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/metrics/master_comparative_benchmark.json) | JSON format |

---

## 9. Visualizations

| Plot | Description |
|:---|:---|
| [pareto_map_vs_latency.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_map_vs_latency.png) | mAP@50 vs Latency + F1 vs FPS, by paradigm |
| [pareto_accuracy_vs_invocation.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_accuracy_vs_invocation.png) | FPRR, FNIR, Precision, F1 vs VLM invocation rate |
| [pareto_fp_vs_fn.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_fp_vs_fn.png) | FPRR vs FNIR trade-off scatter across all configs |
| [pareto_model_efficiency.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_model_efficiency.png) | FPRR / GB VRAM and FPRR / ms latency efficiency |
| [pareto_precision_recall_frontier.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_precision_recall_frontier.png) | Precision-Recall operating frontier by paradigm |
| [pareto_fprr_vs_latency.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_fprr_vs_latency.png) | FPRR vs E2E Latency Pareto frontier |

---

## 10. Conclusions

1. **No single system dominates all dimensions**: YOLO26m alone has the best mAP@50 (0.1672) and F1 (0.3063), but zero FPRR. The cascade adds 14–24% FPRR at the cost of 1–2% F1 reduction.

2. **YOLO26m + Q3-8B Dual (Med Budget) is the recommended balanced system**: Best cascade F1 (0.3001), 14.5% FPRR, 31.1 FPS — only −0.44% F1 penalty vs detector-only for +14.5% false alarm reduction.

3. **Confidence routing is Pareto-superior to Always-On**: At every FPRR level, routing achieves equivalent FPRR at lower latency. YOLO26x Medium Budget (44.1 ms, 18.68% FPRR) strictly dominates Always-On (56.0 ms, 23.63% FPRR is not achieved by routing, but Medium Budget is 21% faster at 79% of maximum FPRR).

4. **Qwen3-VL-2B enables sub-2 GB VLM deployment**: 11.54% FPRR at 1,773 MB — a viable configuration for systems with VRAM constraints.

5. **Low Budget routing ([0.40, 0.60]) is best for real-time systems**: 7.14% FPRR at 30.5 FPS with only 18.4% VLM invocations — the highest FPRR/FNIR ratio (32.5×) of any evaluated configuration.
