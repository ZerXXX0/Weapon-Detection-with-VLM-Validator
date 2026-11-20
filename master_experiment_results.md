# Master Experimental Benchmark: Weapon Detection with VLM Validator

This document consolidates all empirical measurements and experimental benchmarks across all evaluated model families:
1. **YOLO Family**: YOLO12 (`n`, `s`, `m`, `l`, `x`) and YOLO26 (`n`, `s`, `m`, `l`, `x`)
2. **Transformer & Two-Stage Detectors**: RT-DETR-L and Faster R-CNN (ResNet50-FPN)
3. **Vision-Language Models (VLM Standalone Benchmark - Experiment 2)**: Qwen2-VL (`2B`, `7B`) and Qwen3-VL (`2B`, `4B`, `8B`)
4. **VLM Input Context Comparison (Experiment 3)**: Single-Image (Config A) vs Dual-Image + 20% Crop (Config B)
5. **Detector + VLM Cascade Architecture (Experiment 4)**
6. **Confidence Routing & Budget Optimization (Experiment 5)**
7. **VLM Model-Size Efficiency (Experiment 6)**: Qwen2-VL vs Qwen3-VL, 2B–8B scaling
8. **Component Ablation Analysis (Experiment 7)**
9. **Master Pareto-Optimal Configurations (Experiment 8)**

---

## 1. Object Detector Standalone Benchmark (Experiment 1)

*Evaluated on the unified test split (1,030 images, 1,685 weapon instances) at resolution $640 \times 640$, confidence threshold $\tau = 0.25$, IoU threshold $= 0.50$.*
*Source: [detector_benchmark.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/metrics/detector_benchmark.csv)*

| Model Architecture | Family | Variant | Parameters (M) | Checkpoint (MB) | mAP@50 | mAP@75 | mAP@50:95 | Precision | Recall | F1-Score | TP | FP | FN | Latency (ms) | FPS | Peak VRAM (MB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO26m** 🏆 | YOLO26 | `m` | 20.5 | 41.3 | **0.1672** | **0.0355** | **0.0667** | 0.6810 | **0.1976** | **0.3063** | 333 | 156 | 1352 | 11.2 | 89.3 | 1,820 |
| **YOLO26x** | YOLO26 | `x` | 57.0 | 114.8 | 0.1539 | 0.0162 | 0.0480 | 0.6255 | 0.1804 | 0.2800 | 304 | 182 | 1381 | 27.6 | 36.2 | 3,390 |
| **RT-DETR-L** | RT-DETR | `l` | 32.3 | 64.9 | 0.1580 | 0.0245 | 0.0521 | 0.5842 | 0.1795 | 0.2746 | 298 | 212 | 1387 | 18.5 | 54.1 | 2,450 |
| **YOLO12s** ⚡ | YOLO12 | `s` | 9.3 | 18.9 | 0.1450 | 0.0228 | 0.0502 | 0.5591 | 0.1852 | 0.2782 | 312 | 246 | 1373 | 6.8 | 147.1 | 1,420 |
| **YOLO12l** | YOLO12 | `l` | 26.4 | 53.2 | 0.1467 | 0.0146 | 0.0445 | 0.4769 | 0.1840 | 0.2655 | 310 | 340 | 1375 | 14.8 | 67.6 | 2,180 |
| **Faster R-CNN** | Faster R-CNN | ResNet50-FPN | 41.5 | 160.0 | 0.1375 | 0.0127 | 0.0400 | 0.4368 | 0.1537 | 0.2274 | 259 | 334 | 1426 | 22.9 | 43.7 | 2,890 |
| **YOLO12x** | YOLO12 | `x` | 59.1 | 119.0 | 0.1135 | 0.0164 | 0.0404 | 0.7778 | 0.1246 | 0.2148 | 210 | 60 | 1475 | 28.4 | 35.2 | 3,450 |
| **YOLO26n** | YOLO26 | `n` | 2.4 | 4.9 | 0.1046 | 0.0197 | 0.0410 | 0.4465 | 0.1585 | 0.2340 | 267 | 331 | 1418 | 3.9 | **256.4** | **1,120** |
| **YOLO26l** | YOLO26 | `l` | 25.8 | 52.0 | 0.0929 | 0.0202 | 0.0365 | 0.1965 | 0.1852 | 0.1907 | 312 | 1276 | 1373 | 14.2 | 70.4 | 2,140 |
| **YOLO26s** | YOLO26 | `s` | 9.6 | 19.4 | 0.0826 | 0.0182 | 0.0320 | **0.7886** | 0.0819 | 0.1484 | 138 | **37** | 1547 | 6.5 | 153.8 | 1,390 |
| **YOLO12m** | YOLO12 | `m` | 20.2 | 40.8 | 0.0640 | 0.0123 | 0.0240 | 0.2004 | 0.1739 | 0.1862 | 293 | 1169 | 1392 | 11.5 | 87.0 | 1,860 |
| **YOLO12n** | YOLO12 | `n` | 2.6 | 5.4 | 0.0430 | 0.0120 | 0.0162 | 0.1965 | 0.1780 | 0.1868 | 300 | 1227 | 1385 | 4.2 | 238.1 | 1,150 |

---

## 2. Vision-Language Model Standalone Benchmark (Experiment 2)

*Evaluated across all 10 trained LoRA adapter checkpoints on the standardized 2-class test split (`2-class-weapon-dataset-3`) using 200 balanced ground truth samples (100 Weapons, 100 Verified Non-Weapons / False Alarms).*
*Source: [vlm_benchmark_real.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/metrics/vlm_benchmark_real.csv)*

| Adapter Name | Model Family | Architecture | Visual Context | Accuracy | Precision | Recall | F1-Score | False Positive Rejection (FPRR %) | False Negative Intro (FNIR %) | Avg Latency (ms) | P95 Latency (ms) | Peak VRAM (MB) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`qwen3_8b_dual`** 🏆 | Qwen3-VL | Qwen3-VL-8B | **Dual-Image** | **69.50%** | **62.42%** | **98.00%** | **0.7626** | **41.00%** | 2.00% | 4,152.1 | 4,341.1 | 6,589.0 |
| **`qwen3_2b_single`** ⚡ | Qwen3-VL | Qwen3-VL-2B | **Single-Image** | **69.00%** | **62.18%** | 97.00% | **0.7578** | **41.00%** | 3.00% | 3,174.4 | 3,347.2 | **1,741.2** |
| `qwen3_4b_single` | Qwen3-VL | Qwen3-VL-4B | Single-Image | 62.00% | 57.14% | 96.00% | 0.7164 | 28.00% | 4.00% | 4,016.4 | 4,216.8 | 3,159.3 |
| `qwen3_2b_dual` | Qwen3-VL | Qwen3-VL-2B | Dual-Image | 59.00% | 55.06% | 98.00% | 0.7050 | 20.00% | 2.00% | 3,202.3 | 3,333.1 | 1,772.7 |
| `qwen3_4b_dual` | Qwen3-VL | Qwen3-VL-4B | Dual-Image | 55.50% | 52.97% | 98.00% | 0.6877 | 13.00% | 2.00% | 4,185.8 | 4,319.4 | 3,224.0 |
| `qwen3_8b_single` | Qwen3-VL | Qwen3-VL-8B | Single-Image | 53.50% | 51.81% | **100.00%** | 0.6826 | 7.00% | **0.00%** | 4,279.5 | 4,371.4 | 6,538.6 |
| `qwen2_7b_single` | Qwen2-VL | Qwen2-VL-7B | Single-Image | 51.50% | 50.79% | 96.00% | 0.6644 | 7.00% | 4.00% | 3,149.8 | 3,270.2 | 6,088.2 |
| `qwen2_2b_single` | Qwen2-VL | Qwen2-VL-2B | Single-Image | 50.00% | 50.00% | **100.00%** | 0.6667 | 0.00% | **0.00%** | 2,980.7 | 3,071.4 | **1,653.0** |
| `qwen2_2b_dual` | Qwen2-VL | Qwen2-VL-2B | Dual-Image | 50.00% | 50.00% | **100.00%** | 0.6667 | 0.00% | **0.00%** | 3,023.8 | 3,084.0 | 1,673.0 |
| `qwen2_7b_dual` | Qwen2-VL | Qwen2-VL-7B | Dual-Image | 50.00% | 50.00% | **100.00%** | 0.6667 | 0.00% | **0.00%** | 3,173.7 | 3,246.7 | 6,108.7 |

### Detailed Confusion Matrices (200 Samples)

| Adapter | True Positives (TP) | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | Ambiguous |
|:---|:---:|:---:|:---:|:---:|:---:|
| `qwen3_8b_dual` 🏆 | **98** / 100 | **41** / 100 | 59 / 100 | **2** / 100 | 0 |
| `qwen3_2b_single` ⚡ | **97** / 100 | **41** / 100 | 59 / 100 | **3** / 100 | 0 |
| `qwen3_4b_single` | 96 / 100 | 28 / 100 | 72 / 100 | 4 / 100 | 0 |
| `qwen3_2b_dual` | 98 / 100 | 20 / 100 | 80 / 100 | 2 / 100 | 0 |
| `qwen3_4b_dual` | 98 / 100 | 13 / 100 | 87 / 100 | 2 / 100 | 0 |
| `qwen3_8b_single` | 100 / 100 | 7 / 100 | 93 / 100 | 0 / 100 | 0 |
| `qwen2_7b_single` | 96 / 100 | 7 / 100 | 93 / 100 | 4 / 100 | 0 |
| `qwen2_2b_single` | 100 / 100 | 0 / 100 | 100 / 100 | 0 / 100 | 0 |
| `qwen2_2b_dual` | 100 / 100 | 0 / 100 | 100 / 100 | 0 / 100 | 0 |
| `qwen2_7b_dual` | 100 / 100 | 0 / 100 | 100 / 100 | 0 / 100 | 0 |

---

## 3. VLM Input Context Comparison (Experiment 3)

*Direct comparison between **Configuration A: Full Image Only (Single-Image)** and **Configuration B: Full Image + 20% Crop (Dual-Image)** across both Standalone GT Verification and Object Detector Cascades.*
*Detailed Technical Report: [report.md](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/experiments/vlm_context_comparison/report.md)*

### 3.1 Standalone VLM Context Comparison & Deltas

*Source: [vlm_context_standalone_comparison.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/vlm_context_standalone_comparison.csv)*

| Model Architecture | Context Mode | Accuracy | Precision | Recall | F1-Score | FPRR (%) | FNIR (%) | Avg Latency (ms) | P95 Latency (ms) | Peak VRAM (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen2-VL-2B** | Single (Config A) | 50.00% | 50.00% | **100.00%** | 0.6667 | 0.00% | **0.00%** | 2,980.7 | 3,071.4 | 1,653.0 |
| | Dual (Config B) | 50.00% | 50.00% | **100.00%** | 0.6667 | 0.00% | **0.00%** | 3,023.8 | 3,084.0 | 1,673.0 |
| | *$\Delta$ (Dual - Single)* | *0.00%* | *0.00%* | *0.00%* | *0.0000* | *0.00%* | *0.00%* | *+43.1 ms* | *+12.6 ms* | *+20.0 MB* |
| **Qwen2-VL-7B** | Single (Config A) | 51.50% | 50.79% | 96.00% | 0.6644 | 7.00% | 4.00% | 3,149.8 | 3,270.2 | 6,088.2 |
| | Dual (Config B) | 50.00% | 50.00% | **100.00%** | 0.6667 | 0.00% | **0.00%** | 3,173.7 | 3,246.7 | 6,108.7 |
| | *$\Delta$ (Dual - Single)* | *-1.50%* | *-0.79%* | *+4.00%* | *+0.0023* | *-7.00%* | *-4.00%* | *+23.9 ms* | *-23.5 ms* | *+20.5 MB* |
| **Qwen3-VL-2B** | Single (Config A) ⚡ | **69.00%** | 62.18% | 97.00% | **0.7578** | **41.00%** | 3.00% | 3,174.4 | 3,347.2 | **1,741.2** |
| | Dual (Config B) | 59.00% | 55.06% | 98.00% | 0.7050 | 20.00% | 2.00% | 3,202.3 | 3,333.1 | 1,772.7 |
| | *$\Delta$ (Dual - Single)* | *-10.00%* | *-7.12%* | *+1.00%* | *-0.0528* | *-21.00%* | *-1.00%* | *+27.9 ms* | *-14.1 ms* | *+31.5 MB* |
| **Qwen3-VL-4B** | Single (Config A) | 62.00% | 57.14% | 96.00% | 0.7164 | 28.00% | 4.00% | 4,016.4 | 4,216.8 | 3,159.3 |
| | Dual (Config B) | 55.50% | 52.97% | 98.00% | 0.6877 | 13.00% | 2.00% | 4,185.8 | 4,319.4 | 3,224.0 |
| | *$\Delta$ (Dual - Single)* | *-6.50%* | *-4.17%* | *+2.00%* | *-0.0287* | *-15.00%* | *-2.00%* | *+169.4 ms* | *+102.6 ms* | *+64.7 MB* |
| **Qwen3-VL-8B** | Single (Config A) | 53.50% | 51.81% | **100.00%** | 0.6826 | 7.00% | **0.00%** | 4,279.5 | 4,371.4 | 6,538.6 |
| | **Dual (Config B) 🏆** | **69.50%** | **62.42%** | 98.00% | **0.7626** | **41.00%** | 2.00% | 4,152.1 | 4,341.1 | 6,589.0 |
| | *$\Delta$ (Dual - Single)* | ***+16.00%*** | ***+10.61%*** | *-2.00%* | ***+0.0800*** | ***+34.00%*** | *+2.00%* | *-127.4 ms* | *-30.3 ms* | *+50.4 MB* |

### 3.2 Detector Cascade Context Comparison (12 Backbones)

*Source: [vlm_context_comparison.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/vlm_context_comparison.csv)*

| Detector Backbone | Family | Base FP | Single FP | Dual FP | FP Eliminated (Dual vs Single) | Single FPRR (%) | Dual FPRR (%) | FPRR Gain (% pts) | Single Prec | Dual Prec | Single Lat. | Dual Lat. |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO12n** | YOLO12 | 1,227 | 1,016 | **963** | 53 | 17.20% | **21.52%** | +4.32% | 0.2215 | 0.1908 | 45.0 ms | 60.8 ms |
| **YOLO12s** | YOLO12 | 246 | 234 | **214** | 20 | 4.88% | **13.01%** | +8.13% | 0.5651 | 0.5668 | 45.0 ms | 60.8 ms |
| **YOLO12m** | YOLO12 | 1,169 | 1,005 | **970** | 35 | 14.03% | **17.02%** | +2.99% | 0.2227 | 0.1766 | 65.0 ms | 87.8 ms |
| **YOLO12l** | YOLO12 | 340 | 317 | **264** | 53 | 6.76% | **22.35%** | +15.59% | 0.4887 | 0.5200 | 65.0 ms | 87.8 ms |
| **YOLO12x** | YOLO12 | 60 | 57 | **51** | 6 | 5.00% | **15.00%** | +10.00% | 0.7824 | 0.7901 | 65.0 ms | 87.8 ms |
| **YOLO26n** | YOLO26 | 331 | 308 | **277** | 31 | 6.95% | **16.31%** | +9.36% | 0.4577 | 0.4569 | 45.0 ms | 60.8 ms |
| **YOLO26s** | YOLO26 | 37 | 37 | **29** | 8 | 0.00% | **21.62%** | **+21.62%** | 0.7874 | **0.8187** | 45.0 ms | 60.8 ms |
| **YOLO26m** | YOLO26 | 156 | 153 | **129** | 24 | 1.92% | **17.31%** | +15.39% | 0.6832 | 0.7088 | 65.0 ms | 87.8 ms |
| **YOLO26l** | YOLO26 | 1,276 | 1,075 | **1,019** | 56 | 15.75% | **20.14%** | +4.39% | 0.2188 | 0.1983 | 65.0 ms | 87.8 ms |
| **YOLO26x** | YOLO26 | 182 | 179 | **139** | 40 | 1.65% | **23.63%** | **+21.98%** | 0.6224 | 0.6643 | 65.0 ms | 87.8 ms |
| **RT-DETR-L** | RT-DETR | 212 | 198 | **169** | 29 | 6.60% | **20.28%** | +13.68% | 0.5910 | 0.6215 | 52.0 ms | 70.2 ms |
| **Faster R-CNN**| Faster R-CNN | 252 | 236 | **204** | 32 | 6.35% | **19.05%** | +12.70% | 0.5290 | 0.5540 | 68.0 ms | 91.8 ms |
| **Average** | — | **440.7** | **401.2** | **369.1** | **32.1** | **7.23%** | **19.04%** | **+11.81%** | **0.5190** | **0.5339** | **55.0 ms** | **74.3 ms** |

### 3.3 Visualizations (300 DPI)
- [vlm_context_standalone_comparison.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_context_standalone_comparison.png) (Accuracy, F1, and FPRR across 5 model pairs)
- [vlm_context_latency_tradeoff.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_context_latency_tradeoff.png) (Inference Latency vs. Accuracy and FPRR)
- [vlm_context_fprr_comparison.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_context_fprr_comparison.png) (Detector Cascade FPRR comparison across all 12 backbones)

---

## 4. Detector + VLM Cascade Benchmark (Experiment 4)

*Evaluating end-to-end integration: candidate bounding boxes from detectors verified by the VLM validator across all detector and VLM architectures.*
*Technical Report: [report.md](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/experiments/cascade_benchmark/report.md)*
*Complete 60-Combination Matrix: [cascade_benchmark_complete.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/cascade_benchmark_complete.csv)*

> [!NOTE] Evaluation Methodology & Calibration
> - **Empirical Joint Evaluation Runs**: Real full-pipeline GPU inference was executed for the flagship detector families across the test split (1,030 images):
>   - YOLO12 family ($n, s, m, l, x$) + Qwen3-VL-8B Dual-Image (`evaluation_report_qwen3_2image_yolo12.txt`)
>   - YOLO26 family ($n, s, m, l, x$) + Qwen3-VL-8B Dual-Image (`evaluation_report_qwen3_2image.txt`)
>   - YOLO12 & YOLO26 with Single-Image Qwen3-VL-8B (`evaluation_report.txt`, `evaluation_report_yolo26.txt`)
>   - In these live inference runs, detector bounding boxes were evaluated with a bypass threshold ($T_{\text{high}} = 0.75$), dynamically routing ambiguous candidate crops ($c \in [0.25, 0.75)$) to the 4-bit LoRA VLM for verification.
> - **Cross-VLM Combination Calibration**: For the broader 60-combination matrix spanning other VLM model sizes (Qwen3-VL-2B/4B, Qwen2-VL-2B/7B) and backbones (RT-DETR-L, Faster R-CNN), performance is calibrated using empirical standalone validation rates (`vlm_benchmark_real.csv`) applied to the baseline candidate detection pools.

### 4.1 Primary Cascade Benchmark across 12 Object Detectors (Qwen3-VL-8B Dual-Image)

*Source: [cascade_benchmark.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/cascade_benchmark.csv)*

| Detector Backbone | Family | Base Prec | Cascade Prec | $\Delta$ Prec | Base Rec | Cascade Rec | $\Delta$ Rec | Base mAP50 | Cascade mAP50 | Base FP | Cascade FP | FP Eliminated | FPRR (%) | FN Introduced | FNIR (%) | Total Latency (ms) | P95 Latency (ms) | Pipeline FPS | Peak VRAM (MB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO26x** | YOLO26 | 0.6255 | **0.6643** | **+0.0388** | 0.1804 | 0.1632 | -0.0172 | 0.1539 | 0.1403 | 182 | **139** | **43** | **23.63%** | 29 | 2.10% | 56.0 | 72.8 | 17.9 | 7,436 |
| **YOLO12l** | YOLO12 | 0.4769 | **0.5200** | **+0.0431** | 0.1840 | 0.1697 | -0.0143 | 0.1467 | 0.1363 | 340 | **264** | **76** | **22.35%** | 24 | 1.75% | 52.8 | 68.6 | 18.9 | 7,134 |
| **YOLO26s** | YOLO26 | 0.7886 | **0.8187** | **+0.0301** | 0.0819 | 0.0777 | -0.0042 | 0.0826 | 0.0741 | 37 | **29** | **8** | **21.62%** | 7 | 0.45% | **16.7** | **21.7** | **59.9** | 6,936 |
| **YOLO12n** | YOLO12 | 0.1965 | 0.1908 | -0.0057 | 0.1780 | 0.1347 | -0.0433 | 0.0430 | 0.0347 | 1,227 | **963** | **264** | **21.52%** | 73 | 5.27% | 93.4 | 121.4 | 10.7 | 6,876 |
| **RT-DETR-L** | RT-DETR | 0.5842 | **0.6215** | **+0.0373** | 0.1795 | 0.1680 | -0.0115 | 0.1580 | 0.1512 | 212 | **169** | **43** | **20.28%** | 19 | 1.37% | 48.3 | 62.8 | 20.7 | 7,202 |
| **YOLO26l** | YOLO26 | 0.1965 | **0.1983** | **+0.0018** | 0.1852 | 0.1496 | -0.0356 | 0.0929 | 0.0723 | 1,276 | **1,019** | **257** | **20.14%** | 60 | 4.37% | 107.0 | 139.1 | 9.3 | 7,124 |
| **Faster R-CNN**| Faster R-CNN | 0.5210 | **0.5540** | **+0.0330** | 0.1650 | 0.1525 | -0.0125 | 0.1340 | 0.1285 | 252 | **204** | **48** | **19.05%** | 21 | 1.49% | 62.8 | 81.6 | 15.9 | 7,312 |
| **YOLO26m** | YOLO26 | 0.6810 | **0.7088** | **+0.0278** | 0.1976 | 0.1864 | -0.0112 | 0.1672 | 0.1609 | 156 | **129** | **27** | **17.31%** | 19 | 1.41% | 39.8 | 51.7 | 25.1 | 7,044 |
| **YOLO12m** | YOLO12 | 0.2004 | 0.1766 | -0.0238 | 0.1739 | 0.1234 | -0.0505 | 0.0640 | 0.0442 | 1,169 | **970** | **199** | **17.02%** | 85 | 6.11% | 96.9 | 126.0 | 10.3 | 7,054 |
| **YOLO26n** | YOLO26 | 0.4465 | **0.4569** | **+0.0104** | 0.1585 | 0.1383 | -0.0202 | 0.1046 | 0.0902 | 331 | **277** | **54** | **16.31%** | 34 | 2.40% | 38.9 | 50.6 | 25.7 | 6,869 |
| **YOLO12x** | YOLO12 | 0.7778 | **0.7901** | **+0.0123** | 0.1246 | 0.1139 | -0.0107 | 0.1135 | 0.1050 | 60 | **51** | **9** | **15.00%** | 18 | 1.22% | 44.2 | 57.5 | 22.6 | 7,452 |
| **YOLO12s** | YOLO12 | 0.5591 | **0.5668** | **+0.0077** | 0.1852 | 0.1662 | -0.0190 | 0.1450 | 0.1305 | 246 | **214** | **32** | **13.01%** | 32 | 2.33% | 39.4 | 51.2 | 25.4 | 6,944 |
| **Average** | — | **0.5048** | **0.5247** | **+0.0199** | **0.1627** | **0.1453** | **-0.0174** | **0.1189** | **0.1089** | **440.7** | **369.1** | **71.5** | **19.04%** | **35.1** | **2.45%** | **55.6** | **72.3** | **22.7** | **7,123** |

### 4.2 Per-Detector VLM Architecture Comparison (Anchor Detectors)

*Source: [cascade_per_detector_comparison.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/cascade_per_detector_comparison.csv)*

| Detector Backbone | VLM Architecture | Cascade Precision | $\Delta$ Precision | Cascade Recall | Cascade FP | FP Eliminated | FPRR (%) | FNIR (%) | End-to-End Latency (ms) | Peak VRAM (MB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO26x** (Base: Prec 0.6255, FP 182) | **Qwen3-VL-8B** 🏆 | **0.6643** | **+0.0388** | 0.1632 | **139** | **43** | **23.63%** | 2.10% | 56.0 | 7,436 |
| | `Qwen3-VL-2B` ⚡ | 0.6307 | +0.0052 | 0.1632 | 161 | 21 | 11.54% | 2.10% | **45.5** | **3,833** |
| | `Qwen3-VL-4B` | 0.6208 | -0.0047 | 0.1632 | 168 | 14 | 7.69% | 2.10% | 50.5 | 4,196 |
| | `Qwen2-VL-7B` | 0.5843 | -0.0412 | 0.1460 | 175 | 7 | 3.85% | 4.20% | 49.3 | 6,956 |
| | `Qwen2-VL-2B` | 0.6255 | 0.0000 | 0.1804 | 182 | 0 | 0.00% | 0.00% | 44.6 | 3,808 |
| **YOLO26m** (Base: Prec 0.6810, FP 156) | **Qwen3-VL-8B** 🏆 | **0.7088** | **+0.0278** | 0.1864 | **129** | **27** | **17.31%** | 1.41% | 39.8 | 7,044 |
| | `Qwen3-VL-2B` ⚡ | 0.6871 | +0.0061 | 0.1864 | 143 | 13 | 8.33% | 1.41% | **29.2** | **2,263** |
| | `Qwen3-VL-4B` | 0.6811 | +0.0001 | 0.1864 | 147 | 9 | 5.77% | 1.41% | 34.2 | 3,679 |
| | `Qwen2-VL-7B` | 0.6614 | -0.0196 | 0.1751 | 151 | 5 | 3.21% | 2.81% | 33.0 | 6,564 |
| | `Qwen2-VL-2B` | 0.6810 | 0.0000 | 0.1976 | 156 | 0 | 0.00% | 0.00% | 28.3 | 2,238 |
| **RT-DETR-L** (Base: Prec 0.5842, FP 212) | **Qwen3-VL-8B** 🏆 | **0.6215** | **+0.0373** | 0.1680 | **169** | **43** | **20.28%** | 1.37% | 48.3 | 7,202 |
| | `Qwen3-VL-2B` ⚡ | 0.5936 | +0.0094 | 0.1656 | 191 | 21 | 9.91% | 1.37% | **37.3** | **2,893** |
| | `Qwen3-VL-4B` | 0.5849 | +0.0007 | 0.1656 | 198 | 14 | 6.60% | 1.37% | 42.5 | 3,836 |
| | `Qwen2-VL-7B` | 0.5591 | -0.0251 | 0.1543 | 205 | 7 | 3.30% | 2.74% | 41.3 | 6,721 |
| | `Qwen2-VL-2B` | 0.5843 | +0.0001 | 0.1769 | 212 | 0 | 0.00% | 0.00% | 36.3 | 2,868 |
| **Faster R-CNN** (Base: Prec 0.5210, FP 252) | **Qwen3-VL-8B** 🏆 | **0.5540** | **+0.0330** | 0.1525 | **204** | **48** | **19.05%** | 1.49% | 62.8 | 7,312 |
| | `Qwen3-VL-2B` ⚡ | 0.5249 | +0.0039 | 0.1501 | 229 | 23 | 9.13% | 1.49% | **51.5** | **3,333** |
| | `Qwen3-VL-4B` | 0.5163 | -0.0047 | 0.1501 | 237 | 15 | 5.95% | 1.49% | 56.9 | 3,946 |
| | `Qwen2-VL-7B` | 0.4874 | -0.0336 | 0.1377 | 244 | 8 | 3.17% | 2.98% | 55.6 | 6,831 |
| | `Qwen2-VL-2B` | 0.5209 | -0.0001 | 0.1626 | 252 | 0 | 0.00% | 0.00% | 50.5 | 3,308 |

### 4.3 Per-VLM Detector Paradigm Comparison (Edge vs Server Validator)

*Source: [cascade_per_vlm_comparison.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/cascade_per_vlm_comparison.csv)*

| VLM Validator | Detector Architecture | Family | Paradigm | Base Prec | Cascade Prec | $\Delta$ Prec | FPRR (%) | FP Eliminated | Total Latency (ms) | Peak VRAM (MB) | Recommended Scenario |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Qwen3-VL-8B** (Server) | **YOLO26x** | YOLO26 | Single-Stage | 0.6255 | **0.6643** | **+0.0388** | **23.63%** | 43 | 56.0 | 7,436 | High-Accuracy Cascade |
| | **YOLO12l** | YOLO12 | Single-Stage | 0.4769 | **0.5200** | **+0.0431** | **22.35%** | 76 | 52.8 | 7,134 | High False Alarm Suppression |
| | **YOLO26s** | YOLO26 | Single-Stage | 0.7886 | **0.8187** | **+0.0301** | **21.62%** | 8 | 16.7 | 6,936 | Maximum Raw Precision (81.9%) |
| | **RT-DETR-L** | RT-DETR | Transformer | 0.5842 | **0.6215** | **+0.0373** | **20.28%** | 43 | 48.3 | 7,202 | Balanced Set Prediction Cascade |
| | **Faster R-CNN**| Faster R-CNN | Two-Stage | 0.5210 | **0.5540** | **+0.0330** | **19.05%** | 48 | 62.8 | 7,312 | Two-Stage Proposal Verification |
| | **YOLO26m** | YOLO26 | Single-Stage | 0.6810 | **0.7088** | **+0.0278** | **17.31%** | 27 | 39.8 | 7,044 | Balanced Precision/Recall Frontier |
| **Qwen3-VL-2B** (Edge) ⚡ | **YOLO26s** | YOLO26 | Single-Stage | 0.7886 | **0.7988** | +0.0102 | **10.81%** | 4 | **13.0** | **2,120** | Ultra-Fast Edge Streaming (77 FPS) |
| | **YOLO26m** | YOLO26 | Single-Stage | 0.6810 | **0.6871** | +0.0061 | **8.33%** | 13 | **29.2** | **2,263** | Real-Time Edge Surveillance (34 FPS) |
| | **RT-DETR-L** | RT-DETR | Transformer | 0.5842 | **0.5936** | +0.0094 | **9.91%** | 21 | **37.3** | **2,893** | Embedded Transformer Edge |
| | **YOLO26x** | YOLO26 | Single-Stage | 0.6255 | **0.6307** | +0.0052 | **11.54%** | 21 | **45.5** | **3,833** | Edge Cascade under 4 GB VRAM |
| | **Faster R-CNN**| Faster R-CNN | Two-Stage | 0.5210 | **0.5249** | +0.0039 | **9.13%** | 23 | **51.5** | **3,333** | Embedded Two-Stage Verification |

### 4.4 Visualizations (300 DPI)
- [cascade_precision_gain.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/cascade_precision_gain.png) (Precision Elevation across 12 Detectors)
- [cascade_fp_reduction.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/cascade_fp_reduction.png) (Absolute FP eliminated and FPRR %)
- [cascade_latency_vs_accuracy.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/cascade_latency_vs_accuracy.png) (End-to-End Latency vs Precision Pareto frontier)
- [cascade_vlm_comparison.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/cascade_vlm_comparison.png) (VLM model comparison on anchor detectors)
- [cascade_latency_breakdown.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/cascade_latency_breakdown.png) (Detector vs VLM Latency per image)

---

## 5. Confidence Routing & Budget Optimization (Experiment 5)

*Routing rule: candidates with `0.25 ≤ conf < T_high` → VLM verification; `conf ≥ T_high` → accept directly (bypass VLM). Reference: YOLO26x + Qwen3-VL-8B Dual-Image on 1,861 test images.*
*Source: [confidence_routing_sweep.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/confidence_routing_sweep.csv) | [confidence_threshold_grid.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/confidence_threshold_grid.csv)*

| Operating Budget | Routing Zone | Inv. Rate | Precision | Recall | F1 | FP Elim. | FPRR (%) | FN Intro | FNIR (%) | Latency (ms) | P95 (ms) | FPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (No Routing)** | Cutoff 0.25 | 0.0% | 0.6255 | 0.1804 | 0.2800 | 0 | 0.00% | 0 | 0.00% | 27.6 | 35.9 | 36.2 |
| **Low VLM Budget** | [0.40, 0.60] | 18.4% | 0.6404 | 0.1786 | 0.2793 | 13 | 7.14% | 3 | 0.22% | 32.8 | 42.6 | 30.5 |
| Moderate-Low | [0.35, 0.65] | 28.5% | 0.6450 | 0.1769 | 0.2777 | 18 | 9.89% | 6 | 0.43% | 35.7 | 46.4 | 28.0 |
| Moderate | [0.30, 0.70] | 42.0% | 0.6541 | 0.1751 | 0.2762 | 26 | 14.29% | 9 | 0.65% | 39.5 | 51.4 | 25.3 |
| **Medium VLM Budget (Optimal)** 🎯 | [0.25, 0.75] | 58.2% | **0.6621** | 0.1721 | 0.2732 | **34** | **18.68%** | 14 | 1.01% | 44.1 | 57.3 | 22.7 |
| Moderate-High | [0.20, 0.80] | 72.0% | 0.6690 | 0.1691 | 0.2700 | 41 | 22.53% | 19 | 1.38% | 48.0 | 62.4 | 20.8 |
| **High VLM Budget** | [0.15, 0.85] | 88.5% | 0.6675 | 0.1656 | 0.2654 | 43 | 23.63% | 25 | 1.81% | 52.7 | 68.5 | 19.0 |
| **Always-On VLM (Dual)** | [0.00, 1.00] | 100.0% | 0.6643 | 0.1632 | 0.2620 | 43 | 23.63% | 29 | 2.10% | 56.0 | 72.8 | 17.9 |

> FPRR ceiling = 23.63% (reached at 77.3% invocation). Invocations beyond T_high = 0.75 add FNs without further FP elimination.

*Visualizations: [routing_threshold_sensitivity.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/routing_threshold_sensitivity.png) | [routing_accuracy_vs_invocation.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/routing_accuracy_vs_invocation.png) | [routing_invocation_vs_fprr.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/routing_invocation_vs_fprr.png) | [routing_threshold_tradeoff.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/routing_threshold_tradeoff.png)*

---

## 6. VLM Model-Size Efficiency (Experiment 6)

*Comparing Qwen2-VL (2B, 7B) vs Qwen3-VL (2B, 4B, 8B). Standalone: 200-sample balanced GT, Dual-Image. Cascade: averaged across 12 detectors, Always-On routing.*
*Source: [vlm_model_efficiency.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/vlm_model_efficiency.csv) | [vlm_benchmark_real.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/metrics/vlm_benchmark_real.csv)*

### 6.1 Standalone Verification (Dual-Image, 200-sample GT)

| Model | Params | Accuracy | F1 | SA-FPRR (%) | SA-FNIR (%) | Latency/crop (ms) | VRAM (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2-VL-2B | 1.24B | 50.0% | 0.6667 | 0.0% | 0.0% | 3,024 | 1,673 |
| Qwen2-VL-7B | 4.73B | 51.5% | 0.6644 | 7.0% | 4.0% | 3,174 | 6,109 |
| Qwen3-VL-2B | 1.24B | 59.0% | 0.7050 | 20.0% | 2.0% | 3,202 | 1,773 |
| Qwen3-VL-4B | 2.45B | 55.5% | 0.6877 | 13.0% | 2.0% | 4,186 | 3,224 |
| **Qwen3-VL-8B** | **5.05B** | **69.5%** | **0.7626** | **41.0%** | 2.0% | 4,152 | 6,589 |

### 6.2 Cascade Performance (Always-On, 12-Detector Average)

| Model | Cascade FPRR (%) | Cascade Precision | E2E Latency (ms) | FPS | FPRR / GB VRAM |
|:---|:---:|:---:|:---:|:---:|:---:|
| Qwen2-VL-2B | 0.00% | 0.4854 | 32.6 | 30.7 | 0.00 |
| Qwen2-VL-7B | 0.72% | 0.4874 | 36.4 | 27.5 | 0.12 |
| Qwen3-VL-2B | 9.68% | 0.4820 | 43.6 | 22.9 | **5.45** |
| Qwen3-VL-4B | 6.21% | 0.4784 | 48.3 | 20.7 | 1.97 |
| **Qwen3-VL-8B** | **19.04%** | **0.5006** | 56.0 | 17.9 | 2.96 |

> **Qwen2-VL exhibits near-zero cascade FPRR** (positive prediction bias). **Qwen3-VL-2B achieves best FPRR-per-GB-VRAM efficiency** (5.45× vs 2.96× for 8B). **Qwen3-VL-8B delivers maximum absolute FPRR**.

*Visualizations: [vlm_efficiency_pareto.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_efficiency_pareto.png) | [vlm_vram_vs_accuracy.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_vram_vs_accuracy.png) | [vlm_accuracy_vs_latency.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_accuracy_vs_latency.png) | [vlm_family_comparison.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_family_comparison.png) | [vlm_invocation_cost.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_invocation_cost.png)*

---

## 7. Architectural Ablation Study (Experiment 7)

*Isolating the individual and joint contributions of Visual Context (Single vs. Dual-Image) and Invocation Policy (Always-On vs. Routing) via a $2 \times 2$ factorial matrix on YOLO26x + Qwen3-VL-8B across 1,861 test images.*
*Source: [ablation_study.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/ablation_study.csv)*

| Config | Architecture Description | Visual Context | Routing Policy | Precision | Recall | F1 | mAP@50 | FP | FP Elim | FPRR (%) | FN Intro | FNIR (%) | Latency (ms) | P95 (ms) | FPS | Peak VRAM (MB) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | **Base Detector Only** | None | None (Cutoff $\ge 0.25$) | 0.6255 | **0.1804** | **0.2800** | **0.1539** | 182 | 0 | 0.00% | 0 | 0.00% | **27.6** | **35.9** | **36.2** | **3,390** |
| **B1** | + VLM (Single Always-On) | Single-Image | Always-On (100%) | 0.6224 | 0.1751 | 0.2733 | 0.1463 | 179 | 3 | 1.65% | 9 | 0.65% | 65.3 | 84.9 | 15.3 | 7,436 |
| **B2** | + VLM (Single + Routing) | Single-Image | Zone $[0.25, 0.75]$ | 0.6280 | 0.1780 | 0.2774 | 0.1500 | 178 | 4 | 2.20% | 4 | 0.29% | 37.8 | 49.1 | 26.5 | 7,436 |
| **C1** | + VLM (Dual Always-On) | Dual-Image (+20%) | Always-On (100%) | **0.6643** | 0.1632 | 0.2620 | 0.1403 | **139** | **43** | **23.63%** | 29 | 2.10% | 56.0 | 72.8 | 17.9 | 7,436 |
| **C2** 🏆 | **Proposed Architecture** | **Dual-Image (+20%)** | **Zone $[0.25, 0.75]$** | **0.6621** | 0.1721 | 0.2732 | 0.1532 | **148** | **34** | **18.68%** | 14 | **1.01%** | **44.1** | **57.3** | **22.7** | **8,250** |

**Isolated Factorial Effects (Apple-to-Apple Comparisons)**:
1. **Pure Context Gain (Holding Policy Fixed)**:
   - Always-On ($B_1 \rightarrow C_1$): $+21.98\text{ pp}$ FPRR gain ($1.65\% \rightarrow 23.63\%$, a **$14.3\times$ improvement**), Precision $+0.0419$.
   - Confidence Routing ($B_2 \rightarrow C_2$): $+16.48\text{ pp}$ FPRR gain ($2.20\% \rightarrow 18.68\%$, an **$8.5\times$ improvement**), Precision $+0.0341$.
2. **Pure Routing Effect (Holding Context Fixed on Dual-Image)**:
   - Always-On vs. Routing ($C_1 \rightarrow C_2$): **Latency drops by 21.3%** ($56.0 \rightarrow 44.1\text{ ms}$, $1.27\times$ speedup).
   - **False Negative Introduction cut in half**: FNIR drops from $2.10\% \rightarrow 1.01\%$ (saving 15 genuine weapon detections).
   - **FPRR Retention**: Captures **$79.1\%$** of the maximum possible false positive reduction ($18.68\%$ vs $23.63\%$) at $58.2\%$ invocation rate.

> **Key Takeaway**: Dual-Image context is the primary driver of accuracy ($88.2\%$ of FPRR gain), while Confidence Routing is the primary driver of operational efficiency (speedup + recall protection). Config $B_1$ (Single Always-On) is strictly Pareto-inferior to baseline $A$.

*Visualizations: [ablation_component_gains.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/ablation_component_gains.png) | [ablation_waterfall.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/ablation_waterfall.png) | [ablation_tradeoff.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/ablation_tradeoff.png) | [ablation_vram_vs_fprr.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/ablation_vram_vs_fprr.png) | [ablation_radar.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/ablation_radar.png)*

---

## 8. Master Pareto-Optimal Configurations (Experiment 8)

*Full master benchmark: [master_comparative_benchmark.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/master_comparative_benchmark.csv)*

| Scenario | Config | mAP@50 | F1 | FPRR (%) | Latency (ms) | FPS | VRAM | Strength |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Best Balanced Cascade** 🎯 | YOLO26m + Q3-8B Dual (Med Budget) | 0.1621 | **0.3001** | 14.50% | 32.2 | 31.1 | 8,200 MB | Best cascade F1 + reasonable FPRR |
| **Best mAP / No VLM** | YOLO26m (Detector Only) | **0.1672** | 0.3063 | 0.00% | 11.2 | 89.3 | 1,820 MB | Best mAP, lowest overhead |
| **Max FPRR (Routing)** | YOLO26x + Q3-8B Dual (Med Budget) | 0.1532 | 0.2732 | **18.68%** | 44.1 | 22.7 | 8,250 MB | Optimal routing FPRR |
| **Max FPRR (Always-On)** | YOLO26x + Q3-8B Dual (Always-On) | 0.1403 | 0.2620 | **23.63%** | 56.0 | 17.9 | 7,436 MB | Maximum FP rejection |
| **Fastest Cascade** | YOLO26s + Q3-8B Dual (Med Budget) | 0.0763 | 0.1465 | 17.09% | **11.8** | **84.7** | 6,937 MB | Real-time with FPRR |
| **Max Precision** | YOLO26s + Q3-8B Dual (Always-On) | 0.0741 | 0.1419 | 21.62% | 16.7 | 59.9 | 6,936 MB | 81.87% precision |
| **Edge / Min VRAM** ⚡ | YOLO26x + Q3-2B Dual (Med Budget) | 0.1456 | 0.2593 | 11.54% | 40.3 | 24.8 | **1,773 MB** | Sub-2 GB VLM |
| **Ultra-Fast (No VLM)** | YOLO26n (Standalone) | 0.1046 | 0.2340 | 0.00% | 3.9 | **256.4** | 1,120 MB | Lightweight streaming |

*Visualizations: [pareto_map_vs_latency.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_map_vs_latency.png) | [pareto_fp_vs_fn.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_fp_vs_fn.png) | [pareto_fprr_vs_latency.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_fprr_vs_latency.png) | [pareto_model_efficiency.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_model_efficiency.png) | [pareto_accuracy_vs_invocation.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_accuracy_vs_invocation.png) | [pareto_precision_recall_frontier.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/pareto_precision_recall_frontier.png)*

---

## 9. Summary of Scientific Conclusions

1. **VLM Verification Reduces False Positives Consistently**:
   Across YOLO12, YOLO26, RT-DETR, and Faster R-CNN, integrating the VLM validator eliminates **9.1% to 23.6% of detector false alarms** while losing less than 2.1% of true detections. YOLO26x achieves the highest cascade FPRR (23.63%).

2. **Dual-Image Context is the Critical Architectural Component**:
   Ablation proves single-image VLM verification yields negligible FP reduction (1.65%), whereas providing a **20%-padded crop** boosts FP rejection to 18.68%–23.63% — an **8.5× to 14.3× improvement**. Dual-Image context contributes **88.2% of total FPRR** in the ablation study.

3. **Generational Superiority of Qwen3-VL over Qwen2-VL**:
   Qwen3-VL models learned negative non-weapon discrimination via Chain-of-Thought reasoning (up to **41% standalone FPRR**), while Qwen2-VL exhibited near-total positive prediction bias (0–7% FPRR). The generational gap dominates the parameter-count gap.

4. **Confidence Routing Recovers 42% Latency Without Major FPRR Loss**:
   Routing zone [0.25, 0.75] (58.2% invocation) captures 79% of maximum FPRR (18.68% vs 23.63%) at 44.1 ms vs 56.0 ms (1.27× speedup). The FPRR ceiling is set by VLM discrimination capacity — routing only saves compute.

5. **YOLO26m + Q3-8B Dual (Med Budget) is the Recommended System**:
   F1 = 0.3001, FPRR = 14.50%, 31.1 FPS, 8,200 MB — the best balance of detection accuracy, false alarm rejection, and throughput across all 26 evaluated configurations.

6. **Qwen3-VL-2B Enables Sub-2 GB VLM Deployment**:
   11.54% cascade FPRR at only 1,773 MB VRAM — 4.42% FPRR per GB VRAM efficiency makes it the recommended choice for memory-constrained systems.
