# Experiment 1: Object Detector Benchmarking Report

## 1. Executive Summary

This experiment establishes baseline detection performance across 12 distinct detector configurations spanning four model families:
- **YOLO12**: Nano (n), Small (s), Medium (m), Large (l), XLarge (x)
- **YOLO26**: Nano (n), Small (s), Medium (m), Large (l), XLarge (x)
- **RT-DETR**: Real-Time Detection Transformer (Large)
- **Faster R-CNN**: Two-stage detector with ResNet50-FPN backbone

All models were evaluated under identical conditions on the 1,030 standardized test images of the 1-class weapon dataset.

### Key Highlights
1. **Top Performer in mAP@50**: `yolo26m` achieved the highest mAP@50 (**0.1672**), followed by `rtdetr-l` (**0.1580**) and `yolo26x` (**0.1539**).
2. **Top Performer in Precision**: `yolo26s` achieved the highest standalone precision (**0.7886**), with `yolo12x` following closely (**0.7778**).
3. **Throughput Spectrum**: Inference speeds ranged from **256.4 FPS** (`yolo26n`, 3.9 ms/img) down to **31.2 FPS** (`faster_rcnn_resnet50`, 32.1 ms/img).
4. **False Positive Burden**: High-recall single-stage models generate significant false alarms—for example, `yolo12n` produced **1,227 FPs** (1.19 FP/image) and `yolo26l` produced **1,276 FPs** (1.24 FP/image). This heavy false-alarm rate confirms the imperative for downstream Vision-Language Model verification.

---

## 2. Unified Benchmark Results Table

The quantitative metrics compiled in [results/metrics/detector_benchmark.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/metrics/detector_benchmark.csv) are presented below:

| Model Architecture | Variant | mAP@50 | mAP@75 | mAP@50-95 | Precision | Recall | F1-Score | TP | FP | FN | Params (M) | Latency (ms) | FPS | GPU VRAM (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **YOLO26m** | Medium | **0.1672** | **0.0355** | **0.0667** | 0.6810 | **0.1976** | **0.3063** | 333 | 156 | 1352 | 20.5 | 11.2 | 89.3 | 1,820 |
| **RT-DETR-L** | Large | 0.1580 | 0.0245 | 0.0521 | 0.5842 | 0.1795 | 0.2746 | 298 | 212 | 1387 | 32.3 | 18.5 | 54.1 | 2,450 |
| **YOLO26x** | XLarge | 0.1539 | 0.0162 | 0.0480 | 0.6255 | 0.1804 | 0.2800 | 304 | 182 | 1381 | 57.0 | 27.6 | 36.2 | 3,390 |
| **YOLO12l** | Large | 0.1467 | 0.0146 | 0.0445 | 0.4769 | 0.1840 | 0.2655 | 310 | 340 | 1375 | 26.4 | 14.8 | 67.6 | 2,180 |
| **YOLO12s** | Small | 0.1450 | 0.0228 | 0.0502 | 0.5591 | 0.1852 | 0.2782 | 312 | 246 | 1373 | 9.3 | 6.8 | 147.1 | 1,420 |
| **Faster R-CNN** | ResNet50 | 0.1340 | 0.0185 | 0.0410 | 0.5210 | 0.1650 | 0.2506 | 274 | 252 | 1411 | 41.5 | 32.1 | 31.2 | 2,890 |
| **YOLO12x** | XLarge | 0.1135 | 0.0164 | 0.0404 | 0.7778 | 0.1246 | 0.2148 | 210 | 60 | 1475 | 59.1 | 28.4 | 35.2 | 3,450 |
| **YOLO26n** | Nano | 0.1046 | 0.0197 | 0.0410 | 0.4465 | 0.1585 | 0.2340 | 267 | 331 | 1418 | 2.4 | 3.9 | 256.4 | 1,120 |
| **YOLO26l** | Large | 0.0929 | 0.0202 | 0.0365 | 0.1965 | 0.1852 | 0.1907 | 312 | 1,276 | 1373 | 25.8 | 14.2 | 70.4 | 2,140 |
| **YOLO26s** | Small | 0.0826 | 0.0182 | 0.0320 | **0.7886** | 0.0819 | 0.1484 | 138 | 37 | 1547 | 9.6 | 6.5 | 153.8 | 1,390 |
| **YOLO12m** | Medium | 0.0640 | 0.0123 | 0.0240 | 0.2004 | 0.1739 | 0.1862 | 293 | 1,169 | 1392 | 20.2 | 11.5 | 87.0 | 1,860 |
| **YOLO12n** | Nano | 0.0430 | 0.0120 | 0.0162 | 0.1965 | 0.1780 | 0.1868 | 300 | 1,227 | 1385 | 2.6 | 4.2 | 238.1 | 1,150 |

---

## 3. Architecture Comparison & Insights

### 3.1 Single-Stage (YOLO) vs Transformer (RT-DETR) vs Two-Stage (Faster R-CNN)
- **YOLO26 Family**: Exhibits superior precision-to-parameter scaling compared to YOLO12. `yolo26m` provides the most balanced weapon detector, with an F1 score of 0.3063 and strong precision (68.1%).
- **RT-DETR-L**: Transformer-based cross-scale feature interaction and Hungarian matching provide competitive mAP@50 (0.1580) with significantly reduced false positives (212 FPs) compared to YOLO models of comparable capacity (YOLO12l had 340 FPs).
- **Faster R-CNN**: Two-stage proposal classification demonstrates robust precision (52.1%) but suffers from significantly higher computational latency (32.1 ms/img) and larger memory footprint (2,890 MB).

### 3.2 Precision vs Recall Trade-Off
Detectors configure distinct operating points along the precision-recall curve:
- Conservative detectors (`yolo26s`, `yolo12x`) achieve high precision (>77%) by adopting higher internal confidence gates, but sacrifice recall (<13%).
- Aggressive detectors (`yolo12n`, `yolo26l`) capture more weapons (recall ~18.5%) but flood the downstream system with over 1,200 false positives.
- This empirical finding directly motivates the **hybrid YOLO + VLM pipeline**: by pairing high-recall detectors with an intelligent VLM false-positive filter, the system can achieve both high recall and high precision simultaneously.

---

## 4. Visualizations

The generated publication figures are archived at:
- [results/visualizations/detector_mAP_comparison.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/detector_mAP_comparison.png) - Comparison of mAP@50 and mAP@50-95 across all 12 models.
- [results/visualizations/detector_precision_vs_recall.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/detector_precision_vs_recall.png) - Precision vs Recall scatter plot scaled by model size.
- [results/visualizations/detector_latency_vs_fps.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/detector_latency_vs_fps.png) - Latency vs FPS throughput colored by mAP@50.
- [results/visualizations/detector_gpu_memory.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/detector_gpu_memory.png) - Hardware memory footprint vs model parameters.
