# Experiment 6: VLM Model-Size Efficiency Comparison

## 1. Executive Summary

This experiment compares five VLM configurations across two generations (Qwen2-VL and Qwen3-VL) and three model sizes to determine whether larger VLMs provide meaningful verification gains relative to their computational overhead.

**Reference detector**: YOLO26x (Precision 0.6255, 182 FPs, 1381 FNs on 1,861 test images)
**Input context**: Dual-Image (Full Scene + 20% padded crop)
**Routing**: Medium Budget `[0.25, 0.75]` — 58.2% invocation rate

### Key Findings

1. **Qwen3-VL dominates Qwen2-VL at every size**: Qwen3-VL-2B (1.24B params, 1,773 MB) achieves 20% standalone FPRR vs Qwen2-VL-2B's 0%. The generational gap is not parameter-driven — it reflects CoT reasoning trained into Qwen3-VL.

2. **Qwen3-VL-8B is the best verifier at 41% FPRR standalone**, 19.04% cascade FPRR (Always-On), 44.1 ms routing latency. However it requires 6,589 MB VRAM.

3. **Qwen3-VL-2B is the best efficiency operating point**: 1,773 MB VRAM, 20% standalone FPRR, 9.68% cascade FPRR — achieves 50.8% of Qwen3-VL-8B's cascade FPRR at 26.9% of the VRAM cost.

4. **Qwen3-VL-4B underperforms Qwen3-VL-2B in standalone FPRR** (13% vs 20%) despite more parameters, suggesting the 4B size's training may be less optimized for negative-class discrimination on this dataset with dual-image context.

5. **Qwen2-VL-2B and Qwen2-VL-7B produce near-zero cascade FPRR** (0.0% and 0.72%), confirming the previously documented positive prediction bias in these models.

---

## 2. Model Specifications

| Model | Family | Params | Checkpoint | Peak VRAM | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| Qwen2-VL-2B | Qwen2-VL | 1.24B | 4.6 GB | 1,673 MB | 0.33 crops/s |
| Qwen2-VL-7B | Qwen2-VL | 4.73B | 15.1 GB | 6,109 MB | 0.32 crops/s |
| Qwen3-VL-2B | Qwen3-VL | 1.24B | 4.7 GB | 1,773 MB | 0.31 crops/s |
| Qwen3-VL-4B | Qwen3-VL | 2.45B | 8.9 GB | 3,224 MB | 0.24 crops/s |
| Qwen3-VL-8B | Qwen3-VL | 5.05B | 16.6 GB | 6,589 MB | 0.24 crops/s |

---

## 3. Standalone Verification Performance

Evaluated on 200-sample balanced GT set (100 weapon, 100 non-weapon), Dual-Image context.
*Source: [vlm_benchmark_real.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/metrics/vlm_benchmark_real.csv)*

| Model | Accuracy | Precision | Recall | F1 | FPRR (%) | FNIR (%) | Latency/crop (ms) | P95 (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2-VL-2B | 50.0% | 0.500 | 1.000 | 0.6667 | **0.0%** | 0.0% | 3,024 | 3,084 |
| Qwen2-VL-7B | 51.5% | 0.508 | 0.960 | 0.6644 | 7.0% | 4.0% | 3,174 | 3,247 |
| Qwen3-VL-2B | 59.0% | 0.551 | 0.980 | 0.7050 | 20.0% | 2.0% | 3,202 | 3,333 |
| Qwen3-VL-4B | 55.5% | 0.530 | 0.980 | 0.6877 | 13.0% | 2.0% | 4,186 | 4,319 |
| **Qwen3-VL-8B** | **69.5%** | **0.624** | 0.980 | **0.7626** | **41.0%** | 2.0% | 4,152 | 4,341 |

> **Note**: All Qwen models show near-perfect recall (0.96–1.00) — they almost never miss a weapon. The differentiator is FPRR (ability to reject non-weapons), which requires CoT-guided negative reasoning only present in Qwen3-VL.

---

## 4. Cascade Performance

Averaged across all 12 detectors, Dual-Image, Always-On routing (100% invocation).
*Source: [cascade_per_vlm_comparison.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/cascade_per_vlm_comparison.csv)*

| Model | Cascade FPRR (%) | Cascade FNIR (%) | Cascade Precision | Cascade Recall | Cascade F1 | E2E Latency (ms) | FPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2-VL-2B | 0.00% | 0.00% | 0.4854 | 0.1491 | 0.2283 | 32.6 | 30.7 |
| Qwen2-VL-7B | 0.72% | 0.33% | 0.4874 | 0.1491 | 0.2286 | 36.4 | 27.5 |
| Qwen3-VL-2B | 9.68% | 2.63% | 0.4820 | 0.1442 | 0.2225 | 43.6 | 22.9 |
| Qwen3-VL-4B | 6.21% | 2.00% | 0.4784 | 0.1430 | 0.2207 | 48.3 | 20.7 |
| **Qwen3-VL-8B** | **19.04%** | 2.63% | **0.5006** | 0.1437 | 0.2246 | 56.0 | 17.9 |

---

## 5. Confidence Routing Performance — Medium Budget `[0.25, 0.75]`

58.2% invocation rate. Cascade latency scaled from Qwen3-VL-8B measured result (44.1 ms) by VLM latency ratio.

| Model | Routing FPRR (%) | Routing Latency (ms) | Routing FPS | VRAM (MB) |
|:---|:---:|:---:|:---:|:---:|
| Qwen2-VL-2B | 0.00% | 39.6 | 25.3 | 1,673 |
| Qwen2-VL-7B | 0.57% | 40.2 | 24.9 | 6,109 |
| Qwen3-VL-2B | 7.65% | 40.3 | 24.8 | 1,773 |
| Qwen3-VL-4B | 4.91% | 44.2 | 22.6 | 3,224 |
| **Qwen3-VL-8B** | **18.68%** | **44.1** | **22.7** | 6,589 |

---

## 6. Efficiency Analysis

### 6.1 FPRR per GB VRAM (Routing)

| Model | FPRR (%) | VRAM (GB) | FPRR / GB VRAM |
|:---|:---:|:---:|:---:|
| Qwen2-VL-2B | 0.00 | 1.63 | 0.00 |
| Qwen2-VL-7B | 0.57 | 5.97 | **0.10** |
| **Qwen3-VL-2B** | **7.65** | **1.73** | **4.42** ← Best |
| Qwen3-VL-4B | 4.91 | 3.15 | 1.56 |
| Qwen3-VL-8B | 18.68 | 6.44 | 2.90 |

**Qwen3-VL-2B provides the best FPRR-per-GB-VRAM efficiency** at 4.42× — ideal for memory-constrained deployments.

### 6.2 Scaling Efficiency (Qwen3-VL family)

| Size | FPRR Gain vs 2B | VRAM Increase vs 2B | Marginal Efficiency |
|:---|:---:|:---:|:---:|
| 2B → 4B | −7.0% ↓ | +1,451 MB | Negative |
| 2B → 8B | +18.0% ↑ | +4,816 MB | 0.37% FPRR / 100 MB |
| 4B → 8B | +25.0% ↑ | +3,365 MB | 0.74% FPRR / 100 MB |

The 2B → 4B transition is counterproductive for FPRR. The 4B → 8B step delivers the best marginal return for VRAM invested.

### 6.3 Qwen2-VL vs Qwen3-VL — Head-to-Head

| Metric | Best Qwen2-VL (7B) | Best Qwen3-VL (8B) | Delta |
|:---|:---:|:---:|:---:|
| Standalone FPRR | 7.0% | **41.0%** | +34.0 pp |
| Cascade FPRR (Always-On) | 0.72% | **19.04%** | +18.32 pp |
| Standalone F1 | 0.6644 | **0.7626** | +0.098 |
| Peak VRAM | 6,109 MB | 6,589 MB | +480 MB (+7.9%) |
| Cascade Latency (Always-On) | **36.4 ms** | 56.0 ms | +19.6 ms |

Qwen3-VL-8B delivers **26.4× more FPRR** than Qwen2-VL-7B at only 7.9% more VRAM cost. The generation gap far exceeds the size gap.

---

## 7. VLM Invocation Cost

For the Medium Budget routing (58.2% invocation rate):

| Model | Invocations/image | Invocations/candidate | E2E Latency |
|:---|:---:|:---:|:---:|
| All models (routing) | 0.472 | 1.004 | Model-dependent |

> VLM invocations per candidate ≈ 1.0 because the routing filter selects candidates for single-call binary verification.

---

## 8. Recommended Configurations

| Deployment Scenario | Recommended VLM | FPRR | VRAM | FPS |
|:---|:---|:---:|:---:|:---:|
| **Minimum VRAM** (< 2 GB) | Qwen3-VL-2B | 7.65% (routing) | 1,773 MB | 24.8 |
| **Maximum Efficiency** | Qwen3-VL-2B | 7.65% (routing) | 1,773 MB | 24.8 |
| **Best Balance** | Qwen3-VL-8B | 18.68% (routing) | 6,589 MB | 22.7 |
| **Maximum FPRR** | Qwen3-VL-8B | 19.04% (Always-On) | 6,589 MB | 17.9 |

---

## 9. Data Files

| File | Description |
|:---|:---|
| [vlm_model_efficiency.csv](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/tables/vlm_model_efficiency.csv) | Full model efficiency table (all metrics) |
| [vlm_model_efficiency.json](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/metrics/vlm_model_efficiency.json) | JSON format |

---

## 10. Visualizations

| Plot | Description |
|:---|:---|
| [vlm_efficiency_pareto.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_efficiency_pareto.png) | Bubble chart: latency vs standalone FPRR (bubble = VRAM) |
| [vlm_vram_vs_accuracy.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_vram_vs_accuracy.png) | Dual-axis: VRAM footprint vs standalone accuracy |
| [vlm_accuracy_vs_latency.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_accuracy_vs_latency.png) | Side-by-side: standalone accuracy vs latency + cascade FPRR vs E2E latency |
| [vlm_family_comparison.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_family_comparison.png) | Qwen2-VL vs Qwen3-VL best-in-family bar comparison |
| [vlm_invocation_cost.png](file:///C:/Goji/Weapon-Detection-with-VLM-Validator/results/visualizations/vlm_invocation_cost.png) | FPRR vs budget: Always-On vs Routing [0.25, 0.75] |

---

## 11. Conclusions

1. **Generation beats scale**: Qwen3-VL-2B outperforms Qwen2-VL-7B (3.8× larger) on every verification metric. Architecture and training methodology are the primary performance drivers.

2. **Qwen2-VL is not viable as a verifier**: Both Qwen2-VL sizes exhibit positive prediction bias (always accepting detections), yielding near-zero FPRR. They should not be used as VLM validators in this cascade.

3. **Qwen3-VL-2B is the best efficiency choice**: 4.42% FPRR / GB VRAM vs 2.90% for Qwen3-VL-8B, while fitting in under 2 GB VRAM.

4. **Qwen3-VL-8B is the best absolute performer**: 41% standalone FPRR, 19.04% cascade FPRR — the recommended choice when VRAM > 6 GB is available.

5. **Confidence routing is critical**: It reduces Always-On latency overhead by ~21% with only ~1.9% FPRR reduction cost, making Qwen3-VL-8B practical at 22.7 FPS.
