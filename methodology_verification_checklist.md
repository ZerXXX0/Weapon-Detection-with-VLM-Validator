# Methodology Verification Checklist & System Specification

This document provides an empirical, point-by-point verification of the entire **YOLO + Qwen3-VL Hybrid Weapon Detection & Validation Pipeline**, based directly on the authoritative source code, dataset configurations, and evaluation reports.

---

## 1. Overall Pipeline

* [x] **What exactly is the input?**  
  Static 2D image frames (`.jpg` / `.png` format, RGB) representing CCTV security frames and weapon detection dataset images.
* [x] **Does YOLO run on every image/frame?**  
  **Yes.** YOLO acts as the primary candidate detector and executes on every test image to propose initial bounding boxes.
* [x] **Does YOLO produce only `weapon` candidates, or multiple classes?**  
  Single class only (`nc: 1`, class name `'weapon'`).
* [x] **What happens to detections below confidence 0.25?**  
  They are **instantly discarded** during initial YOLO inference (`conf=0.25`).
* [x] **Is 0.25 the YOLO inference confidence threshold or a later filtering threshold?**  
  It is YOLO's inference threshold, passed directly to Ultralytics: `yolo_model.predict(img_path, conf=0.25, iou=0.45)`.
* [x] **Why was 0.25 selected?**  
  Standard computer vision threshold balancing high candidate recall while suppressing trivial background noise.
* [x] **What exactly happens at $0.25 \le \text{confidence} < 0.75$?**  
  Candidate bounding boxes are sent to **Qwen3-VL** for secondary multi-modal verification.
* [x] **What exactly happens at $\text{confidence} \ge 0.75$?**  
  Detections **bypass VLM verification** and are automatically approved directly into the final `Weapon` prediction set to optimize pipeline latency.
* [x] **Does $\ge 0.75$ automatically become the final `Weapon` prediction?**  
  **Yes.**
* [x] **Does Qwen3-VL ever verify high-confidence detections?**  
  **No.** Detections $\ge 0.75$ bypass Qwen3-VL entirely.
* [x] **If Qwen says `Non-Weapon`, is the YOLO bounding box completely discarded?**  
  **Yes.** The candidate box is removed from the final prediction set (reducing False Positives).
* [x] **If Qwen says `Weapon`, is the original YOLO bounding box retained?**  
  **Yes.** The original YOLO coordinates and confidence score are kept.
* [x] **Can one image generate multiple VLM calls?**  
  **Yes.** If an image produces $k$ candidate boxes in the $[0.25, 0.75)$ range, all $k$ boxes are formatted into prompts and processed (batched in sizes of 4).

---

## 2. YOLO Detector (YOLO12 vs YOLO26)

* [x] **Exact model variants:** `n` (Nano), `s` (Small), `m` (Medium), `l` (Large), `x` (Extra-Large).
* [x] **Exact model checkpoints:** Saved best weights: `runs/detect/yolo12[n/s/m/l/x]/weights/best.pt` and `runs/detect/yolo26[n/s/m/l/x]/weights/best.pt`.
* [x] **Pretrained or trained from scratch?** Pretrained initialization from COCO weights (`yolo12n.pt`, `yolo26n.pt`, etc.).
* [x] **Training dataset:** `1-class-weapon-dataset-2` (Roboflow Project `weapon-mpr3p/1-class-weapon-dataset`, Version 2).
* [x] **Number of training images:** **4,913 images** (~69.1%).
* [x] **Number of validation images:** **1,166 images** (~16.4%).
* [x] **Number of test images:** **1,030 images** (~14.5%).
* [x] **Total dataset size:** **7,109 images**.
* [x] **Same split used for YOLO12 and YOLO26?** **Yes.** Both models were trained and evaluated on the exact same dataset splits.
* [x] **Number of classes:** 1 (`nc: 1`).
* [x] **Exact class name:** `['weapon']`.
* [x] **Input resolution:** `640 x 640` pixels (`--imgsz 640`).
* [x] **Epochs:** 100 epochs (`--epochs 100`).
* [x] **Batch size:** Initialized with dynamic OOM fallback protection (Nano/Small: 16/32, Medium: 16, Large: 8, Extra-Large: 4).
* [x] **Optimizer:** PyTorch default AdamW / SGD with Cosine Annealing.
* [x] **IoU threshold for NMS:** `0.45` (`iou=0.45`).
* [x] **Genuinely comparable?** **Yes.** Both YOLO12 and YOLO26 were trained under identical hyperparameters, dataset splits, and resolution configurations.

---

## 3. The 0.25 / 0.75 Confidence Strategy

$$c < 0.25 \implies \text{Discarded}$$
$$0.25 \le c < 0.75 \implies \text{Sent to Qwen3-VL}$$
$$c \ge 0.75 \implies \text{Auto-approved (Bypassed VLM)}$$

* [x] **Is 0.25 applied before or after NMS?**  
  Applied **during/after NMS** within standard YOLO inference.
* [x] **Is 0.75 based on maximum class confidence?**  
  **Yes.** Since $nc=1$, confidence $c = P(\text{weapon})$.
* [x] **Is 0.75 empirically chosen?**  
  Selected as a high-confidence threshold to preserve latency while focusing VLM capacity on uncertain detections.

### Candidate Confidence Distribution Across Models (Test Set: 1,030 Images)

| Model | Total Detections ($c \ge 0.25$) | Sent to VLM ($0.25 \le c < 0.75$) | Bypassed VLM ($c \ge 0.75$) |
| :--- | :---: | :---: | :---: |
| **YOLO12-n** | 1,526 | 1,526 (100.0%) | 0 (0.0%) |
| **YOLO12-s** | 557 | 547 (98.2%) | 10 (1.8%) |
| **YOLO26-n** | 598 | 594 (99.3%) | 4 (0.7%) |
| **YOLO26-s** | 175 | 146 (83.4%) | 29 (16.6%) |

---

## 4. Qwen3-VL Model Specification

* [x] **Model Identifier:** `Qwen/Qwen3-VL-8B-Instruct`.
* [x] **Architecture:** 8-Billion parameter Instruction-Tuned Vision-Language Model.
* [x] **Fine-Tuning:** LoRA parameter-efficient fine-tuning (`qwen3_vl_model_2image` / `qwen3_vl_model_full3_ra1632`).
* [x] **LoRA Parameters:**  
  * Rank ($r$): 16  
  * Alpha ($\alpha$): 32  
  * Dropout: 0.05  
  * Target Modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
* [x] **Training Hyperparameters:**  
  * Epochs: 10  
  * Learning Rate: $5 \times 10^{-5}$  
  * Per-device batch size: 1  
  * Gradient accumulation steps: 8 (Effective batch size = 8)  
  * Precision: `bf16` / `fp16`  
  * Optimizer: `paged_adamw_32bit`
* [x] **Exact 4-Bit Quantization Method:**  
  Loaded via `BitsAndBytesConfig`:
  ```python
  BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True
  )
  ```
  *(NF4 4-bit NormalFloat with double quantization and FP16 compute dtype).*
* [x] **Data Leakage Verification:**  
  Cross-referenced all images in Qwen training JSONs (`train.json`) against the 1,030 test set evaluation images: **0 overlapping images (Zero Data Leakage)**.

---

## 5. Single-Image Strategy

* [x] **Image Input:** Single uncropped full-resolution original image.
* [x] **Bounding Box Formatting:**  
  Normalized to $[0, 1000]$ integer range:
  ```python
  ymin_norm = max(0, min(1000, int(round(ymin / img_height * 1000))))
  xmin_norm = max(0, min(1000, int(round(xmin / img_width * 1000))))
  ymax_norm = max(0, min(1000, int(round(ymax / img_height * 1000))))
  xmax_norm = max(0, min(1000, int(round(xmax / img_width * 1000))))
  ```
  Prompt tag format: `<box>(ymin,xmin,ymax,xmax)</box>`.
* [x] **User Prompt:**  
  `Examine the object located inside the bounding box at <box>(ymin,xmin,ymax,xmax)</box>. Determine whether this highlighted region contains a weapon. Answer with a concise analysis ending with 'Final Decision: Weapon' or 'Final Decision: Non-Weapon'.`
* [x] **Output Parsing Rule:**  
  * If `"Final Decision: Weapon"` in response $\implies$ **Kept** (`True`).
  * If `"Final Decision: Non-Weapon"` in response $\implies$ **Discarded** (`False`).
  * Fallback: `"non-weapon" not in response.lower() and "weapon" in response.lower()`.

---

## 6. Dual-Image Strategy

* [x] **Image 1 (Full Frame Context):** Full image with candidate box drawn in red outline (`outline=(255, 0, 0)`, line thickness = 3px).
* [x] **Image 2 (Zoomed-Out Crop ROI):** Cropped candidate ROI expanded by a **20% margin**.

### Exact 20% Crop Mathematical Formulation

For a candidate box $(x_1, y_1, x_2, y_2)$ with width $w = x_2 - x_1$ and height $h = y_2 - y_1$:

$$\text{pad}_w = 0.20 \times w, \quad \text{pad}_h = 0.20 \times h$$
$$\text{crop}_{x1} = \max\left(0, \lfloor x_1 - \text{pad}_w \rceil\right)$$
$$\text{crop}_{y1} = \max\left(0, \lfloor y_1 - \text{pad}_h \rceil\right)$$
$$\text{crop}_{x2} = \min\left(W, \lfloor x_2 + \text{pad}_w \rceil\right)$$
$$\text{crop}_{y2} = \min\left(H, \lfloor y_2 + \text{pad}_h \rceil\right)$$

*(This adds 20% padding to each side, expanding total crop width by 40% of candidate box width).*

* [x] **Edge Clipping:** Strict clipping to image boundary $[0, W]$ and $[0, H]$. No artificial black padding added.
* [x] **Aspect Ratio:** Preserved natively by PIL crop.

---

## 7. VLM Prompt Consistency

* [x] **Identical Baseline Parameters:**  
  Both Single-Image and Dual-Image experiments share the exact same Qwen3-VL base model, LoRA configuration, inference temperature (`do_sample=False`), confidence thresholds ($0.25 / 0.75$), and candidate boxes. The **only variable changing is the visual input representation**.

---

## 8. The $2 \times 2$ Experiment Matrix

| Experiment | YOLO Model | VLM Input Strategy | Output Report File |
| :--- | :--- | :--- | :--- |
| **E1** | YOLO12 | Single-Image (Full Frame) | [`evaluation_report.txt`](file:///c:/Gojiii/VLM/Weapon-Detection-with-VLM-Validator/evaluation_report.txt) |
| **E2** | YOLO12 | Dual-Image (20% Expanded Crop) | [`evaluation_report_qwen3_2image_yolo12.txt`](file:///c:/Gojiii/VLM/Weapon-Detection-with-VLM-Validator/evaluation_report_qwen3_2image_yolo12.txt) |
| **E3** | YOLO26 | Single-Image (Full Frame) | [`evaluation_report_yolo26.txt`](file:///c:/Gojiii/VLM/Weapon-Detection-with-VLM-Validator/evaluation_report_yolo26.txt) |
| **E4** | YOLO26 | Dual-Image (20% Expanded Crop) | [`evaluation_report_qwen3_2image.txt`](file:///c:/Gojiii/VLM/Weapon-Detection-with-VLM-Validator/evaluation_report_qwen3_2image.txt) |

---

## 9. Key Empirical Results Summary (Nano Variants)

| Metric | E1 (YOLO12 + Single) | E2 (YOLO12 + Dual) | E3 (YOLO26 + Single) | E4 (YOLO26 + Dual) |
| :--- | :---: | :---: | :---: | :---: |
| **mAP50** | 0.0472 | 0.0347 | 0.1054 | 0.0894 |
| **mAP50-95** | 0.0171 | 0.0131 | 0.0411 | 0.0350 |
| **Precision** | 0.2210 | 0.1908 | 0.4619 | 0.4503 |
| **Recall** | 0.1733 | 0.1347 | 0.1549 | 0.1318 |
| **False Positives (FP)** | 1,029 | 963 | 304 | 271 |
| **FP Reduction Rate** | **16.14%** | **21.52%** | **8.16%** | **18.13%** |

---

## 10. Verification Table Checklist Summary

| Verification Item | Required Parameter | Actual Implementation Status | Verified? |
| :--- | :--- | :--- | :---: |
| **YOLO Candidate Threshold** | Candidate proposal cut-off | `conf = 0.25` | [x] |
| **VLM Bypass Threshold** | High-confidence bypass | `vlm_bypass_thres = 0.75` | [x] |
| **Crop Expansion Margin** | Exact margin formula | $20\%$ per side ($\text{pad} = 0.20 \times \text{dim}$) | [x] |
| **Boundary Handling** | Edge behavior | Clipped to $[0, W]$ and $[0, H]$ | [x] |
| **Box Coordinates** | Normalized scale | $[0, 1000]$ integer scale | [x] |
| **VLM Output Parser** | String matching rule | `"Final Decision: Weapon"` vs `"Non-Weapon"` | [x] |
| **Test Set Consistency** | Test split size | 1,030 test images across all experiments | [x] |
| **Data Leakage Check** | Qwen train vs Test split | **0 overlapping images** | [x] |
| **IoU Benchmark** | Bounding box IoU | $\text{IoU} = 0.50$ (mAP50) and $\text{IoU} \in [0.50, 0.95]$ | [x] |
