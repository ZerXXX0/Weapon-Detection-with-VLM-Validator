# MASTER IMPLEMENTATION PROMPT: VISION-LANGUAGE MODEL-BASED FALSE POSITIVE REDUCTION FOR WEAPON DETECTION

## 1. Project Overview

You are working on an ongoing research project involving a two-stage object detection and Vision-Language Model (VLM) verification architecture for weapon detection and false-positive reduction.

The primary research objective is to investigate whether integrating Vision-Language Models into an object detection pipeline can reduce false-positive detections while maintaining high detection accuracy and computational efficiency.

The proposed system consists of:

1. An object detector that generates weapon detection candidates and their confidence scores.
2. A VLM that verifies ambiguous detection candidates using image context.
3. A routing mechanism that determines when VLM verification is necessary.
4. A final detection decision based on the detector prediction and VLM verification result.

The experiments must evaluate detection accuracy, false-positive reduction, false-negative behavior, computational overhead, VLM invocation efficiency, and the trade-off between accuracy and inference cost.

**IMPORTANT: Before implementing anything, inspect the existing repository, codebase, dataset configuration, experiment outputs, model checkpoints, and existing evaluation scripts. Do not assume that everything needs to be implemented from scratch.**

Preserve existing implementations and results wherever possible.

---

# 2. General Implementation Requirements

### 2.1 Dataset Consistency

Use the existing dataset without modifying its images, annotations, class definitions, or train/validation/test splits.

- Preserve the existing dataset structure.
- Use identical test and validation sets across all experiments.
- Maintain the existing single-class weapon detection configuration for object detectors.
- Do not introduce additional classes or modify annotations.
- Do not perform additional training unless explicitly required.
- Use ground-truth annotations for standalone VLM evaluation.
- Use predicted bounding boxes for detector–VLM cascade evaluation.

### 2.2 Reproducibility

Every experiment must:

- Use deterministic evaluation settings wherever possible.
- Record model configurations and checkpoint paths.
- Record inference parameters and confidence thresholds.
- Record hardware and software configurations.
- Record inference batch size and image resolution.
- Record GPU memory consumption.
- Save raw predictions and evaluation results.
- Save experiment configuration files.
- Use consistent evaluation scripts across comparable experiments.

### 2.3 Experiment Organization

Organize experiments into independent directories.

Suggested structure:

experiments/
    detector_benchmark/
    vlm_benchmark/
    vlm_context_comparison/
    cascade_benchmark/
    confidence_routing/
    model_efficiency/
    ablation/
    robustness/

results/
    raw_predictions/
    metrics/
    tables/
    visualizations/

configs/
    detectors/
    vlms/
    routing/

Each experiment must have its own configuration and output directory.

Use CSV or JSON for machine-readable experiment results.

Use Markdown reports for experiment summaries.

---

# 3. Experiment 1: Object Detector Benchmarking

## Objective

Establish baseline object detection performance before integrating Vision-Language Models.

## Models

### Already completed

- YOLO12n
- YOLO12s
- YOLO12m
- YOLO12l
- YOLO12x
- YOLO26n
- YOLO26s
- YOLO26m
- YOLO26l
- YOLO26x

Reuse existing checkpoints and evaluation results. Do not retrain these models unless existing results are incomplete or incompatible with the unified evaluation protocol.

### To implement

1. RT-DETR
2. Faster R-CNN with ResNet50-FPN backbone.

Use pretrained weights and fine-tune using the existing training dataset and configuration conventions where appropriate.

Ensure both models use the same dataset split, image dimensions, class mapping, and evaluation conditions.

## Evaluation Metrics

Calculate:

- Precision
- Recall
- F1-score
- mAP@50
- mAP@50:95
- False Positives (FP)
- False Negatives (FN)
- Inference latency (milliseconds/image)
- FPS
- GPU memory consumption (MB or GB)

Also record:

- Number of model parameters.
- Model checkpoint size.
- Input resolution.
- Confidence threshold.
- NMS IoU threshold, where applicable.

## Outputs

Generate:

1. Unified detector benchmark CSV.
2. JSON containing detailed metrics.
3. Comparison tables for all detector architectures.
4. Precision versus recall visualization.
5. mAP comparison visualization.
6. Inference latency versus FPS visualization.
7. GPU memory comparison.

The results must include all 12 detector configurations.

---

# 4. Experiment 2: Vision-Language Model Benchmarking

## Objective

Evaluate the standalone verification capabilities of different VLM architectures without involving object detector predictions.

This experiment is intended to establish the intrinsic verification capability of each VLM.

## Models

### Qwen2-VL
- Qwen2-VL-2B
- Qwen2-VL-7B

### Qwen3-VL
- Qwen3-VL-2B
- Qwen3-VL-4B
- Qwen3-VL-8B

Use the existing model loading and quantization configurations where applicable.

Evaluate models under identical conditions.

## Evaluation Method

Use ground-truth bounding boxes from the validation and test datasets.

Extract the corresponding weapon and non-weapon image regions using ground-truth annotations.

Provide the VLM with the image and a standardized verification instruction.

The VLM must determine whether the provided region contains a weapon.

Use a standardized binary output format:

WEAPON

NOT_WEAPON

Do not allow models to generate arbitrary classification labels.

Normalize generated responses into the two supported classes. Record invalid or ambiguous responses separately rather than silently converting them into correct predictions.

Use the same prompt, image preprocessing, and evaluation samples for all models.

## Evaluation Metrics

### Classification Metrics

- Verification Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- False Positive Rate
- False Negative Rate

### False-Positive Reduction Metrics

Calculate:

- False Positive Rejection Rate (FPRR).
- False Negative Introduction Rate (FNIR).

For standalone GT evaluation, report the classification-level false-positive rejection and false-negative behavior separately from detector-level cascade metrics.

### NLP Metrics

If the VLM produces textual responses or explanations, evaluate generated text using appropriate NLP metrics, such as:

- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore

Do not apply text-generation metrics artificially to binary classification outputs. If the model only returns WEAPON or NOT_WEAPON, prioritize classification metrics.

### Computational Metrics

- Average inference latency.
- P95 inference latency.
- GPU memory consumption.
- Model parameter count.
- Model loading time.
- Throughput.

## Outputs

- VLM benchmark CSV.
- Per-model confusion matrices.
- Verification accuracy comparison.
- F1-score comparison.
- Latency comparison.
- GPU memory comparison.
- Detailed per-sample prediction logs.

---

# 5. Experiment 3: VLM Input Context Comparison

## Objective

Investigate how the visual context provided to a VLM influences verification performance and computational efficiency.

## Input Configurations

### Configuration A: Full Image Only

Provide the entire original image to the VLM.

The model must determine whether the relevant weapon is present based on the full image context.

Use the candidate bounding box coordinates as spatial information if necessary, but do not provide a cropped image.

### Configuration B: Full Image + Candidate Bounding-Box Crop

Already implemented.

Provide both:

1. Original full image.
2. Cropped candidate region corresponding to the detector bounding box.

The VLM must use the global scene context and local object appearance to verify the detection.

Reuse the existing implementation.

## Evaluation

Compare:

- Verification Accuracy.
- Precision.
- Recall.
- F1-score.
- False Positive Rejection Rate.
- False Negative Introduction Rate.
- Average inference latency.
- P95 inference latency.
- GPU memory consumption.

Ensure both configurations use identical evaluation samples and VLM models.

For standalone VLM testing, use GT bounding boxes. For cascade testing, use detector-generated candidate bounding boxes.

## Outputs

Generate a context comparison table and visualizations showing the relationship between input context, verification performance, and inference latency.

---

# 6. Experiment 4: Detector + VLM Cascade Benchmarking

## Objective

Evaluate the performance of combining different object detectors with different VLM architectures.

This is one of the primary experiments of the research.

The objective is to determine whether the VLM can effectively reject false-positive detections generated by different detector architectures.

## Detector Configurations

- YOLO12
- YOLO26
- RT-DETR
- Faster R-CNN ResNet50-FPN

For YOLO12 and YOLO26, evaluate all five model sizes where existing checkpoints are available.

## VLM Configurations

### Qwen2-VL

- Qwen2-VL-2B
- Qwen2-VL-7B

### Qwen3-VL

- Qwen3-VL-2B
- Qwen3-VL-4B
- Qwen3-VL-8B

## Required Combinations

| Detector | VLM |
|---|---|
| YOLO12 | Qwen2-VL |
| YOLO26 | Qwen2-VL |
| RT-DETR | Qwen2-VL |
| Faster R-CNN | Qwen2-VL |
| YOLO12 | Qwen3-VL |
| YOLO26 | Qwen3-VL |
| RT-DETR | Qwen3-VL |
| Faster R-CNN | Qwen3-VL |

Evaluate individual VLM sizes within each architecture.

## Cascade Implementation

1. Load the detector.
2. Perform object detection.
3. Collect predicted weapon candidates.
4. Extract candidate bounding-box crops.
5. Pass the appropriate visual context to the VLM.
6. Receive the VLM verification result.
7. Retain or reject the detection according to the verification result.
8. Generate final predictions.
9. Evaluate final predictions against ground-truth annotations.

Use the same detection and verification policy across all comparable combinations.

Do not allow the VLM to introduce new bounding boxes. It must only verify existing detector candidates.

Preserve original bounding-box coordinates and class information.

Handle images with zero detector predictions correctly. Do not invoke the VLM unnecessarily.

## Evaluation

Compare:

- Detector-only performance.
- Detector + VLM performance.
- Precision.
- Recall.
- F1-score.
- mAP@50.
- mAP@50:95.
- FP.
- FN.
- False Positive Reduction Rate.
- False Negative Introduction Rate.
- Average end-to-end latency.
- P95 end-to-end latency.
- FPS.
- GPU memory consumption.
- VLM invocation count.

Calculate false-positive reduction relative to the corresponding detector-only baseline:

FPRR = (FP_detector - FP_cascade) / FP_detector × 100%

Calculate false-negative introduction relative to the detector-only baseline:

FNIR = (FN_cascade - FN_detector) / FN_detector × 100%

Also report absolute FP and FN differences to avoid misleading percentages when the baseline denominator is zero.

A negative FNIR indicates that the cascade reduces false negatives.

## Outputs

- Detector-only versus cascade comparison.
- Per-detector VLM comparison.
- Per-VLM detector comparison.
- Complete combination benchmark.
- Confusion matrices.
- FP reduction visualizations.
- Accuracy versus latency plots.

---

# 7. Experiment 5: Fixed Confidence-Based Routing

## Objective

Investigate whether confidence-based routing can reduce unnecessary VLM invocations while maintaining detection performance.

Instead of invoking the VLM for every detection candidate, use detector confidence scores to determine which candidates require additional verification.

## Routing Mechanism

Implement a confidence-based routing system.

Let C represent the detector confidence score and T represent the routing threshold.

The initial routing policy should be:

- Low confidence: Invoke VLM.
- High confidence: Accept detector prediction without VLM verification.

Candidates should be routed according to the configured threshold.

Use a threshold grid with increments of 0.05.

Suggested grid:

0.05, 0.10, 0.15, 0.20, ..., 0.95.

Include all thresholds that are meaningful under the existing detector confidence configuration.

Clearly document the exact routing rule and ensure the threshold is applied consistently.

## VLM Invocation Budgets

Evaluate three operational budgets:

### Low VLM Budget

Minimize VLM invocations and prioritize computational efficiency.

### Medium VLM Budget

Balance detection performance and computational efficiency.

### High VLM Budget

Prioritize verification coverage and detection performance.

Define these budgets quantitatively using measured VLM invocation rates or routing policies. Do not assign arbitrary budget labels without reporting their actual invocation percentages.

## Evaluation

For each threshold and budget, calculate:

- Precision.
- Recall.
- F1-score.
- mAP@50.
- mAP@50:95.
- FP.
- FN.
- False Positive Reduction Rate.
- False Negative Introduction Rate.
- VLM Invocation Rate.
- Average end-to-end latency.
- P95 latency.
- FPS.
- GPU memory.

## Threshold Sensitivity Analysis

Investigate how changing the confidence threshold affects:

1. Detection performance.
2. False-positive reduction.
3. False-negative behavior.
4. VLM invocation rate.
5. End-to-end latency.
6. Computational efficiency.

## Outputs

Generate:

- Confidence threshold comparison CSV.
- Threshold sensitivity plots.
- F1-score versus threshold.
- FP reduction versus threshold.
- VLM invocation rate versus threshold.
- Latency versus threshold.
- Accuracy versus VLM invocation rate.

Identify the best operating points for low, medium, and high VLM invocation budgets.

---

# 8. Experiment 6: VLM Model-Size Efficiency Comparison

## Objective

Investigate whether larger VLM architectures provide meaningful verification improvements relative to their computational overhead.

## Models

- Qwen3-VL-2B
- Qwen3-VL-4B
- Qwen3-VL-8B
- Qwen2-VL-2B
- Qwen2-VL-7B

## Evaluation Conditions

Use identical:

- Detector checkpoints.
- Evaluation dataset.
- Input context.
- Confidence-routing policies.
- Routing thresholds.
- VLM prompts.
- Image preprocessing.
- Quantization configurations, where possible.

Evaluate all models under identical routing policies.

Do not compare a model using always-on verification against another using selective verification.

## Metrics

- Verification Accuracy.
- Precision.
- Recall.
- F1-score.
- False Positive Rejection Rate.
- False Negative Introduction Rate.
- Average inference latency.
- P95 inference latency.
- GPU memory.
- Model parameter count.
- VLM invocation rate.
- Total VLM invocation count.
- Average end-to-end latency.

Also calculate VLM invocation cost as the number of VLM calls per image and per detection candidate.

## Outputs

Generate:

- Model-size comparison table.
- Qwen2-VL versus Qwen3-VL comparison.
- Accuracy versus latency.
- Accuracy versus GPU memory.
- Accuracy versus invocation cost.
- Model-size efficiency analysis.

---

# 9. Experiment 7: Component Ablation Study

## Objective

Determine the contribution of each component of the proposed architecture.

Evaluate the following configurations.

### A. Detector Only

Use the original object detector without VLM verification.

This serves as the baseline.

### B. Detector + Always-On VLM

Invoke the VLM for every valid detector candidate.

This measures the performance of unrestricted VLM verification.

### C. Detector + Fixed-Threshold Routing + VLM

Invoke the VLM only for candidates satisfying the configured confidence-routing policy.

This measures the contribution of selective verification.

### D. Complete Proposed Architecture

Evaluate the complete architecture using all components defined in the existing research implementation.

Before implementing this experiment, inspect the existing architecture and research code to identify its actual components.

Do not invent additional architectural components or change the proposed method.

## Evaluation Metrics

- Precision.
- Recall.
- F1-score.
- mAP@50.
- mAP@50:95.
- FP.
- FN.
- False Positive Reduction Rate.
- False Negative Introduction Rate.
- VLM Invocation Rate.
- Average end-to-end latency.
- P95 latency.
- FPS.
- GPU memory consumption.

## Outputs

Create an ablation comparison table and visualizations showing the contribution of each architectural component.

Use identical detector checkpoints, datasets, and evaluation conditions.

---

# 10. Experiment 8: Accuracy and Computational Efficiency Evaluation

## Objective

Perform a consolidated evaluation of the proposed architecture and all relevant baselines.

This experiment should reuse results generated by Experiments 1–7 rather than unnecessarily rerunning inference.

## Accuracy Metrics

- Precision.
- Recall.
- F1-score.
- mAP@50.
- mAP@50:95.
- False Positive Reduction Rate.
- False Negative Increase/Reduction Rate.

## Computational Metrics

- VLM Invocation Rate.
- Total VLM Invocation Count.
- Average end-to-end inference latency.
- P95 end-to-end inference latency.
- FPS.
- Throughput (images/second).
- GPU memory consumption.
- Model parameter count.
- Model checkpoint size.

## Required Analysis

### A. Accuracy versus Latency

Investigate the relationship between detection accuracy and computational overhead.

### B. Accuracy versus VLM Invocation Rate

Investigate whether lower VLM invocation rates can maintain acceptable detection performance.

### C. FP Reduction versus FN Introduction

Investigate whether reducing false positives introduces additional false negatives.

### D. Model Efficiency

Compare the effectiveness of different detector–VLM combinations relative to their computational costs.

### E. Pareto Analysis

Identify configurations that provide the best trade-offs between accuracy, latency, and VLM invocation rate.

Identify Pareto-optimal configurations rather than selecting the highest-accuracy model alone.

## Outputs

Generate consolidated CSV files, comparison tables, and publication-quality visualizations.

All metrics must be traceable to the original experiment outputs.

---

# 11. Experiment 9: Robustness Evaluation (Future Work)

IMPORTANT: This experiment is planned for a later stage. Do not implement or execute it during the current implementation phase.

The objective is to investigate whether the proposed architecture remains reliable under challenging visual conditions.

## Challenging Conditions

1. Low-light images.
2. Motion-blurred images.
3. Low-resolution images.
4. Partially occluded weapons.
5. Small weapon objects.
6. Different camera viewpoints.
7. Cluttered backgrounds.
8. Different environmental conditions.

## Evaluation Objectives

Investigate:

- Detector performance under challenging conditions.
- VLM verification accuracy.
- False-positive rejection reliability.
- False-negative introduction.
- Confidence-routing reliability.
- VLM invocation behavior under challenging conditions.
- End-to-end detection performance.

The implementation should eventually support condition-specific evaluation and qualitative visualization.

For now, only ensure the experiment framework can accommodate future robustness datasets and evaluation categories.

Do not artificially degrade the existing dataset or generate synthetic robustness results.

---

# 12. Unified Metric Definitions

Use consistent metric definitions throughout the entire project.

### Detection Metrics

Precision:

Precision = TP / (TP + FP)

Recall:

Recall = TP / (TP + FN)

F1-score:

F1 = 2 × Precision × Recall / (Precision + Recall)

mAP@50 and mAP@50:95 must follow standard object detection evaluation conventions.

For object detection, TP, FP, and FN must be calculated using a documented IoU matching threshold and one-to-one ground-truth/prediction matching.

Report the IoU threshold used for precision, recall, and F1-score separately from mAP.

### False Positive Reduction Rate

FPRR = (FP_baseline - FP_proposed) / FP_baseline × 100%

### False Negative Introduction Rate

FNIR = (FN_proposed - FN_baseline) / FN_baseline × 100%

### VLM Invocation Rate

Invocation Rate = Number of VLM Calls / Number of Eligible Detection Candidates × 100%

Also report invocation rate per image:

Image Invocation Rate = Number of Images with at Least One VLM Call / Total Number of Images × 100%

### FPS

FPS = Number of Processed Images / Total Inference Time

### End-to-End Latency

Measure the complete pipeline:

Detector inference + candidate processing + image preprocessing + VLM inference + final postprocessing.

Report both average and P95 latency.

For FPS and latency, distinguish between sequential per-image latency and throughput under batched or concurrent execution.

### GPU Memory

Measure peak GPU memory allocated and reserved during inference.

Ensure measurements include the relevant model-loading and inference configuration, and distinguish model memory from peak inference memory.

---

# 13. Important Evaluation and Implementation Considerations

These requirements are mandatory.

1. Never use ground-truth bounding boxes as detector predictions during cascade evaluation.
2. Never use detector-predicted bounding boxes as substitutes for ground-truth boxes during standalone VLM benchmarking.
3. Prevent data leakage between training, validation, and test sets.
4. Use validation data for threshold selection and model/policy tuning.
5. Keep the test set untouched for final evaluation.
6. Use the same test samples for all final comparisons.
7. Do not silently discard invalid VLM responses.
8. Do not count rejected detector candidates as new false negatives without checking their relationship to ground-truth objects.
9. Use one-to-one IoU matching when calculating final detection TP, FP, and FN.
10. Preserve original detection coordinates when processing cropped images.
11. Do not let the VLM generate new object detections.
12. Avoid unnecessary model reloading between samples.
13. Avoid repeatedly loading the same model for different experiments when sequential execution can reuse it.
14. Ensure GPU memory measurements are not contaminated by previous model allocations.
15. Record quantization settings because they can affect verification performance.
16. Distinguish detector inference latency, VLM inference latency, and total end-to-end latency.
17. Distinguish candidate-level VLM invocation counts from image-level invocation counts.
18. Ensure that a detection rejected by the VLM is actually removed from the final prediction set.
19. Do not claim an improvement in mAP without recomputing final detection metrics using the complete prediction set.
20. Never overwrite existing experiment outputs or checkpoints.

---

# 14. Required Experiment Outputs

Every completed experiment must produce the following where applicable:

1. Configuration file (JSON/YAML).
2. Raw prediction logs.
3. Per-sample evaluation results.
4. Aggregated metrics (CSV).
5. Detailed metrics (JSON).
6. Confusion matrices.
7. Comparison tables.
8. Publication-quality figures.
9. Execution logs.
10. Markdown experiment report.

Each result must contain enough metadata to reproduce the experiment.

Suggested experiment naming convention:

experiment_{number}_{model}_{dataset_split}_{configuration}

Example:

experiment_04_yolo26m_qwen3vl8b_test_full_crop

---

# 15. Implementation Workflow

Follow these implementation phases in order.

## Phase 1: Repository Inspection

Before coding:

1. Inspect the entire repository structure.
2. Identify existing detector training and inference scripts.
3. Identify existing YOLO12 and YOLO26 checkpoints.
4. Identify existing evaluation results.
5. Inspect the existing Qwen2-VL and Qwen3-VL implementations.
6. Inspect existing VLM prompt templates.
7. Inspect the existing full-image + candidate-crop implementation.
8. Identify dataset paths and split configurations.
9. Identify GPU and memory constraints.
10. Identify reusable utilities and evaluation functions.

Produce an implementation assessment identifying:

- Already completed experiments.
- Partially completed experiments.
- Missing experiments.
- Existing reusable components.
- Potential compatibility issues.

Do not modify the repository during the initial inspection.

## Phase 2: Implementation Planning

Create an implementation roadmap.

Prioritize:

1. Unified detector evaluation.
2. RT-DETR and Faster R-CNN implementation.
3. Standalone VLM benchmarking.
4. VLM context comparison.
5. Detector–VLM cascade implementation.
6. Confidence-based routing.
7. Model-size efficiency comparison.
8. Component ablation.
9. Consolidated evaluation.

Robustness evaluation remains deferred.

Identify dependencies and potential GPU memory limitations before implementation.

## Phase 3: Implementation

Implement modular, reusable code.

Avoid duplicating detector inference, VLM loading, preprocessing, and metric calculations.

Use configurable experiment scripts instead of hardcoded model combinations.

The implementation should support running individual experiments and complete experiment batches.

## Phase 4: Validation

Before running the complete benchmark:

1. Run a small evaluation subset.
2. Verify detector predictions.
3. Verify VLM output normalization.
4. Verify bounding-box coordinate transformations.
5. Verify cascade filtering.
6. Verify FP/FN calculations.
7. Verify confidence-routing behavior.
8. Verify latency measurements.
9. Verify output files.

Only proceed to complete evaluation after validating these components.

## Phase 5: Execution

Execute experiments systematically.

Prioritize memory-efficient execution given the available GPU resources.

Use sequential model execution where appropriate.

Save intermediate results so interrupted experiments can resume without restarting completed work.

Do not rerun completed experiments unnecessarily.

## Phase 6: Reporting

Produce a final experiment summary containing:

- Experiment completion status.
- Models evaluated.
- Dataset splits used.
- Best-performing detector.
- Best-performing standalone VLM.
- Best-performing detector–VLM combination.
- Best confidence-routing threshold.
- Best low, medium, and high invocation-budget configurations.
- Most computationally efficient configuration.
- Best accuracy–latency trade-off.
- Ablation results.
- Limitations and failed experiments.
- Recommendations for subsequent research.

---

# 16. Final Instructions

Treat this as a scientific research implementation, not merely a software development task.

Prioritize:

- Experimental validity.
- Fair comparisons.
- Reproducibility.
- Correct metric calculations.
- Computational efficiency.
- Modular implementation.
- Transparent reporting.

Do not fabricate experimental results, infer missing metrics, or substitute theoretical values for actual measurements.

Do not change the research methodology without explicitly documenting the reason.

Do not retrain models or regenerate completed results unnecessarily.

**Start with Phase 1: Repository Inspection.**

First provide the repository assessment and implementation roadmap. Wait for approval before making substantial changes to the existing implementation.

After approval, implement the experiments incrementally, validating each component before proceeding to the next phase.