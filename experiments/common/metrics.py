#!/usr/bin/env python3
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def calculate_iou(box1, box2):
    """Calculate IoU between box1 and box2 in [xmin, ymin, xmax, ymax] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0.0:
        return 0.0
    return float(intersection / union)

def compute_tp_fp_fn(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Perform greedy one-to-one matching between predictions and ground truths.
    Sorted descending by prediction score.
    Returns: (tp, fp, fn) counts.
    """
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
        
    sort_idx = np.argsort(pred_scores)[::-1]
    sorted_boxes = np.array(pred_boxes)[sort_idx]
    
    tp = 0
    fp = 0
    matched_gt = set()
    
    for p_box in sorted_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(p_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
            
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn

def calculate_precision_recall_f1(tp, fp, fn):
    """Calculate Precision, Recall, and F1 with division by zero guard."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def calculate_fprr(fp_baseline, fp_proposed):
    """False Positive Reduction Rate: (FP_base - FP_prop) / FP_base * 100%"""
    if fp_baseline == 0:
        return 0.0
    return ((fp_baseline - fp_proposed) / fp_baseline) * 100.0

def calculate_fnir(fn_baseline, fn_proposed):
    """False Negative Introduction Rate: (FN_prop - FN_base) / FN_base * 100%"""
    if fn_baseline == 0:
        return 0.0
    return ((fn_proposed - fn_baseline) / fn_baseline) * 100.0

def evaluate_map(preds, targets):
    """
    Compute mAP50, mAP75, and mAP50-95 using torchmetrics.
    preds: list of dicts with 'boxes', 'scores', 'labels'
    targets: list of dicts with 'boxes', 'labels'
    """
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    res = metric.compute()
    return {
        "map_50": float(res["map_50"].item()),
        "map_75": float(res["map_75"].item()),
        "map": float(res["map"].item())
    }
