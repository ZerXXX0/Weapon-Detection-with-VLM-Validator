#!/usr/bin/env python3
import os
import sys
import time
import glob
import json
import csv
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.common.metrics import (
    compute_tp_fp_fn,
    calculate_precision_recall_f1,
    evaluate_map
)
from experiments.common.dual_image_utils import load_yolo_ground_truth

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Object Detector Benchmark (Experiment 1)")
    parser.add_argument("--dataset-dir", type=str, default="1-class-weapon-dataset-2", help="Dataset directory")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate for quick validation")
    parser.add_argument("--csv-output", type=str, default="results/metrics/detector_benchmark.csv")
    parser.add_argument("--json-output", type=str, default="results/metrics/detector_benchmark.json")
    return parser.parse_args()

def get_model_size_mb(filepath):
    if os.path.exists(filepath):
        return round(os.path.getsize(filepath) / (1024 * 1024), 2)
    return 0.0

def evaluate_single_detector(model_type, model_path, test_images, test_labels_dir, args):
    """
    Evaluates one detector model and returns standardized metrics dict.
    """
    print(f"\nEvaluating Detector: {model_type} -> {model_path}")
    device = torch.device(args.device)

    # Clean GPU memory before benchmarking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Load model
    load_start = time.time()
    param_count = 0

    if model_type in ["yolo12", "yolo26", "rtdetr"]:
        from ultralytics import YOLO, RTDETR
        if model_type == "rtdetr":
            model = RTDETR(model_path)
        else:
            model = YOLO(model_path)
        param_count = sum(p.numel() for p in model.model.parameters())
    elif model_type == "faster_rcnn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    load_time = time.time() - load_start

    preds_list = []
    targets_list = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    latencies = []

    for img_path in tqdm(test_images, desc=f"Testing {os.path.basename(model_path)}"):
        filename = os.path.basename(img_path)
        image_pil = Image.open(img_path).convert("RGB")
        width, height = image_pil.size

        label_file = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(test_labels_dir, label_file)
        gt_boxes, gt_labels = load_yolo_ground_truth(label_path, width, height)

        pred_boxes = []
        pred_scores = []

        t0 = time.time()
        if model_type in ["yolo12", "yolo26", "rtdetr"]:
            results = model.predict(img_path, conf=args.conf_thres, iou=args.iou_thres, imgsz=args.imgsz, verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                for b in results.boxes:
                    cls_id = int(b.cls[0].item())
                    if cls_id == 0:  # weapon
                        pred_boxes.append(b.xyxy[0].cpu().numpy().tolist())
                        pred_scores.append(float(b.conf[0].item()))
        elif model_type == "faster_rcnn":
            import torchvision.transforms.functional as TF
            img_t = TF.to_tensor(image_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_t)[0]
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for b, s, l in zip(boxes, scores, labels):
                if l == 1 and s >= args.conf_thres:  # class 1: weapon
                    pred_boxes.append(b.tolist())
                    pred_scores.append(float(s))
        
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)  # ms

        # Compute TP, FP, FN
        tp, fp, fn = compute_tp_fp_fn(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Format for mAP computation
        boxes_t = torch.tensor(pred_boxes, dtype=torch.float32).reshape(-1, 4)
        scores_t = torch.tensor(pred_scores, dtype=torch.float32).reshape(-1)
        labels_t = torch.zeros(len(pred_boxes), dtype=torch.int64)
        preds_list.append({"boxes": boxes_t, "scores": scores_t, "labels": labels_t})

        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4)
        gt_labels_t = torch.zeros(len(gt_boxes), dtype=torch.int64)
        targets_list.append({"boxes": gt_boxes_t, "labels": gt_labels_t})

    # Metrics
    precision, recall, f1 = calculate_precision_recall_f1(total_tp, total_fp, total_fn)
    map_res = evaluate_map(preds_list, targets_list)
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    peak_gpu_mem = 0.0
    if torch.cuda.is_available():
        peak_gpu_mem = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    return {
        "model_name": os.path.basename(model_path),
        "architecture": model_type,
        "parameters": param_count,
        "checkpoint_size_mb": get_model_size_mb(model_path),
        "input_resolution": f"{args.imgsz}x{args.imgsz}",
        "conf_threshold": args.conf_thres,
        "nms_iou_threshold": args.iou_thres,
        "total_test_images": len(test_images),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "map_50": round(map_res["map_50"], 4),
        "map_75": round(map_res["map_75"], 4),
        "map_50_95": round(map_res["map"], 4),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "fps": round(fps, 2),
        "peak_gpu_memory_mb": peak_gpu_mem
    }

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)

    test_img_dir = os.path.join(args.dataset_dir, "test", "images")
    test_lbl_dir = os.path.join(args.dataset_dir, "test", "labels")

    image_files = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")) + glob.glob(os.path.join(test_img_dir, "*.png")))
    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f"Total Test Images for Evaluation: {len(image_files)}")

    # Detectors list (12 configurations)
    detectors = [
        ("yolo12", "yolo12n.pt"),
        ("yolo12", "yolo12s.pt"),
        ("yolo12", "yolo12m.pt"),
        ("yolo12", "yolo12l.pt"),
        ("yolo12", "yolo12x.pt"),
        ("yolo26", "yolo26n.pt"),
        ("yolo26", "yolo26s.pt"),
        ("yolo26", "yolo26m.pt"),
        ("yolo26", "yolo26l.pt"),
        ("yolo26", "yolo26x.pt"),
        ("rtdetr", "rtdetr-l.pt"),
        ("faster_rcnn", "runs/detect/faster_rcnn/weights/best.pth")
    ]

    all_results = []
    for model_type, model_path in detectors:
        if not os.path.exists(model_path):
            print(f"[WARNING] Skipping {model_path}: Checkpoint not found.")
            continue
        res = evaluate_single_detector(model_type, model_path, image_files, test_lbl_dir, args)
        all_results.append(res)

    # Save to CSV
    if all_results:
        keys = all_results[0].keys()
        with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Saved benchmark CSV: {args.csv_output}")

        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved benchmark JSON: {args.json_output}")

        # Generate Visualizations
        generate_plots(all_results)

def generate_plots(results):
    models = [r["model_name"].replace(".pt", "").replace(".pth", "") for r in results]
    map50 = [r["map_50"] for r in results]
    map50_95 = [r["map_50_95"] for r in results]
    prec = [r["precision"] for r in results]
    rec = [r["recall"] for r in results]
    lat = [r["avg_latency_ms"] for r in results]
    fps = [r["fps"] for r in results]
    gpu_mem = [r["peak_gpu_memory_mb"] for r in results]

    # 1. mAP Comparison
    plt.figure(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, map50, width, label='mAP@50', color='#2b5c8f')
    plt.bar(x + width/2, map50_95, width, label='mAP@50:95', color='#f39c12')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Detector Benchmark: mAP Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/visualizations/detector_mAP_comparison.png', dpi=300)
    plt.close()

    # 2. Precision vs Recall
    plt.figure(figsize=(8, 6))
    for m, p, r in zip(models, prec, rec):
        plt.scatter(r, p, s=120, label=m)
        plt.annotate(m, (r + 0.005, p + 0.005), fontsize=8)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Detector Benchmark: Precision vs Recall')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/visualizations/detector_precision_vs_recall.png', dpi=300)
    plt.close()

    # 3. Latency vs FPS
    plt.figure(figsize=(8, 6))
    for m, l, f in zip(models, lat, fps):
        plt.scatter(l, f, s=120, label=m)
        plt.annotate(m, (l + 0.5, f + 0.5), fontsize=8)
    plt.xlabel('Average Latency (ms)')
    plt.ylabel('FPS')
    plt.title('Detector Benchmark: Inference Latency vs FPS')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/visualizations/detector_latency_vs_fps.png', dpi=300)
    plt.close()

    # 4. GPU Memory
    plt.figure(figsize=(10, 5))
    plt.bar(models, gpu_mem, color='#27ae60')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Peak GPU Memory (MB)')
    plt.title('Detector Benchmark: Peak GPU Memory Usage')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/visualizations/detector_gpu_memory.png', dpi=300)
    plt.close()

    print("Generated all publication-quality detector benchmark plots in results/visualizations/")

if __name__ == "__main__":
    main()
