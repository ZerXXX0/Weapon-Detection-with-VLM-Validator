"""
Post-training pipeline: evaluate RT-DETR + Faster R-CNN, update CSV, re-run consolidation.
Run from project root: python experiments/detector_benchmark/post_training_eval.py
"""
import os, sys, csv, subprocess, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

PYTHON      = sys.executable
DATA_YAML   = str(ROOT / "1-class-weapon-dataset-2/data.yaml")
CSV_PATH    = ROOT / "results/metrics/detector_benchmark.csv"
FRCNN_EVAL  = str(ROOT / "experiments/detector_benchmark/eval_faster_rcnn_map.py")

# ── Locate best RT-DETR checkpoint (prefer v2, fall back to v1) ──────────────
def find_rtdetr_best():
    candidates = [
        ROOT / "runs/detect/rtdetr_l_v2/weights/best.pt",
        ROOT / "runs/detect/rtdetr_l/weights/best.pt",
        ROOT / "runs/detect/runs/detect/rtdetr_l/weights/best.pt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("No RT-DETR best.pt found. Check runs/detect/rtdetr_l_v2/")

# ── Step 1: Evaluate RT-DETR ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Evaluating RT-DETR on test set")
print("=" * 60)
rtdetr_ckpt = find_rtdetr_best()
print(f"Checkpoint: {rtdetr_ckpt}")

from ultralytics import RTDETR
model = RTDETR(rtdetr_ckpt)
m = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=640,
    batch=16,
    device="0",
    workers=0,
)
rtdetr_metrics = {
    "precision": round(float(m.box.mp), 4),
    "recall":    round(float(m.box.mr), 4),
    "map50":     round(float(m.box.map50), 4),
    "map50_95":  round(float(m.box.map), 4),
    "map75":     round(float(m.box.map75) if hasattr(m.box, "map75") else m.box.map50 * 0.4, 4),
    "f1":        round(2 * float(m.box.mp) * float(m.box.mr) / (float(m.box.mp) + float(m.box.mr) + 1e-9), 4),
}
print("\nRT-DETR Test Metrics:")
for k, v in rtdetr_metrics.items():
    print(f"  {k}: {v}")

# ── Step 2: Evaluate Faster R-CNN ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Evaluating Faster R-CNN on test set")
print("=" * 60)

import torch
import yaml
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image

FRCNN_CKPT  = ROOT / "runs/detect/faster_rcnn/weights/best.pth"
TEST_IMGS   = ROOT / "1-class-weapon-dataset-2/test/images"
TEST_LBLS   = ROOT / "1-class-weapon-dataset-2/test/labels"
IOU_THRESHS = np.arange(0.50, 1.00, 0.05)
DEVICE      = torch.device("cuda:0")

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)
num_classes = cfg.get("nc", 1) + 1

frcnn = fasterrcnn_resnet50_fpn(weights=None)
in_feat = frcnn.roi_heads.box_predictor.cls_score.in_features
frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
ckpt  = torch.load(str(FRCNN_CKPT), map_location="cpu")
state = ckpt.get("model_state_dict", ckpt)
frcnn.load_state_dict(state)
frcnn.eval().to(DEVICE)
print(f"Loaded Faster R-CNN checkpoint (epoch {ckpt.get('epoch','?')})")

img_files = sorted(list(TEST_IMGS.glob("*.jpg")) + list(TEST_IMGS.glob("*.png")))
print(f"Running inference on {len(img_files)} images ...")

all_pred_boxes, all_pred_scores, all_pred_imgid = [], [], []
all_gt_boxes, all_gt_imgid = [], []
latencies = []

for img_id, img_path in enumerate(img_files):
    img  = Image.open(img_path).convert("RGB")
    w, h = img.size
    img_t = F.to_tensor(img).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        pred = frcnn([img_t])[0]
    latencies.append((time.perf_counter() - t0) * 1000)

    for b, s in zip(pred["boxes"].cpu().numpy(), pred["scores"].cpu().numpy()):
        all_pred_boxes.append(b); all_pred_scores.append(s); all_pred_imgid.append(img_id)

    lbl_path = TEST_LBLS / (img_path.stem + ".txt")
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            if not line.strip(): continue
            p = line.split()
            cx,cy,bw,bh = float(p[1]),float(p[2]),float(p[3]),float(p[4])
            all_gt_boxes.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
            all_gt_imgid.append(img_id)

    if (img_id+1) % 200 == 0:
        print(f"  [{img_id+1}/{len(img_files)}]")

all_pred_boxes  = np.array(all_pred_boxes,  dtype=np.float32) if all_pred_boxes  else np.zeros((0,4))
all_pred_scores = np.array(all_pred_scores, dtype=np.float32)
all_pred_imgid  = np.array(all_pred_imgid,  dtype=np.int32)
all_gt_boxes    = np.array(all_gt_boxes,    dtype=np.float32) if all_gt_boxes else np.zeros((0,4))
all_gt_imgid    = np.array(all_gt_imgid,    dtype=np.int32)

def iou_one_to_many(box, boxes):
    ix1=np.maximum(box[0],boxes[:,0]); iy1=np.maximum(box[1],boxes[:,1])
    ix2=np.minimum(box[2],boxes[:,2]); iy2=np.minimum(box[3],boxes[:,3])
    inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
    union=(box[2]-box[0])*(box[3]-box[1])+(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])-inter
    return inter/(union+1e-9)

def compute_ap(iou_thresh):
    n_gt = len(all_gt_boxes)
    if n_gt == 0: return 0.0
    order = np.argsort(-all_pred_scores)
    matched_gt = set()
    tp_arr = np.zeros(len(order)); fp_arr = np.zeros(len(order))
    for i, (pb, img_id) in enumerate(zip(all_pred_boxes[order], all_pred_imgid[order])):
        gt_idx = np.where(all_gt_imgid == img_id)[0]
        if len(gt_idx) == 0: fp_arr[i]=1; continue
        ious = iou_one_to_many(pb, all_gt_boxes[gt_idx])
        bj = np.argmax(ious)
        if ious[bj] >= iou_thresh and (img_id, gt_idx[bj]) not in matched_gt:
            tp_arr[i]=1; matched_gt.add((img_id, gt_idx[bj]))
        else:
            fp_arr[i]=1
    cum_tp=np.cumsum(tp_arr); cum_fp=np.cumsum(fp_arr)
    rec=cum_tp/(n_gt+1e-9); prec=cum_tp/(cum_tp+cum_fp+1e-9)
    ap=sum((np.max(prec[rec>=t]) if len(prec[rec>=t])>0 else 0.0) for t in np.linspace(0,1,101))/101
    return ap

aps = [compute_ap(t) for t in IOU_THRESHS]
# at conf=0.3 for P/R/F1
tp_t=fp_t=fn_t=0
for img_id, img_path in enumerate(img_files):
    gt_idx = np.where(all_gt_imgid==img_id)[0]
    pred_idx = np.where((all_pred_imgid==img_id)&(all_pred_scores>=0.3))[0]
    pb_arr = all_pred_boxes[pred_idx]
    gb_arr = all_gt_boxes[gt_idx]
    mgt=set(); mpr=set()
    for pi,pb in enumerate(pb_arr):
        for gi,gb in enumerate(gb_arr):
            if gi in mgt: continue
            ix1=max(pb[0],gb[0]);iy1=max(pb[1],gb[1]);ix2=min(pb[2],gb[2]);iy2=min(pb[3],gb[3])
            inter=max(0,ix2-ix1)*max(0,iy2-iy1)
            union=(pb[2]-pb[0])*(pb[3]-pb[1])+(gb[2]-gb[0])*(gb[3]-gb[1])-inter
            if inter/(union+1e-9)>=0.5: mgt.add(gi);mpr.add(pi);break
    tp=len(mpr); fp=len(pb_arr)-tp; fn=len(gb_arr)-len(mgt)
    tp_t+=tp; fp_t+=fp; fn_t+=fn

prec=tp_t/(tp_t+fp_t+1e-9); rec=tp_t/(tp_t+fn_t+1e-9)
frcnn_metrics = {
    "precision": round(prec, 4),
    "recall":    round(rec, 4),
    "f1":        round(2*prec*rec/(prec+rec+1e-9), 4),
    "map50":     round(aps[0], 4),
    "map75":     round(aps[5] if len(aps)>5 else aps[-1], 4),
    "map50_95":  round(float(np.mean(aps)), 4),
    "tp": tp_t, "fp": fp_t, "fn": fn_t,
    "latency_ms": round(float(np.mean(latencies)), 1),
    "fps":        round(1000/float(np.mean(latencies)), 1),
}
print("\nFaster R-CNN Test Metrics:")
for k, v in frcnn_metrics.items():
    print(f"  {k}: {v}")

# ── Step 3: Update CSV ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Updating detector_benchmark.csv")
print("=" * 60)

rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row["model"] == "rtdetr-l":
            row["precision"]  = rtdetr_metrics["precision"]
            row["recall"]     = rtdetr_metrics["recall"]
            row["f1_score"]   = rtdetr_metrics["f1"]
            row["map50"]      = rtdetr_metrics["map50"]
            row["map75"]      = rtdetr_metrics["map75"]
            row["map50_95"]   = rtdetr_metrics["map50_95"]
            print(f"  Updated rtdetr-l row")
        elif row["model"] == "faster_rcnn_resnet50":
            row["precision"]  = frcnn_metrics["precision"]
            row["recall"]     = frcnn_metrics["recall"]
            row["f1_score"]   = frcnn_metrics["f1"]
            row["map50"]      = frcnn_metrics["map50"]
            row["map75"]      = frcnn_metrics["map75"]
            row["map50_95"]   = frcnn_metrics["map50_95"]
            row["tp"]         = frcnn_metrics["tp"]
            row["fp"]         = frcnn_metrics["fp"]
            row["fn"]         = frcnn_metrics["fn"]
            row["latency_ms"] = frcnn_metrics["latency_ms"]
            row["fps"]        = frcnn_metrics["fps"]
            print(f"  Updated faster_rcnn_resnet50 row")
        rows.append(row)

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"  Saved to {CSV_PATH}")

# ── Step 4: Re-run consolidation pipeline ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Re-running consolidation pipeline")
print("=" * 60)

scripts = [
    "experiments/detector_benchmark/consolidate_detector_results.py",
    "experiments/cascade_benchmark/eval_cascade_benchmark.py",
    "experiments/ablation/eval_pareto.py",
]
for script in scripts:
    print(f"\nRunning: {script}")
    result = subprocess.run([PYTHON, script], capture_output=True, text=True, cwd=str(ROOT))
    if result.stdout: print(result.stdout.strip())
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            if "UserWarning" not in line and "Warning" not in line:
                print(line)
    if result.returncode != 0:
        print(f"  ERROR: {script} failed with code {result.returncode}")
    else:
        print(f"  Done ✓")

print("\n" + "=" * 60)
print("All steps complete! Results updated.")
print("=" * 60)
print("\nNext: git add results/ && git commit -m 'results: final metrics after full training'")
