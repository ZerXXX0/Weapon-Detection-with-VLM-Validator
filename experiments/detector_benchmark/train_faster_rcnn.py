#!/usr/bin/env python3
import os
import glob
import time
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class YOLODetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_files = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.png"))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        label_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_file)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_c = float(parts[1]) * width
                        y_c = float(parts[2]) * height
                        w = float(parts[3]) * width
                        h = float(parts[4]) * height

                        xmin = max(0.0, x_c - w / 2.0)
                        ymin = max(0.0, y_c - h / 2.0)
                        xmax = min(float(width), x_c + w / 2.0)
                        ymax = min(float(height), y_c + h / 2.0)

                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            # In Faster R-CNN, 0 is background, so weapon is class 1
                            labels.append(1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        # Convert image to tensor in [0, 1]
        img_tensor = torchvision.transforms.functional.to_tensor(image)

        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def build_model(num_classes=2):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def compute_val_loss(model, data_loader, device):
    """
    Computes validation loss without updating weights.
    Torchvision Faster R-CNN requires model.train() mode to return loss dict.
    """
    model.train()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    return val_loss / len(data_loader) if len(data_loader) > 0 else 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on the 1-class weapon dataset with early stopping and resume.")
    parser.add_argument("--dataset-dir", type=str, default="1-class-weapon-dataset-2", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="runs/detect/faster_rcnn/weights", help="Directory to save checkpoints")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without validation loss improvement)")
    parser.add_argument("--resume", nargs="?", const="default", default=None, help="Resume training (optional path to .pth checkpoint; defaults to last.pth or best.pth in output-dir)")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("Faster R-CNN (ResNet50-FPN) Training Setup")
    print("=" * 60)

    train_dataset = YOLODetectionDataset(
        os.path.join(args.dataset_dir, "train", "images"),
        os.path.join(args.dataset_dir, "train", "labels")
    )
    val_dataset = YOLODetectionDataset(
        os.path.join(args.dataset_dir, "valid", "images"),
        os.path.join(args.dataset_dir, "valid", "labels")
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"Train samples : {len(train_dataset)} ({len(train_loader)} batches/epoch)")
    print(f"Val samples   : {len(val_dataset)} ({len(val_loader)} batches/epoch)")
    print(f"Max Epochs    : {args.epochs}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Learning Rate : {args.lr}")
    print(f"Patience      : {args.patience} (Early Stopping)")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = build_model(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0

    # Resume handling
    if args.resume is not None:
        if args.resume == "default":
            # Check last.pth then best.pth
            cand_last = os.path.join(args.output_dir, "last.pth")
            cand_best = os.path.join(args.output_dir, "best.pth")
            resume_path = cand_last if os.path.exists(cand_last) else cand_best
        else:
            resume_path = args.resume

        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"--> Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", checkpoint.get("loss", float("inf"))))
        patience_counter = checkpoint.get("patience_counter", 0)
        print(f"    Resumed at Epoch {start_epoch} | Best Val Loss: {best_val_loss:.4f} | Patience: {patience_counter}/{args.patience}")

    print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_t = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            epoch_loss += losses.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(train_loader)}] Train Loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        duration = time.time() - start_t

        # Validation Loss
        val_loss = compute_val_loss(model, val_loader, device)

        print(f"--> Epoch {epoch} complete in {duration:.1f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Checkpoint payload
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter
        }

        # Save last checkpoint
        last_path = os.path.join(args.output_dir, "last.pth")
        torch.save(ckpt_payload, last_path)

        # Early Stopping & Best Checkpoint Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_payload["best_val_loss"] = best_val_loss
            ckpt_payload["patience_counter"] = patience_counter
            best_path = os.path.join(args.output_dir, "best.pth")
            torch.save(ckpt_payload, best_path)
            print(f"    [+] Saved new BEST checkpoint (Val Loss: {val_loss:.4f}): {best_path}")
        else:
            patience_counter += 1
            print(f"    [-] Val loss did not improve ({val_loss:.4f} >= {best_val_loss:.4f}). Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print("=" * 60)
                print(f"Early Stopping Triggered: No validation improvement for {args.patience} consecutive epochs.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"Best checkpoint saved at: {os.path.join(args.output_dir, 'best.pth')}")
                print("=" * 60)
                break

    print("Faster R-CNN training finished successfully.")

if __name__ == "__main__":
    main()
