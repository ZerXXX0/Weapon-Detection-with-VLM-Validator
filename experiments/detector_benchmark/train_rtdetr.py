#!/usr/bin/env python3
import os
import sys
import argparse
from ultralytics import RTDETR

def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-DETR detector on the 1-class weapon dataset.")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="Base RT-DETR weights")
    parser.add_argument("--data", type=str, default="1-class-weapon-dataset-2/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output directory")
    parser.add_argument("--name", type=str, default="rtdetr_l", help="Run name")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience (epochs without metric improvement)")
    parser.add_argument("--resume", nargs="?", const="default", default=None, help="Resume training (optional checkpoint path, defaults to last.pt in run dir)")
    parser.add_argument("--lr0", type=float, default=0.0001, help="Initial learning rate (lower = more stable, default 0.0001)")
    parser.add_argument("--amp", type=lambda x: x.lower() != "false", default=False, help="Use AMP mixed precision (False = safer, avoids NaN on small datasets)")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 60)
    print("RT-DETR Training Configuration")
    print("=" * 60)

    # Determine if resuming
    if args.resume is not None:
        if args.resume == "default":
            # Attempt to find last.pt in default run path
            possible_last = os.path.join(args.project, args.name, "weights", "last.pt")
            if not os.path.exists(possible_last):
                # Check nested runs path if present
                nested_last = os.path.join(args.project, "runs", "detect", args.name, "weights", "last.pt")
                if os.path.exists(nested_last):
                    possible_last = nested_last
            checkpoint_path = possible_last
        else:
            checkpoint_path = args.resume

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        print(f"Resuming training from checkpoint: {checkpoint_path}")
        print(f"Data YAML  : {args.data}")
        print(f"Epochs     : {args.epochs}")
        print(f"Batch Size : {args.batch}")
        print(f"Img Size   : {args.imgsz}")
        print(f"LR0        : {args.lr0}")
        print(f"AMP        : {args.amp}")
        print(f"Patience   : {args.patience} (Early Stopping)")
        print(f"Save Path  : {args.project}/{args.name}")
        print("=" * 60)

        # NOTE: Do NOT use resume=True here — Ultralytics ignores all re-passed
        # args when resume=True, loading stale baked-in args from the checkpoint.
        # Instead, load from last.pt (which contains weights + optimizer state)
        # and pass all args explicitly with exist_ok=True to continue the run.
        model = RTDETR(checkpoint_path)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            lr0=args.lr0,
            amp=args.amp,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            workers=0  # Windows safe
        )
    else:
        print(f"Base Model : {args.model}")
        print(f"Data YAML  : {args.data}")
        print(f"Epochs     : {args.epochs}")
        print(f"Batch Size : {args.batch}")
        print(f"Img Size   : {args.imgsz}")
        print(f"LR0        : {args.lr0}")
        print(f"AMP        : {args.amp}")
        print(f"Patience   : {args.patience} (Early Stopping)")
        print(f"Save Path  : {args.project}/{args.name}")
        print("=" * 60)

        model = RTDETR(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            lr0=args.lr0,
            amp=args.amp,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            workers=0  # Windows safe
        )

    print("RT-DETR Training completed.")
    return results

if __name__ == "__main__":
    main()
