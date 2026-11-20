#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image, ImageDraw

def format_coordinate_prompt(box, img_width, img_height):
    """
    Format single-image normalized coordinates: <box>(ymin,xmin,ymax,xmax)</box>
    Coordinates are normalized to [0, 1000].
    """
    xmin, ymin, xmax, ymax = box
    ymin_norm = int(round(ymin / img_height * 1000))
    xmin_norm = int(round(xmin / img_width * 1000))
    ymax_norm = int(round(ymax / img_height * 1000))
    xmax_norm = int(round(xmax / img_width * 1000))
    
    ymin_norm = max(0, min(1000, ymin_norm))
    xmin_norm = max(0, min(1000, xmin_norm))
    ymax_norm = max(0, min(1000, ymax_norm))
    xmax_norm = max(0, min(1000, xmax_norm))
    
    return f"<box>({ymin_norm},{xmin_norm},{ymax_norm},{xmax_norm})</box>"

def prepare_dual_images(image_pil, box, pad_ratio=0.20):
    """
    Prepare two images:
    Image 1: Full frame with candidate bounding box drawn in 3px red outline.
    Image 2: Crop of candidate box expanded by 20% on each side, clipped to image boundary.
    """
    width, height = image_pil.size
    xmin, ymin, xmax, ymax = box
    
    # Ensure coordinates are within valid range
    xmin = max(0, min(width - 1, xmin))
    ymin = max(0, min(height - 1, ymin))
    xmax = max(xmin + 1, min(width, xmax))
    ymax = max(ymin + 1, min(height, ymax))
    
    # 1. Full Image with red bounding box
    img1_full = image_pil.copy()
    draw = ImageDraw.Draw(img1_full)
    for offset in range(3):
        draw.rectangle(
            [xmin - offset, ymin - offset, xmax + offset, ymax + offset],
            outline=(255, 0, 0)
        )
        
    # 2. Zoomed-out Crop ROI (+20% padding per side)
    box_w = xmax - xmin
    box_h = ymax - ymin
    pad_w = pad_ratio * box_w
    pad_h = pad_ratio * box_h
    
    crop_xmin = max(0, int(round(xmin - pad_w)))
    crop_ymin = max(0, int(round(ymin - pad_h)))
    crop_xmax = min(width, int(round(xmax + pad_w)))
    crop_ymax = min(height, int(round(ymax + pad_h)))
    
    img2_crop = image_pil.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
    
    return img1_full, img2_crop

def load_yolo_ground_truth(label_path, img_width, img_height):
    """Load YOLO ground truth label file into pixel coordinates."""
    boxes = []
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    xmin = (x_c - w / 2) * img_width
                    ymin = (y_c - h / 2) * img_height
                    xmax = (x_c + w / 2) * img_width
                    ymax = (y_c + h / 2) * img_height
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls_id)
                    
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
