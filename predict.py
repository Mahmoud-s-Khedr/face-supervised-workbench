#!/usr/bin/env python3
"""
Run local CPU inference with any saved checkpoint (`model.pt` / `best_model.pt`)
on an external image.

Usage:
  python predict.py --image /path/to/photo.jpg --checkpoint runs/m25_ce/best_model.pt
  python predict.py --image /path/to/photo.jpg --checkpoint runs/m25_*/best_model.pt --crop 0.7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from data import image_to_tensor
from models import MultiHeadFaceModel


def square_face_crop(img: Image.Image, frac: float = 0.72, y_center: float = 0.45) -> Image.Image:
    """Center-weighted square crop approximating FairFace/UTKFace tight face framing."""
    w, h = img.size
    side = int(min(w, h) * frac)
    cx = w // 2
    cy = int(h * y_center)
    left = max(0, min(w - side, cx - side // 2))
    top = max(0, min(h - side, cy - side // 2))
    return img.crop((left, top, left + side, top + side))


@torch.no_grad()
def predict_one(ckpt_path: Path, img: Image.Image, device: torch.device) -> None:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["config"]
    meta = blob["meta"]

    model = MultiHeadFaceModel(
        backbone_name=cfg["model"],
        img_size=cfg["img_size"],
        num_origins=meta["num_origins"],
        num_genders=meta["num_genders"],
        age_mode=meta["age_mode"],
        num_age_classes=meta["num_age_classes"],
        activation=cfg["activation"],
        norm=cfg["norm"],
        dropout=cfg.get("dropout", 0.2),
        pretrained=False,
    ).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()

    x = image_to_tensor(img, img_size=cfg["img_size"], augment="none").unsqueeze(0).to(device)
    preds = model(x)

    o_probs = F.softmax(preds["origin"][0], dim=-1)
    g_probs = F.softmax(preds["gender"][0], dim=-1)

    o_top = sorted(zip(meta["origin_names"], o_probs.tolist()), key=lambda kv: kv[1], reverse=True)
    g_top = sorted(zip(meta["gender_names"], g_probs.tolist()), key=lambda kv: kv[1], reverse=True)

    run_label = ckpt_path.parent.name + "/" + ckpt_path.name
    print(f"\n=== {run_label} ({cfg['model']}, {cfg['cls_loss']}, {meta['num_origins']}-origin) ===")
    print("  Gender :", " | ".join(f"{k}: {v*100:.1f}%" for k, v in g_top))
    print("  Origin :", " | ".join(f"{k}: {v*100:.1f}%" for k, v in o_top[:4]))

    if meta["age_mode"] == "classification":
        a_probs = F.softmax(preds["age"][0], dim=-1)
        a_top = sorted(zip(meta["age_names"], a_probs.tolist()), key=lambda kv: kv[1], reverse=True)
        print("  Age    :", " | ".join(f"{k}: {v*100:.1f}%" for k, v in a_top[:3]))
    else:
        age_yr = preds["age"][0].item()
        print(f"  Age    : {age_yr:.1f} years")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint", type=str, nargs="+", required=True)
    p.add_argument("--crop", type=float, default=0.0, help="Optional square face crop fraction (e.g., 0.72)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(args.image).convert("RGB")
    if args.crop > 0:
        img = square_face_crop(img, frac=args.crop)
        print(f"Loaded {args.image} -> square face crop ({img.size[0]}x{img.size[1]})")
    else:
        print(f"Loaded {args.image} ({img.size[0]}x{img.size[1]}, full frame)")

    for ckpt in args.checkpoint:
        predict_one(Path(ckpt), img, device)


if __name__ == "__main__":
    main()
