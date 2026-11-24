#!/usr/bin/env python3
"""
Export a saved MultiHeadFaceModel checkpoint to a single-file ONNX model
for browser (onnxruntime-web) or Go/C++ inference, and verify numerical parity
against PyTorch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models import MultiHeadFaceModel


class OnnxFaceWrapper(nn.Module):
    """Returns a tuple (origin, gender, age) instead of a dict for clean ONNX outputs."""

    def __init__(self, model: MultiHeadFaceModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out["origin"], out["gender"], out["age"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="runs/champion_cnn_100ep/best_model.pt")
    p.add_argument("--out-dir", type=str, default="docs")
    p.add_argument("--name", type=str, default="champion_cnn.onnx")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / args.name

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, meta = blob["config"], blob["meta"]
    img_size = int(cfg["img_size"])

    model = MultiHeadFaceModel(
        backbone_name=cfg["model"],
        img_size=img_size,
        num_origins=meta["num_origins"],
        num_genders=meta["num_genders"],
        age_mode=meta["age_mode"],
        num_age_classes=meta["num_age_classes"],
        activation=cfg["activation"],
        norm=cfg["norm"],
        dropout=cfg.get("dropout", 0.2),
        pretrained=False,
    )
    model.load_state_dict(blob["model_state_dict"])
    model.eval()

    wrapper = OnnxFaceWrapper(model).eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["origin", "gender", "age"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Save metadata JSON for the browser app
    meta_out = {
        "img_size": img_size,
        "origin_names": meta["origin_names"],
        "gender_names": meta["gender_names"],
        "age_names": meta["age_names"],
        "age_mode": meta["age_mode"],
        "params": sum(p.numel() for p in model.parameters()),
        "source_checkpoint": str(ckpt_path),
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta_out, indent=2))

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported {onnx_path} ({size_mb:.2f} MB) + {out_dir / 'model_meta.json'}")

    # Verify exact numerical parity against PyTorch via onnxruntime
    import onnxruntime as ort

    with torch.no_grad():
        pt_o, pt_g, pt_a = wrapper(dummy)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_o, ort_g, ort_a = sess.run(None, {"image": dummy.numpy()})

    max_diff = max(
        float(np.max(np.abs(pt_o.numpy() - ort_o))),
        float(np.max(np.abs(pt_g.numpy() - ort_g))),
        float(np.max(np.abs(pt_a.numpy() - ort_a))),
    )
    print(f"PyTorch <-> ONNX max_abs_diff = {max_diff:.2e}")
    assert max_diff < 1e-4, f"Parity check failed: {max_diff}"
    print("ONNX parity check PASSED!")


if __name__ == "__main__":
    main()
