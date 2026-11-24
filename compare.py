#!/usr/bin/env python3
"""
Compare all experiment runs saved in runs/*/metrics.json side-by-side
and plot validation loss + origin accuracy curves to runs/comparison.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    runs_dir = Path(__file__).resolve().parent / "runs"
    metric_files = sorted(runs_dir.glob("*/metrics.json"))
    if not metric_files:
        print("No runs found in runs/*/metrics.json yet. Run train.py first!")
        return

    runs = [json.loads(p.read_text()) for p in metric_files]

    header = (
        f"{'Run Name':42s} | {'Model':8s} | {'Act':6s} | {'Norm':9s} | "
        f"{'Loss':13s} | {'Opt':12s} | {'ValLoss':7s} | {'Origin':6s} | {'Gender':6s} | {'Age':8s} | {'Time':5s}"
    )
    print(header)
    print("-" * len(header))

    for r in runs:
        c = r["config"]
        f = r["final"]
        age_val = f"{f['age_acc']:.3f}" if "age_acc" in f else f"{f['age_mae_years']:.2f}y"
        print(
            f"{r['run_name'][:42]:42s} | {c['model']:8s} | {c['activation']:6s} | {c['norm']:9s} | "
            f"{c['cls_loss'][:13]:13s} | {c['optimizer']:12s} | "
            f"{f['val_loss']:7.4f} | {f['origin_acc']:6.3f} | {f['gender_acc']:6.3f} | "
            f"{age_val:8s} | {r['elapsed_sec']:4.1f}s"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in runs:
        epochs = [h["epoch"] for h in r["history"]]
        axes[0].plot(epochs, [h["val_loss"] for h in r["history"]], marker="o", label=r["run_name"])
        axes[1].plot(epochs, [h["origin_acc"] for h in r["history"]], marker="o", label=r["run_name"])

    axes[0].set_title("Validation Loss by Experiment")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Val Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Origin Accuracy by Experiment")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = runs_dir / "comparison.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved comparison plot -> {out_png}")


if __name__ == "__main__":
    main()
