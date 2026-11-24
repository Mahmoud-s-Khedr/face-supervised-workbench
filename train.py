#!/usr/bin/env python3
"""
Modular Supervised Learning Workbench (PyTorch).

Every layer of the training pipeline is swappable via CLI flags so you can
run controlled A/B comparisons:
  - Dataset        : --dataset fairface | utkface | synthetic
  - Architecture   : --model mlp | cnn | resnet18
  - Activation     : --activation relu | tanh | gelu | leaky_relu | silu
  - Normalization  : --norm none | batchnorm | layernorm
  - Initialization : --init default | kaiming | xavier | zeros
  - Classification : --cls-loss cross_entropy | label_smoothing | focal | mse_onehot
  - Regression     : --reg-loss mse | mae | huber
  - Optimizer      : --optimizer sgd | sgd_momentum | adam | adamw | rmsprop
  - LR Scheduler   : --scheduler none | step | cosine
  - Augmentation   : --augment none | basic
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import (
    CachedFaceDataset,
    FairFaceDataset,
    MergedFaceDataset,
    SyntheticFaceDataset,
    UTKFaceDataset,
    get_dataset_meta,
    load_or_build_cache,
)
from losses import MultiTaskLoss
from models import MultiHeadFaceModel
from optimizers import build_optimizer, build_scheduler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modular Face Origin/Gender/Age Supervised Learning Workbench")
    p.add_argument("--run-name", type=str, default=None, help="Name for saving metrics in runs/<run-name>/")

    # 1. Data layer
    p.add_argument("--dataset", choices=["fairface", "utkface", "merged", "synthetic"], default="synthetic")
    p.add_argument("--data-dir", type=str, default="", help="Optional local path to UTKFace images (defaults to HF nu-delta/utkface)")
    p.add_argument("--fairface-config", choices=["0.25", "1.25"], default="0.25")
    p.add_argument("--age-mode", choices=["classification", "regression"], default=None, help="Age head mode for merged/synthetic dataset")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--augment", choices=["none", "basic"], default="none")
    p.add_argument("--max-samples", type=int, default=None, help="Subsample dataset for quick experiments")

    # 2. Model layer
    p.add_argument("--model", choices=["mlp", "cnn", "resnet18"], default="cnn")
    p.add_argument("--activation", choices=["relu", "tanh", "gelu", "leaky_relu", "silu"], default="relu")
    p.add_argument("--norm", choices=["none", "batchnorm", "layernorm"], default="batchnorm")
    p.add_argument("--init", choices=["default", "kaiming", "xavier", "zeros"], default="default")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-pretrained", action="store_true", help="Train resnet18 from scratch instead of pretrained")

    # 3. Loss layer
    p.add_argument("--cls-loss", choices=["cross_entropy", "label_smoothing", "focal", "mse_onehot"], default="cross_entropy")
    p.add_argument("--reg-loss", choices=["mse", "mae", "huber"], default="huber")

    # 4. Optimizer & training loop layer
    p.add_argument("--optimizer", choices=["sgd", "sgd_momentum", "adam", "adamw", "rmsprop"], default="adamw")
    p.add_argument("--scheduler", choices=["none", "step", "cosine"], default="cosine")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-dir", type=str, default="", help="Optional dir to cache decoded uint8 tensors once for 15x faster multi-epoch runs")
    p.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast on CUDA")
    p.add_argument("--no-save-checkpoint", action="store_true", help="Skip saving model.pt")
    return p.parse_args()


def build_datasets(args: argparse.Namespace, meta: dict):
    if args.dataset == "fairface":
        train_ds = FairFaceDataset(
            split="train", config=args.fairface_config, img_size=args.img_size,
            augment=args.augment, max_samples=args.max_samples,
        )
        val_ds = FairFaceDataset(
            split="validation", config=args.fairface_config, img_size=args.img_size,
            augment="none", max_samples=(args.max_samples // 4 if args.max_samples else None),
        )
    elif args.dataset == "utkface":
        train_ds = UTKFaceDataset(
            root_dir=args.data_dir, split="train", img_size=args.img_size,
            augment=args.augment, age_mode=meta["age_mode"], max_samples=args.max_samples, seed=args.seed,
        )
        val_ds = UTKFaceDataset(
            root_dir=args.data_dir, split="val", img_size=args.img_size,
            augment="none", age_mode=meta["age_mode"], max_samples=args.max_samples, seed=args.seed,
        )
    elif args.dataset == "merged":
        train_ds = MergedFaceDataset(
            split="train", fairface_config=args.fairface_config, utkface_dir=args.data_dir,
            img_size=args.img_size, augment=args.augment, age_mode=meta["age_mode"],
            max_samples=args.max_samples, seed=args.seed,
        )
        val_ds = MergedFaceDataset(
            split="validation", fairface_config=args.fairface_config, utkface_dir=args.data_dir,
            img_size=args.img_size, augment="none", age_mode=meta["age_mode"],
            max_samples=(args.max_samples // 4 if args.max_samples else None), seed=args.seed,
        )
    else:
        n_train = args.max_samples or 800
        n_val = max(128, n_train // 4)
        train_ds = SyntheticFaceDataset(n_train, args.img_size, meta["age_mode"], seed=args.seed)
        val_ds = SyntheticFaceDataset(n_val, args.img_size, meta["age_mode"], seed=args.seed + 1)
    return train_ds, val_ds


def compute_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


class FastTensorLoader:
    """
    Vectorized batch slicer for CachedFaceDataset.
    Avoids per-sample Python __getitem__ overhead (~10x faster epochs on GPU!).
    """

    def __init__(self, ds: CachedFaceDataset, batch_size: int, shuffle: bool, device: torch.device):
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.n = len(ds)

    def __iter__(self):
        perm = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for start in range(0, self.n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            x = self.ds.images_u8[idx].to(self.device, non_blocking=True).float().mul_(1.0 / 127.5).sub_(1.0)
            if self.ds.augment == "basic":
                flip_mask = torch.rand(x.size(0), device=self.device) < 0.5
                x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
            yield {
                "image": x,
                "origin": self.ds.origins[idx].to(self.device, non_blocking=True),
                "gender": self.ds.genders[idx].to(self.device, non_blocking=True),
                "age": self.ds.ages[idx].to(self.device, non_blocking=True),
            }

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size


@torch.no_grad()
def evaluate(model, loader, criterion, device, meta: dict) -> dict:
    model.eval()
    total_loss = 0.0
    n_samples = 0

    origin_correct = 0
    gender_correct = 0
    age_correct = 0
    age_abs_err = 0.0
    age_sq_err = 0.0

    # Per-origin-group accuracy tracking (crucial for fairness / balance analysis!)
    num_origins = meta["num_origins"]
    per_origin_correct = [0] * num_origins
    per_origin_total = [0] * num_origins

    for batch in loader:
        x = batch["image"].to(device)
        targets = {
            "origin": batch["origin"].to(device),
            "gender": batch["gender"].to(device),
            "age": batch["age"].to(device),
        }
        preds = model(x)
        losses = criterion(preds, targets)

        bs = x.size(0)
        total_loss += losses["total"].item() * bs
        n_samples += bs

        pred_o = preds["origin"].argmax(dim=-1)
        pred_g = preds["gender"].argmax(dim=-1)
        origin_correct += (pred_o == targets["origin"]).sum().item()
        gender_correct += (pred_g == targets["gender"]).sum().item()

        for c in range(num_origins):
            mask = targets["origin"] == c
            cnt = mask.sum().item()
            if cnt > 0:
                per_origin_total[c] += cnt
                per_origin_correct[c] += (pred_o[mask] == c).sum().item()

        if meta["age_mode"] == "classification":
            pred_a = preds["age"].argmax(dim=-1)
            age_correct += (pred_a == targets["age"]).sum().item()
        else:
            diff = preds["age"].squeeze(-1) - targets["age"].float()
            age_abs_err += diff.abs().sum().item()
            age_sq_err += (diff ** 2).sum().item()

    per_origin_acc = {
        meta["origin_names"][c]: round(per_origin_correct[c] / per_origin_total[c], 4)
        if per_origin_total[c] > 0 else None
        for c in range(num_origins)
    }

    metrics = {
        "val_loss": round(total_loss / max(1, n_samples), 4),
        "origin_acc": round(origin_correct / max(1, n_samples), 4),
        "gender_acc": round(gender_correct / max(1, n_samples), 4),
        "per_origin_acc": per_origin_acc,
    }
    if meta["age_mode"] == "classification":
        metrics["age_acc"] = round(age_correct / max(1, n_samples), 4)
    else:
        metrics["age_mae_years"] = round(age_abs_err / max(1, n_samples), 3)
        metrics["age_rmse_years"] = round((age_sq_err / max(1, n_samples)) ** 0.5, 3)
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = get_dataset_meta(args.dataset, args.age_mode)

    run_name = args.run_name or (
        f"{args.dataset}_{args.model}_{args.activation}_{args.norm}_"
        f"{args.cls_loss}_{args.optimizer}_lr{args.lr}"
    )

    print(f"=== Run: {run_name} | Device: {device} ===")
    if args.cache_dir and args.dataset != "synthetic":
        tag = f"{args.dataset}_{meta['age_mode']}_{args.img_size}_s{args.seed}_n{args.max_samples or 'all'}"
        cdir = Path(args.cache_dir)
        train_pt, val_pt = cdir / f"{tag}_train.pt", cdir / f"{tag}_val.pt"
        if train_pt.is_file() and val_pt.is_file():
            train_ds = load_or_build_cache(None, train_pt, augment=args.augment)
            val_ds = load_or_build_cache(None, val_pt, augment="none")
        else:
            raw_train, raw_val = build_datasets(args, meta)
            train_ds = load_or_build_cache(raw_train, train_pt, augment=args.augment, num_workers=args.num_workers)
            val_ds = load_or_build_cache(raw_val, val_pt, augment="none", num_workers=args.num_workers)
        train_loader = FastTensorLoader(train_ds, batch_size=args.batch_size, shuffle=True, device=device)
        val_loader = FastTensorLoader(val_ds, batch_size=args.batch_size, shuffle=False, device=device)
    else:
        train_ds, val_ds = build_datasets(args, meta)
        num_workers = 0 if args.dataset == "synthetic" else args.num_workers
        pin = torch.cuda.is_available()
        pw = num_workers > 0
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, persistent_workers=pw)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=pw)

    model = MultiHeadFaceModel(
        backbone_name=args.model,
        img_size=args.img_size,
        num_origins=meta["num_origins"],
        num_genders=meta["num_genders"],
        age_mode=meta["age_mode"],
        num_age_classes=meta["num_age_classes"],
        activation=args.activation,
        norm=args.norm,
        dropout=args.dropout,
        init_name=args.init,
        pretrained=not args.no_pretrained,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    use_amp = (device.type == "cuda") and (not args.no_amp) and torch.cuda.is_bf16_supported()
    print(f"Samples: train={len(train_ds)}, val={len(val_ds)} | Params: {n_params:,} | bf16_amp={use_amp}", flush=True)

    criterion = MultiTaskLoss(
        cls_loss_name=args.cls_loss,
        reg_loss_name=args.reg_loss,
        age_mode=meta["age_mode"],
    )
    optimizer = build_optimizer(model.parameters(), args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.scheduler, epochs=args.epochs)

    run_dir = Path(__file__).resolve().parent / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        grad_norm_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            targets = {
                "origin": batch["origin"].to(device),
                "gender": batch["gender"].to(device),
                "age": batch["age"].to(device),
            }

            # --- The exact 5-step supervised loop from micrograd ---
            optimizer.zero_grad(set_to_none=True)   # 1. Reset gradients
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                preds = model(x)                    # 2. Forward pass
                losses = criterion(preds, targets)  # 3. Compute loss
            losses["total"].backward()              # 4. Backpropagation
            grad_norm_sum += compute_grad_norm(model)
            optimizer.step()                        # 5. Update weights

            train_loss_sum += losses["total"].item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = round(train_loss_sum / max(1, n_batches), 4)
        avg_grad_norm = round(grad_norm_sum / max(1, n_batches), 4)
        val_metrics = evaluate(model, val_loader, criterion, device, meta)

        row = {
            "epoch": epoch,
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "train_loss": train_loss,
            "grad_norm": avg_grad_norm,
            **val_metrics,
        }
        history.append(row)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            if not args.no_save_checkpoint:
                torch.save(
                    {"model_state_dict": model.state_dict(), "config": vars(args), "meta": meta, "epoch": epoch},
                    run_dir / "best_model.pt",
                )

        age_str = (
            f"age_acc={val_metrics['age_acc']:.3f}"
            if meta["age_mode"] == "classification"
            else f"age_MAE={val_metrics['age_mae_years']:.2f}y"
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['val_loss']:.4f} | "
            f"origin={val_metrics['origin_acc']:.3f} gender={val_metrics['gender_acc']:.3f} {age_str} | "
            f"grad_norm={avg_grad_norm:.3f}",
            flush=True,
        )

    elapsed = round(time.time() - t0, 2)

    summary = {
        "run_name": run_name,
        "device": str(device),
        "params": n_params,
        "elapsed_sec": elapsed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "config": vars(args),
        "final": history[-1],
        "history": history,
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_save_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "meta": meta,
            },
            run_dir / "model.pt",
        )

    print(f"Done in {elapsed}s -> saved {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
