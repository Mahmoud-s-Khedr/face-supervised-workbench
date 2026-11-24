#!/usr/bin/env python3
"""
Inspect Dataset Balance & Label Distributions for FairFace and UTKFace.

Usage:
  python inspect_data.py --dataset fairface
  python inspect_data.py --dataset utkface --data-dir /path/to/UTKFace
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import (
    FAIRFACE_AGES,
    FAIRFACE_GENDERS,
    FAIRFACE_ORIGINS,
    FAIRFACE_TO_UNIFIED_ORIGIN,
    GENDER_STR_TO_ID,
    UNIFIED_ORIGINS,
    UTK_ETHNICITY_STR_TO_ID,
    UTKFACE_GENDERS,
    UTKFACE_ORIGINS,
    age_years_to_fairface_bin,
)


def print_distribution(title: str, counts: Counter, names: list[str]) -> None:
    total = sum(counts.values())
    vals = [counts.get(i, 0) for i in range(len(names))]
    non_zero = [v for v in vals if v > 0]
    ratio = (max(non_zero) / min(non_zero)) if len(non_zero) >= 2 else 1.0

    print(f"\n--- {title} (n={total:,} | max/min imbalance ratio = {ratio:.2f}x) ---")
    for i, name in enumerate(names):
        c = counts.get(i, 0)
        pct = 100.0 * c / max(1, total)
        bar = "#" * int(pct / 2)
        print(f"  {name:18s} {c:7,d} ({pct:5.1f}%) {bar}")


def inspect_fairface(config: str, out_path: Path) -> None:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/FairFace", config, split="train")
    origin_c = Counter(ds["race"])
    gender_c = Counter(ds["gender"])
    age_c = Counter(ds["age"])

    print(f"=== FairFace (config={config}, train split) ===")
    print_distribution("Origin / Race", origin_c, FAIRFACE_ORIGINS)
    print_distribution("Gender", gender_c, FAIRFACE_GENDERS)
    print_distribution("Age Groups", age_c, FAIRFACE_AGES)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].bar(FAIRFACE_ORIGINS, [origin_c[i] for i in range(len(FAIRFACE_ORIGINS))])
    axes[0].set_title("FairFace — Origin Balance")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(FAIRFACE_GENDERS, [gender_c[i] for i in range(len(FAIRFACE_GENDERS))])
    axes[1].set_title("FairFace — Gender Balance")

    axes[2].bar(FAIRFACE_AGES, [age_c[i] for i in range(len(FAIRFACE_AGES))])
    axes[2].set_title("FairFace — Age Group Balance")
    axes[2].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved chart -> {out_path}")


def inspect_utkface(data_dir: str, out_path: Path) -> None:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"UTKFace directory not found: {root}")

    ages, genders, origins = [], [], []
    for p in root.rglob("*.jpg"):
        parts = p.name.split("_")
        if len(parts) < 4:
            continue
        try:
            a, g, r = float(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if 0 <= a <= 120 and g in (0, 1) and 0 <= r < len(UTKFACE_ORIGINS):
            ages.append(a)
            genders.append(g)
            origins.append(r)

    origin_c = Counter(origins)
    gender_c = Counter(genders)

    print(f"=== UTKFace ({root}) ===")
    print_distribution("Origin / Race", origin_c, UTKFACE_ORIGINS)
    print_distribution("Gender", gender_c, UTKFACE_GENDERS)
    arr = np.array(ages)
    print(
        f"\n--- Continuous Age (n={len(arr):,}) ---\n"
        f"  min={arr.min():.0f}, median={np.median(arr):.1f}, mean={arr.mean():.1f}, "
        f"std={arr.std():.1f}, max={arr.max():.0f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].bar(UTKFACE_ORIGINS, [origin_c[i] for i in range(len(UTKFACE_ORIGINS))])
    axes[0].set_title("UTKFace — Origin Balance")

    axes[1].bar(UTKFACE_GENDERS, [gender_c[i] for i in range(len(UTKFACE_GENDERS))])
    axes[1].set_title("UTKFace — Gender Balance")

    axes[2].hist(arr, bins=30)
    axes[2].set_title("UTKFace — Continuous Age Distribution")
    axes[2].set_xlabel("Age (years)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved chart -> {out_path}")


def inspect_merged(config: str, out_path: Path) -> None:
    from datasets import load_dataset

    ff = load_dataset("HuggingFaceM4/FairFace", config, split="train")
    utk = load_dataset("nu-delta/utkface", split="train")

    origin_c: Counter = Counter()
    gender_c: Counter = Counter()
    age_c: Counter = Counter()

    for r in ff["race"]:
        origin_c[FAIRFACE_TO_UNIFIED_ORIGIN[int(r)]] += 1
    for g in ff["gender"]:
        gender_c[int(g)] += 1
    for a in ff["age"]:
        age_c[int(a)] += 1

    utk_kept = 0
    for a, g, e in zip(utk["age"], utk["gender"], utk["ethnicity"]):
        g_key = str(g).strip().lower()
        e_key = str(e).strip().lower()
        if a is not None and 0 <= float(a) <= 120 and g_key in GENDER_STR_TO_ID and e_key in UTK_ETHNICITY_STR_TO_ID:
            origin_c[UTK_ETHNICITY_STR_TO_ID[e_key]] += 1
            gender_c[GENDER_STR_TO_ID[g_key]] += 1
            age_c[age_years_to_fairface_bin(float(a))] += 1
            utk_kept += 1

    print(f"=== MERGED (FairFace train={len(ff):,} + UTKFace={utk_kept:,} => Total={len(ff) + utk_kept:,}) ===")
    print_distribution("Unified Origin (5 classes)", origin_c, UNIFIED_ORIGINS)
    print_distribution("Gender (2 classes)", gender_c, FAIRFACE_GENDERS)
    print_distribution("Unified Age Groups (9 bins)", age_c, FAIRFACE_AGES)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].bar(UNIFIED_ORIGINS, [origin_c[i] for i in range(len(UNIFIED_ORIGINS))])
    axes[0].set_title("Merged — Unified Origin Balance")

    axes[1].bar(FAIRFACE_GENDERS, [gender_c[i] for i in range(len(FAIRFACE_GENDERS))])
    axes[1].set_title("Merged — Gender Balance")

    axes[2].bar(FAIRFACE_AGES, [age_c[i] for i in range(len(FAIRFACE_AGES))])
    axes[2].set_title("Merged — Unified Age Group Balance")
    axes[2].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved chart -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fairface", "utkface", "merged"], required=True)
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--fairface-config", choices=["0.25", "1.25"], default="0.25")
    args = p.parse_args()

    out_dir = Path(__file__).resolve().parent / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "fairface":
        inspect_fairface(args.fairface_config, out_dir / "fairface_balance.png")
    elif args.dataset == "merged":
        inspect_merged(args.fairface_config, out_dir / "merged_balance.png")
    else:
        inspect_utkface(args.data_dir, out_dir / "utkface_balance.png")


if __name__ == "__main__":
    main()
