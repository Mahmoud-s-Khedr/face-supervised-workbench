"""
Swappable Datasets and Image Transforms (FairFace, UTKFace, and fast Synthetic smoke test).

Uses only PIL + NumPy + PyTorch for image preprocessing so it runs locally without
requiring torchvision, and seamlessly loads FairFace / UTKFace on Colab or locally.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

# Standard label mappings
FAIRFACE_ORIGINS = [
    "East Asian",
    "Indian",
    "Black",
    "White",
    "Middle Eastern",
    "Latino_Hispanic",
    "Southeast Asian",
]
FAIRFACE_GENDERS = ["Male", "Female"]
FAIRFACE_AGES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]

UTKFACE_ORIGINS = ["White", "Black", "Asian", "Indian", "Others"]
UTKFACE_GENDERS = ["Male", "Female"]

# Unified 5-class origin schema when merging FairFace (7) + UTKFace (5):
#   0: White  <- FairFace White(3), UTKFace White(0)
#   1: Black  <- FairFace Black(2), UTKFace Black(1)
#   2: Asian  <- FairFace East Asian(0) + Southeast Asian(6), UTKFace Asian(2)
#   3: Indian <- FairFace Indian(1), UTKFace Indian(3)
#   4: Others <- FairFace Middle Eastern(4) + Latino_Hispanic(5), UTKFace Others(4)
UNIFIED_ORIGINS = ["White", "Black", "Asian", "Indian", "Others"]
FAIRFACE_TO_UNIFIED_ORIGIN = {
    0: 2,  # East Asian -> Asian
    1: 3,  # Indian -> Indian
    2: 1,  # Black -> Black
    3: 0,  # White -> White
    4: 4,  # Middle Eastern -> Others
    5: 4,  # Latino_Hispanic -> Others
    6: 2,  # Southeast Asian -> Asian
}
FAIRFACE_AGE_MIDPOINTS = [1.0, 6.0, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75.0]

UTK_ETHNICITY_STR_TO_ID = {
    "white": 0,
    "black": 1,
    "asian": 2,
    "indian": 3,
    "others": 4,
    "other": 4,
}
GENDER_STR_TO_ID = {"male": 0, "female": 1, "0": 0, "1": 1}


def age_years_to_fairface_bin(age: float) -> int:
    """Map continuous age in years to FairFace's 9 discrete age bins."""
    if age <= 2:
        return 0
    if age <= 9:
        return 1
    if age <= 19:
        return 2
    if age <= 29:
        return 3
    if age <= 39:
        return 4
    if age <= 49:
        return 5
    if age <= 59:
        return 6
    if age <= 69:
        return 7
    return 8


def image_to_tensor(img: Image.Image, img_size: int = 64, augment: str = "none") -> torch.Tensor:
    """Resize PIL image to (3, img_size, img_size) normalized to [-1, 1]."""
    img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)

    if augment == "basic":
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

    arr = np.asarray(img, dtype=np.float32) / 255.0  # [0, 1]
    arr = (arr - 0.5) / 0.5  # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)


class FairFaceDataset(Dataset):
    """
    Loads HuggingFaceM4/FairFace via Hugging Face `datasets`.
    Tasks:
      - origin : 7-class classification
      - gender : 2-class classification
      - age    : 9-class discrete age group classification
    """

    def __init__(
        self,
        split: str = "train",
        config: str = "0.25",
        img_size: int = 64,
        augment: str = "none",
        max_samples: int | None = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "Hugging Face `datasets` is required for FairFace. "
                "Install on Colab or run: uv pip install datasets"
            ) from e

        hf_split = "train" if split == "train" else "validation"
        ds = load_dataset("HuggingFaceM4/FairFace", config, split=hf_split)
        if max_samples is not None and max_samples < len(ds):
            ds = ds.shuffle(seed=42).select(range(max_samples))
        self.ds = ds
        self.img_size = img_size
        self.augment = augment if split == "train" else "none"

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.ds[idx]
        x = image_to_tensor(row["image"], self.img_size, self.augment)
        return {
            "image": x,
            "origin": torch.tensor(int(row["race"]), dtype=torch.long),
            "gender": torch.tensor(int(row["gender"]), dtype=torch.long),
            "age": torch.tensor(int(row["age"]), dtype=torch.long),
        }


class UTKFaceDataset(Dataset):
    """
    Loads UTKFace either from:
      - Hugging Face `nu-delta/utkface` (default when root_dir is empty), OR
      - A local directory of images named `[age]_[gender]_[race]_[date&time].jpg`.
    Tasks:
      - origin : 5-class classification
      - gender : 2-class classification
      - age    : continuous regression (years) or 9-bin classification (age_mode='classification')
    """

    def __init__(
        self,
        root_dir: str | Path = "",
        split: str = "train",
        img_size: int = 64,
        augment: str = "none",
        age_mode: str = "regression",
        val_ratio: float = 0.2,
        max_samples: int | None = None,
        seed: int = 42,
        hf_repo: str = "nu-delta/utkface",
    ):
        self.img_size = img_size
        self.augment = augment if split == "train" else "none"
        self.age_mode = age_mode
        self.use_hf = not (root_dir and Path(root_dir).is_dir())

        if self.use_hf:
            try:
                from datasets import load_dataset
            except ImportError as e:
                raise ImportError(
                    "Hugging Face `datasets` is required to auto-load UTKFace. "
                    "Install on Colab or pass --data-dir /path/to/local/UTKFace"
                ) from e

            ds = load_dataset(hf_repo, split="train")
            valid_indices: list[int] = []
            # Filter any malformed rows (e.g., missing ethnicity/gender)
            genders = ds["gender"]
            eths = ds["ethnicity"]
            ages = ds["age"]
            for i in range(len(ds)):
                g_key = str(genders[i]).strip().lower()
                e_key = str(eths[i]).strip().lower()
                a_val = ages[i]
                if a_val is not None and 0 <= float(a_val) <= 120 and g_key in GENDER_STR_TO_ID and e_key in UTK_ETHNICITY_STR_TO_ID:
                    valid_indices.append(i)

            rng = random.Random(seed)
            rng.shuffle(valid_indices)
            if max_samples is not None and max_samples < len(valid_indices):
                valid_indices = valid_indices[:max_samples]

            split_idx = int(len(valid_indices) * (1.0 - val_ratio))
            chosen = valid_indices[:split_idx] if split == "train" else valid_indices[split_idx:]
            self.ds = ds.select(chosen)
            self.records = []
        else:
            root = Path(root_dir)
            records: list[tuple[Path, float, int, int]] = []
            for p in sorted(root.rglob("*.jpg")):
                parts = p.name.split("_")
                if len(parts) < 4:
                    continue
                try:
                    age = float(parts[0])
                    gender = int(parts[1])
                    race = int(parts[2])
                except ValueError:
                    continue
                if 0 <= age <= 120 and gender in (0, 1) and 0 <= race < len(UTKFACE_ORIGINS):
                    records.append((p, age, gender, race))

            if not records:
                raise RuntimeError(f"No valid UTKFace [age]_[gender]_[race]_*.jpg files found in {root}")

            rng = random.Random(seed)
            rng.shuffle(records)
            if max_samples is not None and max_samples < len(records):
                records = records[:max_samples]

            split_idx = int(len(records) * (1.0 - val_ratio))
            self.records = records[:split_idx] if split == "train" else records[split_idx:]
            self.ds = None

    def __len__(self) -> int:
        return len(self.ds) if self.use_hf else len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.use_hf:
            row = self.ds[idx]
            x = image_to_tensor(row["image"], self.img_size, self.augment)
            age_val = float(row["age"])
            gender = GENDER_STR_TO_ID[str(row["gender"]).strip().lower()]
            race = UTK_ETHNICITY_STR_TO_ID[str(row["ethnicity"]).strip().lower()]
        else:
            path, age_val, gender, race = self.records[idx]
            with Image.open(path) as img:
                x = image_to_tensor(img, self.img_size, self.augment)

        if self.age_mode == "classification":
            age_t = torch.tensor(age_years_to_fairface_bin(age_val), dtype=torch.long)
        else:
            age_t = torch.tensor(age_val, dtype=torch.float32)

        return {
            "image": x,
            "origin": torch.tensor(race, dtype=torch.long),
            "gender": torch.tensor(gender, dtype=torch.long),
            "age": age_t,
        }


class MergedFaceDataset(Dataset):
    """
    Merges ALL of FairFace + ALL of UTKFace into one unified dataset (~105.7k train / ~15.7k val):
      - origin : Unified 5 classes [White, Black, Asian, Indian, Others]
      - gender : 2 classes [Male, Female]
      - age    : 9 unified FairFace bins (age_mode='classification') OR
                 continuous years (age_mode='regression': exact UTKFace years + FairFace bin midpoints)
    """

    def __init__(
        self,
        split: str = "train",
        fairface_config: str = "0.25",
        utkface_dir: str | Path = "",
        img_size: int = 64,
        augment: str = "none",
        age_mode: str = "classification",
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.age_mode = age_mode
        self.fairface = FairFaceDataset(
            split=split,
            config=fairface_config,
            img_size=img_size,
            augment=augment,
            max_samples=None,
        )
        self.utkface = UTKFaceDataset(
            root_dir=utkface_dir,
            split=split,
            img_size=img_size,
            augment=augment,
            age_mode=age_mode,
            max_samples=None,
            seed=seed,
        )

        # Build index table: (source_id: 0=FairFace, 1=UTKFace, local_idx)
        index: list[tuple[int, int]] = (
            [(0, i) for i in range(len(self.fairface))]
            + [(1, i) for i in range(len(self.utkface))]
        )
        if max_samples is not None and max_samples < len(index):
            rng = random.Random(seed)
            rng.shuffle(index)
            index = index[:max_samples]
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        src, local_idx = self.index[idx]
        if src == 0:
            item = self.fairface[local_idx]
            ff_origin = int(item["origin"].item())
            item["origin"] = torch.tensor(FAIRFACE_TO_UNIFIED_ORIGIN[ff_origin], dtype=torch.long)
            if self.age_mode == "regression":
                age_bin = int(item["age"].item())
                item["age"] = torch.tensor(FAIRFACE_AGE_MIDPOINTS[age_bin], dtype=torch.float32)
            return item
        return self.utkface[local_idx]


class CachedFaceDataset(Dataset):
    """
    Pre-decoded uint8 tensor dataset held in RAM for 10-20x faster multi-epoch training.
    105k x 3 x 96 x 96 uint8 = ~2.9 GB RAM (easily fits in Colab L4's 53 GB RAM!).
    """

    def __init__(self, images_u8: torch.Tensor, origins: torch.Tensor, genders: torch.Tensor, ages: torch.Tensor, augment: str = "none"):
        self.images_u8 = images_u8
        self.origins = origins
        self.genders = genders
        self.ages = ages
        self.augment = augment

    def __len__(self) -> int:
        return self.images_u8.size(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x = self.images_u8[idx].float().mul_(1.0 / 127.5).sub_(1.0)  # [-1, 1]
        if self.augment == "basic" and random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal flip
        return {
            "image": x,
            "origin": self.origins[idx],
            "gender": self.genders[idx],
            "age": self.ages[idx],
        }


def load_or_build_cache(ds: Dataset, cache_path: Path, augment: str = "none", num_workers: int = 8) -> CachedFaceDataset:
    """Decode & resize a dataset once into a compact uint8 .pt file; reload in <1s on future runs."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        return CachedFaceDataset(blob["images"], blob["origin"], blob["gender"], blob["age"], augment=augment)

    from torch.utils.data import DataLoader

    print(f"Building RAM/disk uint8 cache ({len(ds):,} samples) -> {cache_path} ...", flush=True)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=num_workers)
    imgs, origins, genders, ages = [], [], [], []
    for batch in loader:
        u8 = ((batch["image"] * 0.5 + 0.5) * 255.0).clamp(0, 255).to(torch.uint8)
        imgs.append(u8)
        origins.append(batch["origin"])
        genders.append(batch["gender"])
        ages.append(batch["age"])

    blob = {
        "images": torch.cat(imgs, dim=0),
        "origin": torch.cat(origins, dim=0),
        "gender": torch.cat(genders, dim=0),
        "age": torch.cat(ages, dim=0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, cache_path)
    gb = blob["images"].numel() / (1024 ** 3)
    print(f"Cached {len(blob['images']):,} images ({gb:.2f} GB uint8) -> {cache_path}", flush=True)
    return CachedFaceDataset(blob["images"], blob["origin"], blob["gender"], blob["age"], augment=augment)


class SyntheticFaceDataset(Dataset):
    """
    Fast, self-contained synthetic dataset for instant local CPU experiments.
    Encodes learnable visual patterns (color tint, horizontal/vertical stripes, brightness)
    so you can test losses, optimizers, activations, normalizations, and init in seconds!
    """

    def __init__(
        self,
        num_samples: int = 800,
        img_size: int = 64,
        age_mode: str = "classification",
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.age_mode = age_mode

        g = torch.Generator().manual_seed(seed)
        self.origins = torch.randint(0, 7, (num_samples,), generator=g)
        self.genders = torch.randint(0, 2, (num_samples,), generator=g)
        if age_mode == "classification":
            self.ages = torch.randint(0, 9, (num_samples,), generator=g)
        else:
            self.ages = torch.rand((num_samples,), generator=g) * 80.0 + 5.0  # 5..85 yrs

        # Fixed class color prototypes (shared across train/val splits) so a neural net can generalize
        proto_g = torch.Generator().manual_seed(0)
        self.origin_colors = torch.randn((7, 3, 1, 1), generator=proto_g) * 0.5
        self.noise = torch.randn((num_samples, 3, img_size, img_size), generator=g) * 0.35

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        o = int(self.origins[idx])
        g = int(self.genders[idx])
        a = self.ages[idx]

        img = self.noise[idx].clone()
        # Origin -> RGB tint
        img += self.origin_colors[o]
        # Gender -> top vs bottom half brightness shift
        half = self.img_size // 2
        if g == 0:
            img[:, :half, :] += 0.4
        else:
            img[:, half:, :] += 0.4
        # Age -> center patch intensity
        q1, q3 = self.img_size // 4, (3 * self.img_size) // 4
        age_norm = (float(a) / 8.0) if self.age_mode == "classification" else (float(a) / 85.0)
        img[:, q1:q3, q1:q3] += (age_norm - 0.5) * 1.2

        return {
            "image": img.clamp(-1.0, 1.0),
            "origin": self.origins[idx],
            "gender": self.genders[idx],
            "age": a,
        }


def get_dataset_meta(dataset_name: str, age_mode: str | None = None) -> dict[str, Any]:
    dataset_name = dataset_name.lower()
    if dataset_name == "fairface":
        return {
            "origin_names": FAIRFACE_ORIGINS,
            "gender_names": FAIRFACE_GENDERS,
            "age_names": FAIRFACE_AGES,
            "num_origins": len(FAIRFACE_ORIGINS),
            "num_genders": len(FAIRFACE_GENDERS),
            "num_age_classes": len(FAIRFACE_AGES),
            "age_mode": "classification",
        }
    if dataset_name == "utkface":
        return {
            "origin_names": UTKFACE_ORIGINS,
            "gender_names": UTKFACE_GENDERS,
            "age_names": None,
            "num_origins": len(UTKFACE_ORIGINS),
            "num_genders": len(UTKFACE_GENDERS),
            "num_age_classes": 1,
            "age_mode": "regression",
        }
    if dataset_name == "merged":
        mode = age_mode or "classification"
        return {
            "origin_names": UNIFIED_ORIGINS,
            "gender_names": FAIRFACE_GENDERS,
            "age_names": FAIRFACE_AGES if mode == "classification" else None,
            "num_origins": len(UNIFIED_ORIGINS),
            "num_genders": len(FAIRFACE_GENDERS),
            "num_age_classes": len(FAIRFACE_AGES) if mode == "classification" else 1,
            "age_mode": mode,
        }
    if dataset_name == "synthetic":
        mode = age_mode or "classification"
        return {
            "origin_names": FAIRFACE_ORIGINS,
            "gender_names": FAIRFACE_GENDERS,
            "age_names": FAIRFACE_AGES if mode == "classification" else None,
            "num_origins": len(FAIRFACE_ORIGINS),
            "num_genders": len(FAIRFACE_GENDERS),
            "num_age_classes": len(FAIRFACE_AGES) if mode == "classification" else 1,
            "age_mode": mode,
        }
    raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: fairface, utkface, merged, synthetic")
