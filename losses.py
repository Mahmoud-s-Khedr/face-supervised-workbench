"""
Swappable Loss Functions for Supervised Learning Experiments.

Classification losses (Origin, Gender, FairFace Age Groups):
  - cross_entropy    : Standard -log(p_true). Best default for classification.
  - label_smoothing  : Cross-entropy with soft targets (reduces overconfidence).
  - focal            : Down-weights easy examples, focuses on hard/minority classes.
  - mse_onehot       : Mean Squared Error against one-hot targets (great to compare
                       against Cross-Entropy to see why CE converges faster!).

Regression losses (UTKFace Continuous Age & Facial Landmarks):
  - mse              : L2 loss (mean((y_hat - y)^2)) — penalizes large outliers heavily.
  - mae              : L1 loss (mean(|y_hat - y|)) — robust to outliers, constant grad.
  - huber            : Smooth L1 — quadratic near 0 (smooth), linear for big errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """FL(p_t) = -(1 - p_t)^gamma * log(p_t). Useful for imbalanced datasets (e.g. UTKFace)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


class MSEOneHotLoss(nn.Module):
    """
    MSE on softmax probabilities vs one-hot targets.
    Educational comparison: watch how gradients saturate compared to Cross-Entropy!
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        return F.mse_loss(probs, one_hot)


def build_classification_loss(name: str, label_smoothing: float = 0.1, focal_gamma: float = 2.0) -> nn.Module:
    name = name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma)
    if name == "mse_onehot":
        return MSEOneHotLoss()
    raise ValueError(f"Unknown classification loss '{name}'. Choose from: cross_entropy, label_smoothing, focal, mse_onehot")


def build_regression_loss(name: str, huber_delta: float = 5.0) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name in ("huber", "smooth_l1"):
        return nn.HuberLoss(delta=huber_delta)
    raise ValueError(f"Unknown regression loss '{name}'. Choose from: mse, mae, huber")


class MultiTaskLoss(nn.Module):
    """
    Combines Origin + Gender + Age (+ optional Landmarks) losses into one scalar
    for backpropagation:
        total = w_origin * L_origin + w_gender * L_gender + w_age * L_age (+ w_lm * L_lm)
    """

    def __init__(
        self,
        cls_loss_name: str = "cross_entropy",
        reg_loss_name: str = "huber",
        age_mode: str = "classification",  # "classification" (FairFace) or "regression" (UTKFace)
        weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.age_mode = age_mode
        self.origin_loss = build_classification_loss(cls_loss_name)
        self.gender_loss = build_classification_loss(cls_loss_name)
        self.age_loss = (
            build_classification_loss(cls_loss_name)
            if age_mode == "classification"
            else build_regression_loss(reg_loss_name)
        )
        self.landmark_loss = build_regression_loss(reg_loss_name)
        self.weights = weights or {"origin": 1.0, "gender": 1.0, "age": 1.0, "landmarks": 1.0}

    def forward(self, preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        l_origin = self.origin_loss(preds["origin"], targets["origin"])
        l_gender = self.gender_loss(preds["gender"], targets["gender"])

        if self.age_mode == "classification":
            l_age = self.age_loss(preds["age"], targets["age"].long())
        else:
            l_age = self.age_loss(preds["age"].squeeze(-1), targets["age"].float())

        total = (
            self.weights.get("origin", 1.0) * l_origin
            + self.weights.get("gender", 1.0) * l_gender
            + self.weights.get("age", 1.0) * l_age
        )

        out = {
            "total": total,
            "origin": l_origin.detach(),
            "gender": l_gender.detach(),
            "age": l_age.detach(),
        }

        if "landmarks" in preds and "landmarks" in targets and targets["landmarks"] is not None:
            l_lm = self.landmark_loss(preds["landmarks"], targets["landmarks"].float())
            out["total"] = out["total"] + self.weights.get("landmarks", 1.0) * l_lm
            out["landmarks"] = l_lm.detach()

        return out
