"""
Swappable Optimizers and Learning Rate Schedulers.

Optimizers:
  - sgd          : Vanilla gradient descent (same math as micrograd!)
  - sgd_momentum : SGD + velocity (smooths noisy mini-batch gradients)
  - adam         : Adaptive per-parameter learning rates (1st + 2nd moments)
  - adamw        : Adam with decoupled weight decay (modern default)
  - rmsprop      : Adaptive learning rate using moving average of squared grads

Schedulers:
  - none         : Constant learning rate
  - step         : Halves LR every 1/3 of epochs
  - cosine       : Smooth cosine decay down to 1% of initial LR
"""

import torch
import torch.optim as optim


def build_optimizer(
    params,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Choose from: sgd, sgd_momentum, adam, adamw, rmsprop")


def build_scheduler(optimizer: optim.Optimizer, name: str, epochs: int):
    name = name.lower()
    if name == "none":
        return None
    if name == "step":
        step_size = max(1, epochs // 3)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    raise ValueError(f"Unknown scheduler '{name}'. Choose from: none, step, cosine")
