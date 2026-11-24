"""
Swappable Model Architectures, Activations, Normalization, and Weight Initializations.

Compare:
  - Backbones     : mlp (micrograd-style flattened net), cnn (4-block conv net), resnet18
  - Activations   : relu, tanh, gelu, leaky_relu, silu
  - Normalization : none, batchnorm, layernorm
  - Initialization: default, kaiming, xavier, zeros (watch symmetry breaking fail!)
  - Regularization: dropout (0.0 .. 0.5)
"""

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation '{name}'. Choose from: relu, tanh, gelu, leaky_relu, silu")


def get_norm1d(norm: str, num_features: int) -> nn.Module:
    norm = norm.lower()
    if norm == "none":
        return nn.Identity()
    if norm == "batchnorm":
        return nn.BatchNorm1d(num_features)
    if norm == "layernorm":
        return nn.LayerNorm(num_features)
    raise ValueError(f"Unknown norm '{norm}'. Choose from: none, batchnorm, layernorm")


def get_norm2d(norm: str, num_channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == "none":
        return nn.Identity()
    if norm == "batchnorm":
        return nn.BatchNorm2d(num_channels)
    if norm == "layernorm":
        # GroupNorm(1, C) is channel-wise LayerNorm for 2D feature maps
        return nn.GroupNorm(1, num_channels)
    raise ValueError(f"Unknown norm '{norm}'. Choose from: none, batchnorm, layernorm")


def apply_weight_init(model: nn.Module, init_name: str) -> None:
    """Apply a chosen weight initialization strategy to Linear and Conv2d layers."""
    init_name = init_name.lower()
    if init_name == "default":
        return

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if init_name == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif init_name == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init_name == "zeros":
                # Classic experiment: all weights = 0 -> symmetry never breaks!
                nn.init.zeros_(m.weight)
            else:
                raise ValueError(f"Unknown init '{init_name}'. Choose from: default, kaiming, xavier, zeros")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class MLPBackbone(nn.Module):
    """
    Direct PyTorch cousin of micrograd's MLP: flattens the image (C * H * W)
    and passes it through fully-connected layers.
    """

    def __init__(
        self,
        img_size: int = 64,
        hidden_dims: tuple[int, ...] = (512, 256),
        activation: str = "relu",
        norm: str = "batchnorm",
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = 3 * img_size * img_size
        layers: list[nn.Module] = [nn.Flatten()]
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(get_norm1d(norm, h))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNBackbone(nn.Module):
    """
    Simple 4-block Convolutional Neural Network built from scratch.
    Preserves 2D spatial structure (edges -> eyes/nose -> face) with far fewer params than MLP.
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        activation: str = "relu",
        norm: str = "batchnorm",
        dropout: float = 0.2,
    ):
        super().__init__()
        in_ch = 3
        blocks: list[nn.Module] = []
        for out_ch in channels:
            blocks.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=(norm != "batchnorm")),
                get_norm2d(norm, out_ch),
                get_activation(activation),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_dim = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.drop(x)


class ResNet18Backbone(nn.Module):
    """Optional pretrained (or scratch) ResNet18 backbone via torchvision."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as e:
            raise ImportError(
                "torchvision is required for --model resnet18. "
                "Install it on Colab or run: uv pip install torchvision"
            ) from e
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        self.out_dim = net.fc.in_features
        net.fc = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadFaceModel(nn.Module):
    """
    Shared feature extractor + separate task heads:
      - origin_head   : num_origins logits
      - gender_head   : 2 logits
      - age_head      : num_age_classes logits (FairFace) OR 1 scalar (UTKFace regression)
      - landmark_head : optional 136 scalars (68 x,y facial landmarks for UTKFace)
    """

    def __init__(
        self,
        backbone_name: str = "cnn",
        img_size: int = 64,
        num_origins: int = 7,
        num_genders: int = 2,
        age_mode: str = "classification",
        num_age_classes: int = 9,
        predict_landmarks: bool = False,
        activation: str = "relu",
        norm: str = "batchnorm",
        dropout: float = 0.2,
        init_name: str = "default",
        pretrained: bool = True,
    ):
        super().__init__()
        backbone_name = backbone_name.lower()
        if backbone_name == "mlp":
            self.backbone = MLPBackbone(
                img_size=img_size, activation=activation, norm=norm, dropout=dropout
            )
        elif backbone_name == "cnn":
            self.backbone = CNNBackbone(
                activation=activation, norm=norm, dropout=dropout
            )
        elif backbone_name == "resnet18":
            self.backbone = ResNet18Backbone(pretrained=pretrained, dropout=dropout)
        else:
            raise ValueError(f"Unknown model '{backbone_name}'. Choose from: mlp, cnn, resnet18")

        d = self.backbone.out_dim
        self.origin_head = nn.Linear(d, num_origins)
        self.gender_head = nn.Linear(d, num_genders)
        self.age_head = nn.Linear(d, num_age_classes if age_mode == "classification" else 1)
        self.landmark_head = nn.Linear(d, 136) if predict_landmarks else None

        # Only override init for scratch models (never wipe pretrained ResNet18 weights!)
        if not (backbone_name == "resnet18" and pretrained):
            apply_weight_init(self, init_name)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        out = {
            "origin": self.origin_head(feat),
            "gender": self.gender_head(feat),
            "age": self.age_head(feat),
        }
        if self.landmark_head is not None:
            out["landmarks"] = self.landmark_head(feat)
        return out
