"""Synthetic Sight final ResNet-50 adapter; provenance in docs/WEI_PHASE1.md."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from models.base import ExplainableFaceDetector


OFFICIAL_SHA256 = "d9a7fd6a692c942b550f9848500dc3ffb10d5809cb0d0091990648bf369ad21c"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class SyntheticSightDetector(ExplainableFaceDetector):
    name = "syntheticsight"

    def __init__(self, *, weights_path: Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        weights_path = Path(weights_path)
        with weights_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        # Pin the audited final checkpoint. Legacy heads and unverified replacements
        # must not silently inherit the official threshold or label convention.
        if digest != OFFICIAL_SHA256:
            raise ValueError("SyntheticSight 權重 SHA256 不符官方 final checkpoint；請依 docs/WEI_PHASE1.md 下載")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        expected = {
            "head_configuration": {"input_features": 2048, "hidden_units": 256, "batch_normalization": True,
                                   "dropout": 0.3, "output_features": 1},
            "label_convention": {"Real": 0, "Fake": 1},
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("SyntheticSight checkpoint 的分類頭或真假標籤不相容")
        if tuple(checkpoint.get("image_size", ())) != (224, 224):
            raise ValueError("SyntheticSight checkpoint 輸入尺寸不相容")
        if (tuple(checkpoint.get("normalize_mean", MEAN)) != MEAN
                or tuple(checkpoint.get("normalize_std", STD)) != STD):
            raise ValueError("SyntheticSight checkpoint 正規化不相容")
        self.threshold = float(checkpoint["decision_threshold"])
        if self.threshold != 0.51:
            raise ValueError("SyntheticSight final checkpoint 門檻應為 0.51")
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 1),
        )
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.model.to(self.device).eval()
        except (RuntimeError, KeyError) as error:
            raise ValueError(f"SyntheticSight 載入失敗：{error}") from error
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
        self.weights_id = f"SyntheticSight final epoch={checkpoint.get('epoch')}; sha256:{digest}"
        self.threshold_source = "Official final checkpoint: validation Fake F1 selection; threshold=0.51"
        self.preprocessing = json.dumps({
            "input": "RGB uint8; shared loader handles EXIF", "resize": [224, 224],
            "interpolation": "PIL bilinear", "crop": None, "scale": "1/255",
            "mean": MEAN, "std": STD,
        })

    def preprocess_tensor(self, image: np.ndarray) -> torch.Tensor:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
            raise ValueError("SyntheticSight 輸入必須是非空 H×W×3 RGB uint8")
        return self.transform(Image.fromarray(image)).unsqueeze(0).to(self.device)

    def forward_logits(self, tensor: torch.Tensor) -> torch.Tensor:
        # Keep the graph intact for input gradients, IG and intermediate hooks.
        return self.model(tensor).reshape(-1)

    def get_target_layers(self) -> dict[str, nn.Module]:
        return {"layer3": self.model.layer3, "layer4": self.model.layer4}


def create_syntheticsight(*, weights_path: Path | None, device: str) -> SyntheticSightDetector:
    if weights_path is None:
        raise ValueError("SyntheticSight 需要官方 checkpoint")
    return SyntheticSightDetector(weights_path=weights_path, device=device)
