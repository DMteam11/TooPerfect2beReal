"""UnivFD：CLIP ViT-L/14 backbone + 線性分類頭。前處理依官方 validate.py（CenterCrop 224，不 resize）。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn

from models.base import ExplainableFaceDetector

CLIP_NAME = "ViT-L/14"
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
# 官方 clip.py 下載網址中的 sha256，clip.load 會自行校驗。
CLIP_BACKBONE_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"


class UnivFDDetector(ExplainableFaceDetector):
    name = "univfd"
    threshold = 0.5
    threshold_source = "官方 validate.py：sigmoid(fake logit) 以 0.5 判斷真假"
    preprocessing = "官方 validate.py：CenterCrop(224)，不 resize；ToTensor 後以 CLIP mean/std 正規化"

    def __init__(self, weights_path: Path, device: str) -> None:
        import clip

        head_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        self.weights_id = f"univfd-head-{head_sha[:12]}-clipL14-{CLIP_BACKBONE_SHA256[:12]}"

        # 依官方做法：先在 CPU 載入並轉成 fp32，再搬到目標裝置。
        backbone, _ = clip.load(CLIP_NAME, device="cpu", jit=False)
        self.clip_model = backbone.float()
        self.fc = nn.Linear(768, 1)
        self.fc.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

        self.device = torch.device(device)
        self.clip_model.requires_grad_(False).eval().to(self.device)
        self.fc.requires_grad_(False).eval().to(self.device)

        self._crop = transforms.CenterCrop(224)
        self._normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def preprocess_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        tensor = self._normalize(self._crop(tensor))
        return tensor.unsqueeze(0).to(self.device)

    def forward_logits(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.fc(self.clip_model.encode_image(tensor)).squeeze(-1)

    def get_target_layers(self) -> Mapping[str, nn.Module]:
        visual = self.clip_model.visual
        layers: dict[str, nn.Module] = {
            f"visual.transformer.resblocks.{index}": block
            for index, block in enumerate(visual.transformer.resblocks)
        }
        layers["visual.ln_post"] = visual.ln_post
        return layers
