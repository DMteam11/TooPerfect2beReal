"""模型介面與分數方向，見 INTERFACE_CONTRACT.md。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor, nn


class FaceDetector(ABC):
    name: str
    threshold: float = 0.5

    @abstractmethod
    def score_fake(self, image: np.ndarray) -> float:
        """輸入 RGB uint8 影像，回傳 0~1 分數；越高越偏向生成影像。"""

    def predict_label(self, image: np.ndarray) -> int:
        return self.label_from_score(self.score_fake(image))

    def label_from_score(self, score: float) -> int:
        """對已算出的分數套用同一門檻，省去重跑推論。"""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{self.name} 的 score_fake 必須在 0~1:{score}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"{self.name} 的 threshold 必須在 0~1:{self.threshold}")
        return int(score >= self.threshold)


class ExplainableFaceDetector(FaceDetector):
    """三個正式模型的基底，讓效能與 XAI 使用同一組權重和前處理。"""

    weights_id: str
    preprocessing: str
    threshold_source: str

    @abstractmethod
    def preprocess_tensor(self, image: np.ndarray) -> Tensor:
        """將共用 RGB 影像轉成模型裝置上的浮點張量，形狀為 (1, 3, H, W)。"""

    @abstractmethod
    def forward_logits(self, tensor: Tensor) -> Tensor:
        """回傳 (N,) 的 fake logit。呼叫端須能對輸入和中間層求梯度。"""

    @abstractmethod
    def get_target_layers(self) -> Mapping[str, nn.Module]:
        """回傳同一模型上可設 hook 的候選層。分析者再選擇適合方法的層。"""

    def score_fake(self, image: np.ndarray) -> float:
        # 只有取分數時關閉梯度，IG、Grad-CAM、TCAV 直接呼叫 forward_logits。
        import torch

        with torch.no_grad():
            logits = self.forward_logits(self.preprocess_tensor(image))
            return float(logits.sigmoid().item())

