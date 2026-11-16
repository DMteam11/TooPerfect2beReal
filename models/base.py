"""所有模型共同遵守的最小介面。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FaceDetector(ABC):
    name: str
    threshold: float = 0.5

    @abstractmethod
    def score_fake(self, image: np.ndarray) -> float:
        """輸入 RGB uint8 影像，回傳 0～1 分數；越高越偏向生成影像。"""

    def predict_label(self, image: np.ndarray) -> int:
        score = self.score_fake(image)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{self.name} 的 score_fake 必須在 0～1：{score}")
        return int(score >= self.threshold)

