"""只用來測試流程的假模型，不能當成研究結果。"""

import numpy as np

from models.base import FaceDetector


class DemoBrightnessDetector(FaceDetector):
    name = "demo"
    threshold = 0.5

    def score_fake(self, image: np.ndarray) -> float:
        # 用平均亮度當分數，方便檢查資料和結果能否從頭跑到尾。
        return float(image.mean() / 255.0)

