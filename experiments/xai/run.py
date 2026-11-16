"""WEI 的歸因方法與圖像輸出。"""

from pathlib import Path

from data.manifest import Sample
from models.base import FaceDetector


def run(model: FaceDetector, samples: list[Sample], output_dir: Path) -> None:
    # 先記錄每張圖的 sample_id、模型、方法；不同模型可使用不同歸因器。
    # IG／Grad-CAM 走可微分的 forward_logits；LIME 走 score_fake。
    # DA 提供逐圖分數，YUZI 提供 FP／FN 樣本清單。
    raise NotImplementedError("XAI 實驗尚待第二階段實作")

