"""B：在這裡實作模型適用的歸因方法與圖像輸出。"""

from pathlib import Path

from data.manifest import Sample
from models.base import FaceDetector


def run(model: FaceDetector, samples: list[Sample], output_dir: Path) -> None:
    # 先記錄每張圖的 sample_id、模型、方法；不同模型可使用不同歸因器。
    # 若要跨模型比較，需說明方法差異，並確保圖像尺寸與色階的呈現一致。
    raise NotImplementedError("XAI 實驗尚待 B 實作")

