"""HSIANG 的遮擋、局部平滑與對照條件。"""

from pathlib import Path

from data.manifest import Sample
from models.base import FaceDetector


def run(model: FaceDetector, samples: list[Sample], output_dir: Path) -> None:
    # 同一位置、大小和遮擋方式要套用到所有模型與同一批影像。
    # 提供共用 ROI／Mask 給 DA；修改前後分數均使用同一模型介面。
    raise NotImplementedError("遮擋實驗尚待第二階段實作")

