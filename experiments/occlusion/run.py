"""C：在這裡實作遮擋、局部平滑與對照條件。"""

from pathlib import Path

from data.manifest import Sample
from models.base import FaceDetector


def run(model: FaceDetector, samples: list[Sample], output_dir: Path) -> None:
    # 同一位置、大小和遮擋方式要套用到所有模型與同一批影像。
    # 輸出原始分數、擾動後分數與差值，讓 C 可直接橫跨模型比較。
    raise NotImplementedError("遮擋實驗尚待 C 實作")

