"""D：在這裡實作臉部特徵統計與可行的概念測試。"""

from pathlib import Path

from data.manifest import Sample
from models.base import FaceDetector


def run(model: FaceDetector, samples: list[Sample], output_dir: Path) -> None:
    # 先固定特徵定義和計算方式；概念測試另記樣本選取與控制組。
    raise NotImplementedError("特徵實驗尚待 D 實作")

