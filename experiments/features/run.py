"""DA 的逐圖特徵與 TCAV 工作區。YUZI 使用交付的特徵表做統計。"""

from pathlib import Path

from data.manifest import Sample
from models.base import FaceDetector


def run(model: FaceDetector, samples: list[Sample], output_dir: Path) -> None:
    # 特徵每個 sample_id 只算一次，保存 roi_failed，供 YUZI 用同一編號接表。
    # TCAV 沿用同一 ROI 定義，透過正式模型的中間層與梯度介面取值。
    raise NotImplementedError("特徵計算與 TCAV 尚待第二階段實作")

