"""模型登記處。正式模型的 factory 規格見 INTERFACE_CONTRACT.md。"""

from pathlib import Path
from typing import Protocol

from models.base import ExplainableFaceDetector, FaceDetector
from models.demo import DemoBrightnessDetector


class ModelFactory(Protocol):
    def __call__(self, *, weights_path: Path | None, device: str) -> FaceDetector: ...


def _create_demo(*, weights_path: Path | None, device: str) -> FaceDetector:
    if weights_path is not None or device != "cpu":
        raise ValueError("demo 只用 CPU，無需權重")
    return DemoBrightnessDetector()


def _create_univfd(*, weights_path: Path | None, device: str) -> FaceDetector:
    # 在 factory 內才匯入，讓 demo 維持不需要 PyTorch。
    from models.univfd import UnivFDDetector

    if weights_path is None:
        raise ValueError("univfd 需要指定 weights")
    return UnivFDDetector(weights_path=weights_path, device=device)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "demo": _create_demo,
    "univfd": _create_univfd,
    # 接入後依約定新增 cnndetection、syntheticsight。
}


def create_model(name: str, *, weights_path: Path | None = None, device: str = "cpu") -> FaceDetector:
    factory = MODEL_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"模型 {name!r} 尚未登記；目前可用：{available}")
    if name != "demo" and weights_path is None:
        raise ValueError(f"正式模型 {name!r} 需要指定 weights")
    if weights_path is not None and not weights_path.is_file():
        raise FileNotFoundError(f"找不到模型權重：{weights_path}")

    model = factory(weights_path=weights_path, device=device)
    if model.name != name:
        raise ValueError(f"模型 name {model.name!r} 必須與登記名稱 {name!r} 一致")
    if name != "demo":
        if not isinstance(model, ExplainableFaceDetector):
            raise ValueError(f"正式模型 {name!r} 必須繼承 ExplainableFaceDetector")
        for attribute in ("weights_id", "preprocessing", "threshold_source"):
            value = getattr(model, attribute, "")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"正式模型 {name!r} 必須提供 {attribute}")
    return model

