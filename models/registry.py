"""模型登記處；新模型完成後在這裡加一筆。"""

from collections.abc import Callable

from models.base import FaceDetector
from models.demo import DemoBrightnessDetector


MODEL_REGISTRY: dict[str, Callable[[], FaceDetector]] = {
    "demo": DemoBrightnessDetector,
}


def create_model(name: str) -> FaceDetector:
    try:
        return MODEL_REGISTRY[name]()
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"模型 {name!r} 尚未登記；目前可用：{available}") from error

