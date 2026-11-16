"""用少量真、假圖驗收介面。分類答對幾張屬於第二階段的效能問題。"""

from collections.abc import Mapping
from contextlib import ExitStack
from math import isclose

import numpy as np

from data.manifest import Sample, load_rgb_image
from models.base import ExplainableFaceDetector, FaceDetector


def check_interface(model: FaceDetector, samples: list[Sample], device: str) -> None:
    examples = {}
    for sample in samples:
        examples.setdefault(sample.label, sample)
        if len(examples) == 2:
            break
    if set(examples) != {0, 1}:
        raise ValueError("介面驗收的清單須至少有一張真圖與一張假圖")

    for sample in examples.values():
        image = load_rgb_image(sample.path)
        score = model.score_fake(image)
        if not isinstance(score, float):
            raise ValueError(f"{model.name} 的 score_fake 必須回傳 float")
        model.label_from_score(score)
        if isinstance(model, ExplainableFaceDetector):
            layers = _check_gradients(model, image, score, device)
            print(f"{sample.sample_id}：分數與梯度通過，可讀取的梯度層：{', '.join(layers)}")

    if not isinstance(model, ExplainableFaceDetector):
        print("demo：共用讀圖、真假分數和門檻通過。梯度與中間層需用正式模型驗收。")


def _check_gradients(
    model: ExplainableFaceDetector, image: np.ndarray, score: float, device: str
) -> list[str]:
    import torch

    tensor = model.preprocess_tensor(image)
    expected_device = torch.device(device)
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 4
        or tensor.shape[:2] != (1, 3)
        or tensor.numel() == 0
        or not tensor.is_floating_point()
        or not torch.isfinite(tensor).all().item()
    ):
        raise ValueError("preprocess_tensor 必須回傳有限的 (1, 3, H, W) 浮點張量")
    if tensor.device.type != expected_device.type or (
        expected_device.index is not None and tensor.device.index != expected_device.index
    ):
        raise ValueError(f"前處理張量位於 {tensor.device}，預期裝置為 {device}")

    layers = model.get_target_layers()
    if not isinstance(layers, Mapping) or not layers or any(
        not isinstance(name, str) or not name.strip() or not isinstance(layer, torch.nn.Module)
        for name, layer in layers.items()
    ):
        raise ValueError("get_target_layers 必須提供具名的 torch.nn.Module")

    activations = {}
    # 用重複影像測 N=2，也能確認個別推論與批次的分數語意相同。
    batch = torch.cat((tensor, tensor)).detach().requires_grad_()
    with ExitStack() as stack:
        for name, layer in layers.items():
            handle = layer.register_forward_hook(
                lambda _module, _inputs, output, name=name: activations.__setitem__(name, output)
            )
            stack.callback(handle.remove)
        logits = model.forward_logits(batch)

    if (
        not isinstance(logits, torch.Tensor)
        or logits.shape != (2,)
        or not logits.is_floating_point()
        or logits.device != batch.device
        or not torch.isfinite(logits).all().item()
        or not logits.requires_grad
    ):
        raise ValueError("forward_logits 必須在同裝置回傳可微分的 (N,) 浮點 fake logits")
    if any(not isclose(value, score, abs_tol=1e-4) for value in logits.detach().sigmoid().tolist()):
        raise ValueError("score_fake 與 sigmoid(fake logit) 不一致")

    candidates = {
        name: output for name, output in activations.items()
        if isinstance(output, torch.Tensor) and output.requires_grad
    }
    input_grad, *layer_grads = torch.autograd.grad(
        logits.sum(), (batch, *candidates.values()), allow_unused=True
    )
    if input_grad is None or not torch.isfinite(input_grad).all().item():
        raise ValueError("無法對模型輸入取得有限的梯度")
    usable = [
        name for name, grad in zip(candidates, layer_grads)
        if grad is not None and torch.isfinite(grad).all().item()
    ]
    if not usable:
        raise ValueError("至少需要一個可取得 activation 與梯度的候選層")
    return usable
