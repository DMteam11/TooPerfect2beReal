"""讀取共用樣本清單，欄位定義見 INTERFACE_CONTRACT.md。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


REQUIRED_COLUMNS = ("sample_id", "path", "label", "source", "split")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: Path
    label: int  # 0：真實影像；1：生成影像。
    source: str  # 資料集或子來源，例如 140k、dff_wiki。
    split: str  # 例如 pilot、val、test。
    generator: str | None = None  # 真圖留空，假圖記生成方法。


def load_manifest(
    manifest_path: Path, split: str | None = None, *, require_generator: bool = False
) -> list[Sample]:
    """讀取 CSV；相對影像路徑以 CSV 所在目錄為基準。"""
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到樣本清單：{manifest_path}")

    samples: list[Sample] = []
    seen_ids: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = set(REQUIRED_COLUMNS)
        if require_generator:
            columns.add("generator")
        missing = columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"樣本清單缺少欄位：{', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            values = {key: (row[key] or "").strip() for key in REQUIRED_COLUMNS}
            if not all(values.values()):
                raise ValueError(f"第 {row_number} 列有空白的必要欄位")
            if values["label"] not in {"0", "1"}:
                raise ValueError(f"第 {row_number} 列 label 必須是 0 或 1")
            generator = (row.get("generator") or "").strip() or None
            if require_generator:
                if values["label"] == "1" and generator is None:
                    raise ValueError(f"第 {row_number} 列是假圖，generator 必須填寫")
                if values["label"] == "0" and generator is not None:
                    raise ValueError(f"第 {row_number} 列是真圖，generator 應留空")
            if values["sample_id"] in seen_ids:
                raise ValueError(f"sample_id 重複：{values['sample_id']}")
            seen_ids.add(values["sample_id"])

            image_path = Path(values["path"])
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            if split is not None and values["split"] != split:
                continue
            if not image_path.is_file():
                raise FileNotFoundError(f"第 {row_number} 列找不到影像：{image_path}")
            samples.append(
                Sample(
                    sample_id=values["sample_id"],
                    path=image_path.resolve(),
                    label=int(values["label"]),
                    source=values["source"],
                    split=values["split"],
                    generator=generator,
                )
            )

    if not samples:
        raise ValueError(f"樣本清單沒有符合 split={split!r} 的影像")
    return samples


def load_rgb_image(path: Path) -> np.ndarray:
    """統一傳入 H×W×3、uint8、RGB 影像；模型專用縮放留給 adapter。"""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.uint8).copy()

