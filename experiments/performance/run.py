"""DA 的逐圖推論入口。基本摘要供試跑，正式比較由 YUZI 使用逐圖分數進行。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from data.manifest import Sample, load_rgb_image
from models.base import FaceDetector


def _metrics(rows: list[dict]) -> dict:
    count = len(rows)
    real = [row for row in rows if row["label"] == 0]
    fake = [row for row in rows if row["label"] == 1]
    return {
        "count": count,
        "accuracy": sum(row["predicted_label"] == row["label"] for row in rows) / count,
        # 某來源可能只有真或只有假；不適用時用 null，不捏造數值。
        "real_accuracy": sum(row["predicted_label"] == 0 for row in real) / len(real) if real else None,
        "fake_recall": sum(row["predicted_label"] == 1 for row in fake) / len(fake) if fake else None,
    }


def run(model: FaceDetector, samples: list[Sample], output_dir: Path) -> None:
    if not samples:
        raise ValueError("效能實驗至少需要一張影像")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for sample in samples:
        image = load_rgb_image(sample.path)
        score = float(model.score_fake(image))
        try:
            predicted_label = model.label_from_score(score)
        except ValueError as error:
            raise ValueError(f"樣本 {sample.sample_id}：{error}") from error
        rows.append({
            "sample_id": sample.sample_id,
            "path": str(sample.path),
            "label": sample.label,
            "source": sample.source,
            "split": sample.split,
            "generator": sample.generator,
            "model": model.name,
            "score_fake": score,
            "threshold": model.threshold,
            "predicted_label": predicted_label,
        })

    # 每個模型每張圖一列。跨模型彙整用 (model, sample_id)，接特徵表用 sample_id。
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    summary = {
        "model": model.name,
        "threshold": model.threshold,
        "overall": _metrics(rows),
        "by_source": {source: _metrics(group) for source, group in sorted(by_source.items())},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

