"""建立小型假資料，只用來確認整個程式流程能跑通。"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "demo"
MANIFEST = PROJECT_DIR / "data" / "demo_manifest.csv"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id, label, brightness in [
        ("demo_real_1", 0, 32),
        ("demo_real_2", 0, 64),
        ("demo_fake_1", 1, 192),
        ("demo_fake_2", 1, 224),
    ]:
        image = np.full((16, 16, 3), brightness, dtype=np.uint8)
        Image.fromarray(image).save(DATA_DIR / f"{sample_id}.png")
        rows.append({
            "sample_id": sample_id,
            "path": f"demo/{sample_id}.png",
            "label": label,
            "source": "toy_real" if label == 0 else "toy_fake",
            "split": "test",
            "generator": "toy_brightness" if label == 1 else "",
        })
    with MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["sample_id", "path", "label", "source", "split", "generator"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"已建立示範資料：{MANIFEST}")


if __name__ == "__main__":
    main()

