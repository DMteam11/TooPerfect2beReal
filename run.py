"""統一實驗入口：讀設定、載入同一批影像、執行指定實驗。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.manifest import load_manifest
from experiments.features.run import run as run_features
from experiments.occlusion.run import run as run_occlusion
from experiments.performance.run import run as run_performance
from experiments.xai.run import run as run_xai
from models.registry import create_model


PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = {
    "performance": run_performance,
    "xai": run_xai,
    "occlusion": run_occlusion,
    "features": run_features,
}


def project_path(value: str) -> Path:
    """設定檔中的相對路徑一律從 project/ 起算。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="執行 AI 臉部偵測研究實驗")
    parser.add_argument("--config", required=True, help="JSON 設定檔路徑")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    required = {"experiment", "model", "manifest", "output_dir"}
    missing = required - set(config)
    if missing:
        parser.error(f"設定檔缺少：{', '.join(sorted(missing))}")

    experiment = config["experiment"]
    if experiment not in EXPERIMENTS:
        parser.error(f"未知實驗 {experiment!r}；可用：{', '.join(EXPERIMENTS)}")
    try:
        model = create_model(config["model"])
        samples = load_manifest(project_path(config["manifest"]), config.get("split"))
        output_dir = project_path(config["output_dir"])
        EXPERIMENTS[experiment](model, samples, output_dir)
    except (FileNotFoundError, ValueError, NotImplementedError) as error:
        parser.exit(2, f"執行失敗：{error}\n")
    print(f"完成：{experiment} / {model.name} / {len(samples)} 張；結果在 {output_dir}")


if __name__ == "__main__":
    main()

