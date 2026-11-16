"""統一實驗入口：讀設定、載入同一批影像、執行指定實驗。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.manifest import load_manifest
from experiments.features.run import run as run_features
from experiments.occlusion.run import run as run_occlusion
from experiments.performance.run import run as run_performance
from experiments.xai.run import run as run_xai
from models.base import ExplainableFaceDetector
from models.interface_check import check_interface
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
    parser.add_argument("--check-interface", action="store_true", help="只做第一階段介面驗收")
    args = parser.parse_args()

    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("設定檔必須是 JSON 物件")
        for key in ("experiment", "model", "manifest", "output_dir"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"設定檔的 {key} 必須是非空字串")
        for key in ("weights", "device"):
            if key in config and (not isinstance(config[key], str) or not config[key].strip()):
                raise ValueError(f"設定檔的 {key} 必須是非空字串")
        split = config.get("split")
        if split is not None and (not isinstance(split, str) or not split.strip()):
            raise ValueError("設定檔的 split 必須是非空字串或 null")

        experiment = config["experiment"]
        if experiment not in EXPERIMENTS:
            raise ValueError(f"未知實驗 {experiment!r}；可用：{', '.join(EXPERIMENTS)}")
        device = config.get("device", "cpu")
        weights = project_path(config["weights"]) if "weights" in config else None
        model = create_model(config["model"], weights_path=weights, device=device)
        manifest_path = project_path(config["manifest"])
        samples = load_manifest(
            manifest_path, split, require_generator=model.name != "demo"
        )
        if args.check_interface:
            check_interface(model, samples, device)
            print(f"介面驗收完成：{model.name} / {device}")
            return

        output_dir = project_path(config["output_dir"])
        EXPERIMENTS[experiment](model, samples, output_dir)
        run_info = {
            "experiment": experiment,
            "model": model.name,
            "device": device,
            "manifest": str(manifest_path.resolve()),
            "split": split,
            "sample_count": len(samples),
            "threshold": float(model.threshold),
            "python_version": sys.version.split()[0],
        }
        if isinstance(model, ExplainableFaceDetector) and weights is not None:
            run_info.update(
                weights_id=model.weights_id,
                weights_path=str(weights.resolve()),
                preprocessing=model.preprocessing,
                threshold_source=model.threshold_source,
                torch_version=getattr(sys.modules.get("torch"), "__version__", None),
            )
        (output_dir / "run_info.json").write_text(
            json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, ImportError, NotImplementedError) as error:
        parser.exit(2, f"執行失敗：{error}\n")
    print(f"完成：{experiment} / {model.name} / {len(samples)} 張；結果在 {output_dir}")


if __name__ == "__main__":
    main()

