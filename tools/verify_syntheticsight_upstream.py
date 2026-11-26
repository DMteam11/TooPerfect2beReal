"""Compare adapter preprocessing/logits with a pinned official source checkout.

The checkout must contain src/synthetic_sight/{config,model,inference}.py.
This uses deterministic pixel fixtures, not labelled face evaluation data.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.registry import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=Path("weights/syntheticsight.pth"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream_root.resolve() / "src"))
    from synthetic_sight.inference import SyntheticSightDetector as OfficialDetector

    torch.set_num_threads(2)
    official = OfficialDetector(args.weights, device="cpu")
    adapter = create_model("syntheticsight", weights_path=args.weights, device="cpu")
    rng = np.random.default_rng(13)
    rows = []
    for height, width in [(16, 16), (224, 224), (301, 187), (93, 311)]:
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        reference_tensor = official.preprocess(Image.fromarray(image))
        tensor = adapter.preprocess_tensor(image)
        torch.testing.assert_close(tensor, reference_tensor, rtol=0, atol=0)
        with torch.no_grad():
            reference_logit = official.model(reference_tensor).reshape(-1)
            logit = adapter.forward_logits(tensor)
        torch.testing.assert_close(logit, reference_logit, rtol=0, atol=1e-6)
        reference = official.predict(Image.fromarray(image))
        score = adapter.score_fake(image)
        assert abs(score - reference.fake_probability) < 1e-6
        assert adapter.predict_label(image) == int(reference.label == "Fake")
        rows.append({"shape": [height, width, 3], "preprocessing_exact": True,
                     "max_logit_error": float((logit - reference_logit).abs().max()),
                     "score_error": abs(score - reference.fake_probability)})
    result = {"device": "cpu", "torch": torch.__version__, "cases": rows,
              "scope": "Official inference parity on pixel fixtures; not accuracy validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
