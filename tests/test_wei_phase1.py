"""Meaningful regression checks; model integration needs the official checkpoint."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_140k_overlap import audit


class DemoRegistryTests(unittest.TestCase):
    def test_demo_does_not_import_torch(self):
        subprocess.run([sys.executable, "-c", "from models.registry import create_model; import sys; create_model('demo'); assert 'torch' not in sys.modules; assert 'torchvision' not in sys.modules"], check=True)


def manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "path", "label", "source", "split", "generator"])
        writer.writerows(rows)


class OverlapAuditTests(unittest.TestCase):
    def test_renamed_and_reencoded_pixel_duplicate_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.zeros((8, 9, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / "train.png")
            Image.fromarray(pixels).save(root / "renamed.bmp")
            Image.fromarray(pixels + 1).save(root / "other.png")
            manifest(root / "reference.csv", [["train1", "train.png", 0, "140k", "train", ""]])
            manifest(root / "candidate.csv", [
                ["copy", "renamed.bmp", 0, "140k", "pilot", ""],
                ["other", "other.png", 1, "140k", "pilot", "stylegan"],
            ])
            result = audit(root / "candidate.csv", [root / "reference.csv"], root / "out")
            self.assertEqual(result["known_overlap_count"], 1)
            with (root / "out/overlap_audit.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["status"], "known_overlap")
            self.assertEqual(rows[1]["status"], "no_exact_match_in_supplied_references")
            audit(root / "candidate.csv", [], root / "unknown")
            with (root / "unknown/overlap_audit.csv").open(encoding="utf-8") as stream:
                self.assertTrue(all(row["status"] == "unverified_no_reference" for row in csv.DictReader(stream)))

    def test_test_split_is_not_training_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "image.png")
            manifest(root / "samples.csv", [["a", "image.png", 0, "140k", "test", ""]])
            with self.assertRaisesRegex(ValueError, "參考清單"):
                audit(root / "samples.csv", [root / "samples.csv"], root / "out")


class SyntheticSightIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from models.registry import create_model

        torch.set_num_threads(2)
        weights = Path(__file__).resolve().parents[1] / "weights/syntheticsight.pth"
        if not weights.is_file():
            raise unittest.SkipTest("Download official checkpoint first; docs/WEI_PHASE1.md")
        cls.detector = create_model("syntheticsight", weights_path=weights, device="cpu")

    def test_official_model_score_batch_input_and_layer_gradients(self):
        from data.manifest import Sample
        from models.interface_check import check_interface

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            for label in (0, 1):
                path = root / f"fixture{label}.png"
                Image.new("RGB", (31, 43), (40 + label * 150, 70, 90)).save(path)
                samples.append(Sample(str(label), path, label, "synthetic_interface_fixture", "pilot"))
            check_interface(self.detector, samples, "cpu")
        self.assertFalse(self.detector.model.training)
        self.assertEqual(self.detector.threshold, 0.51)

    def test_input_contract_and_unverified_weights_are_rejected(self):
        from models.syntheticsight import SyntheticSightDetector

        with self.assertRaises(ValueError):
            self.detector.preprocess_tensor(np.zeros((8, 8), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pth"
            path.write_bytes(b"invalid or unverified checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                SyntheticSightDetector(weights_path=path)


if __name__ == "__main__":
    unittest.main()
