"""Compare candidate images against supplied training/validation manifests.

Run from the repository root: python -m tools.audit_140k_overlap --help
No exact match is evidence only within the supplied reference set, never a
certificate of independence or absence of near duplicates / repeated identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from data.manifest import load_manifest, load_rgb_image


def fingerprints(path: Path) -> tuple[str, str]:
    with path.open("rb") as stream:
        file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    rgb = load_rgb_image(path)
    pixel_hash = hashlib.sha256(str(rgb.shape).encode() + b"\0" + rgb.tobytes()).hexdigest()
    return file_hash, pixel_hash


def audit(manifest: Path, references: list[Path], output_dir: Path) -> dict:
    candidates = load_manifest(manifest, require_generator=True)
    index: dict[str, set[str]] = defaultdict(set)
    reference_counts: dict[str, int] = defaultdict(int)
    for reference in references:
        for sample in load_manifest(reference, require_generator=True):
            if sample.split.lower() not in {"train", "training", "val", "valid", "validation"}:
                raise ValueError("參考清單只能包含訓練／驗證影像，不能混入 test 或 pilot")
            reference_counts[sample.split] += 1
            identifier = f"{reference.resolve()}::{sample.sample_id}::{sample.split}"
            for fingerprint in fingerprints(sample.path):
                index[fingerprint].add(identifier)

    rows = []
    for sample in candidates:
        file_hash, pixel_hash = fingerprints(sample.path)
        matches = index.get(file_hash, set()) | index.get(pixel_hash, set())
        status = ("known_overlap" if matches else
                  "no_exact_match_in_supplied_references" if references else "unverified_no_reference")
        rows.append({
            "sample_id": sample.sample_id, "path": str(sample.path),
            "source": sample.source, "split": sample.split,
            "file_sha256": file_hash, "rgb_sha256": pixel_hash,
            "status": status, "reference_matches": json.dumps(sorted(matches), ensure_ascii=False),
            "exclude_known_overlap": bool(matches),
        })
    summary = {
        "candidate_manifest": str(manifest.resolve()), "candidate_count": len(rows),
        "reference_manifests": [str(p.resolve()) for p in references],
        "reference_counts": dict(reference_counts),
        "known_overlap_count": sum(row["exclude_known_overlap"] for row in rows),
        "reference_coverage": "User-supplied scope only; completeness not established",
        "limitations": "Exact file/RGB matches only. No guarantee against re-encoding, near duplicates, repeated identities or unseen training data. Do not treat unmatched candidates as certified independent.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "overlap_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "overlap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference", action="append", default=[], type=Path,
                        help="共用六欄 manifest，可重複指定；只含官方 train/val")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.manifest, args.reference, args.output_dir), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(2, f"核對失敗：{error}\n")


if __name__ == "__main__":
    main()
