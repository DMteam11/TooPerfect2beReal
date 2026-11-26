"""Download the pinned upstream checkpoint and verify SHA256 before installation."""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


UPSTREAM_COMMIT = "8d4c81c9f5cac65d66f2ddebd02f2111f0f1c3c7"
SHA256 = "d9a7fd6a692c942b550f9848500dc3ffb10d5809cb0d0091990648bf369ad21c"
URL = f"https://media.githubusercontent.com/media/kristinealli/SyntheticSight/{UPSTREAM_COMMIT}/models/best_resnet50.pth"


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest == SHA256:
            print(f"Verified existing checkpoint: {destination}")
            return
        raise ValueError(f"Existing file has a different SHA256; choose another destination: {destination}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(URL, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    digest.update(block)
        if digest.hexdigest() != SHA256:
            raise ValueError("Downloaded checkpoint SHA256 does not match upstream")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("weights/syntheticsight.pth"))
    args = parser.parse_args()
    try:
        download(args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Download failed: {error}\n")
