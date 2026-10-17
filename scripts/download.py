"""Download a dataset through anomalib's datamodule.

usage: python scripts/download.py visa|mvtec [--root DATA_DIR] [--categories bottle hazelnut]

MVTec's official host (mydrive.ch) is unreliable; if it fails we fall back to a
Hugging Face mirror of the extracted folders and fetch only the categories asked for.
"""
import argparse
from pathlib import Path

from anomalib.data import MVTecAD, Visa

DATASETS = {"visa": Visa, "mvtec": MVTecAD}
MVTEC_MIRROR = "foersben/mvtec-ad"


def mvtec_from_mirror(root: Path, categories: list[str]) -> None:
    from huggingface_hub import snapshot_download

    patterns = [f"{c}/*" for c in categories] + ["license.txt", "readme.txt"]
    snapshot_download(MVTEC_MIRROR, repo_type="dataset", local_dir=root, allow_patterns=patterns)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("--root", default="data")
    p.add_argument("--categories", nargs="*", default=["bottle", "hazelnut"],
                   help="mvtec mirror fallback only; default: bottle hazelnut")
    a = p.parse_args()
    root = Path(a.root) / a.dataset
    try:
        DATASETS[a.dataset](root=root).prepare_data()
    except Exception as e:  # noqa: BLE001
        if a.dataset != "mvtec":
            raise
        print(f"official download failed ({e.__class__.__name__}), using mirror {MVTEC_MIRROR}")
        mvtec_from_mirror(root, a.categories)
    print(f"ready: {root}")


if __name__ == "__main__":
    main()
