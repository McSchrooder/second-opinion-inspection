"""Download a dataset into data/<name>.

usage: python scripts/download.py visa|mvtec|goodsad [--root data] [--categories bottle hazelnut]

MVTec's official host is unreliable. If it fails we pull the requested categories from a
Hugging Face mirror of the extracted folders instead.
GoodsAD (supermarket goods, MVTec AD layout, 3000 px images) comes as one zip per category
from the authors' OneDrive. Default: the five food and drink categories, about 11 GB.
"""
import argparse
import subprocess
import zipfile
from pathlib import Path

from anomalib.data import MVTecAD, Visa

DATASETS = {"visa": Visa, "mvtec": MVTecAD}
MVTEC_MIRROR = "foersben/mvtec-ad"
GOODSAD_BASE = "https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/2015216892_mail_hfut_edu_cn/"
GOODSAD = {  # github.com/jianzhang96/GoodsAD
    "drink_bottle": "EeoscD4PU4VAoaiTeeGyrgEBRXoibgXiRHACWdRily-i-w?e=YENamB",
    "drink_can": "Efs7rgdmVWJKu_eW2RxgswIBr15PdwwoDPnftnLbbjAyAw?e=iMR6Q6",
    "food_bottle": "ESib3l3xt4NLqEjVq76MykUBqgLsLbeDnSeCMb8YAOKbzg?e=fQDecg",
    "food_box": "EbZlumiFMxZGi2cjIrE-IGYBxiRFEjEBNNZCCI6frPEQVg?e=rMDeRj",
    "food_package": "ETInGCW7EOBKoFmh31-Y8PkB17MKP_iaVOhGLRuWyU1EQA?e=vQMcW4",
    "cigarette_box": "EU2Lgyz64k1En435HtDAtVMB1GlzidKCUA_tFLUIr5Wq-g?e=ZxabJ7",
}
GOODSAD_DEFAULT = ["drink_can", "drink_bottle", "food_bottle", "food_package", "food_box"]


def mvtec_from_mirror(root: Path, categories: list[str]) -> None:
    from huggingface_hub import snapshot_download

    patterns = [f"{c}/*" for c in categories] + ["license.txt", "readme.txt"]
    snapshot_download(MVTEC_MIRROR, repo_type="dataset", local_dir=root, allow_patterns=patterns)


def goodsad(root: Path, categories: list[str]) -> None:
    """One zip per category. OneDrive needs the redirect cookie, so curl with a cookie jar."""
    root.mkdir(parents=True, exist_ok=True)
    for c in categories:
        if (root / c / "train").exists():
            print(f"have {c}")
            continue
        url = GOODSAD_BASE + GOODSAD[c] + "&download=1"
        zpath = root / f"{c}.zip"
        jar = root / ".cookies"
        print(f"downloading {c} ...", flush=True)
        subprocess.run(["curl", "-L", "-sS", "-b", str(jar), "-c", str(jar), "-o", str(zpath), url], check=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
        zpath.unlink()
        print(f"ready {c}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=[*DATASETS, "goodsad"])
    p.add_argument("--root", default="data")
    p.add_argument("--categories", nargs="*", default=None, help="mvtec mirror and goodsad only")
    a = p.parse_args()
    root = Path(a.root) / a.dataset
    if a.dataset == "goodsad":
        goodsad(root, a.categories or GOODSAD_DEFAULT)
        return
    try:
        DATASETS[a.dataset](root=root).prepare_data()
    except Exception as e:  # noqa: BLE001
        if a.dataset != "mvtec":
            raise
        print(f"official download failed ({type(e).__name__}), using mirror {MVTEC_MIRROR}")
        mvtec_from_mirror(root, a.categories or ["bottle", "hazelnut"])
    print(f"ready: {root}")


if __name__ == "__main__":
    main()
