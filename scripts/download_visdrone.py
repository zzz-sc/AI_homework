import argparse
import shutil
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

BASE_URL = "https://github.com/VisDrone/VisDrone-Dataset/releases/download/VisionChallenge2019"
FILE_LIST = [
    "VisDrone2019-DET-train.zip",
    "VisDrone2019-DET-val.zip",
    "VisDrone2019-DET-test-dev.zip",
]


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def move_split(extracted_dir: Path, target_root: Path, split_name: str) -> None:
    src = extracted_dir / f"VisDrone2019-DET-{split_name}"
    if not src.exists():
        raise FileNotFoundError(f"Expected extracted dir {src} not found")
    dest = target_root / split_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(src.as_posix(), dest.as_posix())


def update_yaml_path(cfg_path: Path, dataset_root: Path) -> None:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(dataset_root.resolve())
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"Updated {cfg_path} path -> {cfg['path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VisDrone2019-DET dataset and update YOLO config path.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Directory to store dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/visdrone.yaml"), help="YOLO data config to update.")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if zips already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_root = data_root / "VisDrone2019-DET"
    tmp_dir = data_root / "tmp_visdrone"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for fname in FILE_LIST:
        url = f"{BASE_URL}/{fname}"
        zip_path = tmp_dir / fname
        if zip_path.exists() and args.skip_download:
            print(f"Skipping existing file: {zip_path}")
        else:
            print(f"Downloading {url} -> {zip_path}")
            download_file(url, zip_path)
        print(f"Extracting {zip_path}")
        extract_zip(zip_path, tmp_dir)

    dataset_root.mkdir(parents=True, exist_ok=True)
    move_split(tmp_dir, dataset_root, "train")
    move_split(tmp_dir, dataset_root, "val")
    move_split(tmp_dir, dataset_root, "test-dev")

    shutil.rmtree(tmp_dir)
    update_yaml_path(args.config, dataset_root)
    print(f"Dataset ready at {dataset_root}")


if __name__ == "__main__":
    main()
