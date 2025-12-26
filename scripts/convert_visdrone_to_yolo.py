import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

VISDRONE_IGNORE_CATEGORIES = {10, 11}  # "others" and "ignoring regions" in official spec


def convert_annotation(label_path: Path, image_path: Path, output_label_path: Path) -> None:
    """将单张图片的 VisDrone 标注转为 YOLO txt（归一化坐标，类别从 0 起）。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    rows = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split(",")
            if len(fields) < 8:
                continue
            x0 = float(fields[0])
            y0 = float(fields[1])
            w = float(fields[2])
            h = float(fields[3])
            category = int(fields[5])
            if category in VISDRONE_IGNORE_CATEGORIES:
                continue
            category -= 1  # shift to zero-based for YOLO
            # 计算归一化坐标（中心点 + 宽高），避免训练时依赖图像分辨率
            x_center = (x0 + w / 2) / width
            y_center = (y0 + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            rows.append(f"{category} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    with output_label_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")


def convert_split(split_dir: Path, output_dir: Path) -> None:
    image_dir = split_dir / "images"
    label_dir = split_dir / "annotations"
    output_image_dir = output_dir / "images"
    output_image_dir.parent.mkdir(parents=True, exist_ok=True)
    # 复用原图：创建符号链接，避免重复占用磁盘
    if not output_image_dir.exists():
        output_image_dir.symlink_to(image_dir, target_is_directory=True)
    for label_path in tqdm(sorted(label_dir.glob("*.txt")), desc=f"Converting {split_dir.name}"):
        image_path = image_dir / (label_path.stem + ".jpg")
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for label: {label_path}")
        output_label_path = output_dir / "labels" / (label_path.stem + ".txt")
        convert_annotation(label_path, image_path, output_label_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO format with truncation/occlusion kept.")
    parser.add_argument("--visdrone-root", type=Path, required=True, help="Path to VisDrone2019-DET dataset root.")
    parser.add_argument("--output-root", type=Path, default=Path("VisDrone2019-DET"), help="Output root containing images/ and labels/.")
    parser.add_argument("--split", type=str, nargs="+", default=["train", "val"], help="Dataset splits to convert.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.split:
        convert_split(args.visdrone_root / split, args.output_root / split)


if __name__ == "__main__":
    main()
