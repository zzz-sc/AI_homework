import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

VISDRONE_CLASSES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


def read_annotation_file(label_path: Path) -> pd.DataFrame:
    """解析单个 VisDrone 标注 txt，返回 DataFrame 便于统计分析。"""
    columns = [
        "bbox_left",
        "bbox_top",
        "bbox_width",
        "bbox_height",
        "score",
        "category",
        "truncation",
        "occlusion",
    ]
    records = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            values = [float(x) for x in parts[: len(columns)]]
            records.append(values)
    df = pd.DataFrame(records, columns=columns)
    # 衍生特征：面积与宽高比便于分析小目标比例和长宽分布
    df["area"] = df["bbox_width"] * df["bbox_height"]
    df["aspect_ratio"] = df["bbox_width"] / (df["bbox_height"] + 1e-6)
    df["category_name"] = df["category"].astype(int).map(VISDRONE_CLASSES)
    return df


def collect_dataset_stats(label_dir: Path) -> pd.DataFrame:
    """遍历标注目录，汇总为单个 DataFrame。"""
    label_files = sorted(label_dir.glob("*.txt"))
    dataframes = []
    for label_path in tqdm(label_files, desc="Loading annotations"):
        df = read_annotation_file(label_path)
        df["image_id"] = label_path.stem
        dataframes.append(df)
    if not dataframes:
        raise ValueError(f"No annotation files found under {label_dir}")
    return pd.concat(dataframes, ignore_index=True)


def plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """绘制宽/高/面积/类别分布以及宽高散点，输出 png 和统计 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 4))
    sns.histplot(df["bbox_width"], bins=50)
    plt.title("Width distribution (pixels)")
    plt.tight_layout()
    plt.savefig(output_dir / "width_hist.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(df["bbox_height"], bins=50)
    plt.title("Height distribution (pixels)")
    plt.tight_layout()
    plt.savefig(output_dir / "height_hist.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(df["area"], bins=60, log_scale=True)
    plt.title("Bounding box area (log scale)")
    plt.tight_layout()
    plt.savefig(output_dir / "area_hist.png")

    plt.figure(figsize=(8, 4))
    sns.countplot(y="category_name", data=df, order=df["category_name"].value_counts().index)
    plt.title("Category frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "category_freq.png")

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="bbox_width", y="bbox_height", data=df.sample(min(5000, len(df))), alpha=0.3)
    plt.title("Width vs Height")
    plt.tight_layout()
    plt.savefig(output_dir / "wh_scatter.png")

    stats = {
        "num_boxes": int(len(df)),
        "num_images": int(df["image_id"].nunique()),
        "class_counts": df["category_name"].value_counts().to_dict(),
        "area_quantiles": df["area"].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).round(2).to_dict(),
        "aspect_ratio_quantiles": df["aspect_ratio"].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).round(3).to_dict(),
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for VisDrone detection dataset.")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels directory (VisDrone txt format).")
    parser.add_argument("--output", type=Path, default=Path("eda_outputs"), help="Directory to store plots and stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect_dataset_stats(args.labels)
    plot_distributions(df, args.output)


if __name__ == "__main__":
    main()
