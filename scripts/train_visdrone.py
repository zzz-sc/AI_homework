import argparse
from pathlib import Path

from ultralytics import YOLO


def run_training(data_cfg: Path, model_name: str, epochs: int, img_size: int, batch: int, project: Path, name: str, augment_overrides: dict) -> None:
    """Launch a single YOLO training run with provided augment overrides."""
    model = YOLO(model_name)
    model.train(
        data=str(data_cfg),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        project=str(project),
        name=name,
        exist_ok=True,
        workers=8,
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=augment_overrides.get("mosaic", 1.0),
        mixup=augment_overrides.get("mixup", 0.0),
        copy_paste=augment_overrides.get("copy_paste", 0.0),
        hsv_h=augment_overrides.get("hsv_h", 0.015),
        hsv_s=augment_overrides.get("hsv_s", 0.7),
        hsv_v=augment_overrides.get("hsv_v", 0.4),
        degrees=augment_overrides.get("degrees", 0.0),
        translate=augment_overrides.get("translate", 0.1),
        scale=augment_overrides.get("scale", 0.5),
        shear=augment_overrides.get("shear", 0.0),
        perspective=augment_overrides.get("perspective", 0.0),
        fl_gamma=augment_overrides.get("fl_gamma", 0.0),
        box=augment_overrides.get("box", 7.5),
        cls=augment_overrides.get("cls", 0.5),
        nbs=augment_overrides.get("nbs", 64),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO models on VisDrone with baseline and improved settings.")
    parser.add_argument("--data-cfg", type=Path, default=Path("configs/visdrone.yaml"))
    parser.add_argument("--project", type=Path, default=Path("runs/visdrone"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project
    project.mkdir(parents=True, exist_ok=True)

    # 基线：温和数据增强 + BCE
    baseline_aug = {
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "scale": 0.5,
        "fl_gamma": 0.0,  # standard BCE loss
    }
    # 改进：更强的数据增强 + Focal Loss，针对小目标/前景稀疏
    improved_aug = {
        "mosaic": 0.8,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "scale": 0.75,
        "hsv_s": 0.9,
        "hsv_v": 0.5,
        "degrees": 5.0,
        "perspective": 0.0005,
        "fl_gamma": 1.5,  # Focal loss to handle imbalance
    }

    run_training(
        data_cfg=args.data_cfg,
        model_name="yolov8n.pt",
        epochs=args.epochs,
        img_size=args.img_size,
        batch=args.batch,
        project=project,
        name="baseline_yolov8n",
        augment_overrides=baseline_aug,
    )

    run_training(
        data_cfg=args.data_cfg,
        model_name="yolov8n.pt",
        epochs=args.epochs,
        img_size=args.img_size,
        batch=args.batch,
        project=project,
        name="improved_yolov8n",
        augment_overrides=improved_aug,
    )


if __name__ == "__main__":
    main()
