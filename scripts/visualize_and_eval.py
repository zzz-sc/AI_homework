import argparse
from pathlib import Path
import time

import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO


def run_validation(model: YOLO, data_cfg: Path, split: str) -> dict:
    results = model.val(data=str(data_cfg), split=split, save_json=True, verbose=True)
    metrics = {
        "map50": results.box.map50,
        "map50_95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
        "fps": results.speed["inference"] * 1000 if "inference" in results.speed else None,
    }
    return metrics


def save_predictions(model: YOLO, image_dir: Path, output_dir: Path, imgsz: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_dir.glob("*.jpg"):
        results = model.predict(source=str(image_path), imgsz=imgsz, conf=0.25, save=False, verbose=False)
        for result in results:
            plotted = result.plot()
            cv2.imwrite(str(output_dir / image_path.name), plotted)


def save_gradcam(model: YOLO, image_path: Path, output_path: Path) -> None:
    model.model.eval()
    # Pick the last C2f block if available; otherwise fall back to the penultimate module
    candidate_layers = [layer for layer in model.model.model if hasattr(layer, "cv3")]
    target_layers = [candidate_layers[-1] if candidate_layers else model.model.model[-2]]

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model.model.cuda()

    with GradCAM(model=model.model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    visualization = show_cam_on_image(rgb / 255.0, grayscale_cam, use_rgb=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


def benchmark_fps(model: YOLO, dummy_image: Path, warmup: int = 5, runs: int = 20) -> float:
    image = cv2.imread(str(dummy_image))
    if image is None:
        raise FileNotFoundError(f"Cannot read image {dummy_image}")
    for _ in range(warmup):
        model.predict(image, verbose=False)
    start = time.time()
    for _ in range(runs):
        model.predict(image, verbose=False)
    end = time.time()
    return runs / (end - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize YOLO results on VisDrone.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained weights (.pt).")
    parser.add_argument("--data-cfg", type=Path, default=Path("configs/visdrone.yaml"))
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--sample-dir", type=Path, help="Directory of images for qualitative results.")
    parser.add_argument("--gradcam-image", type=Path, help="Single image path for Grad-CAM visualization.")
    parser.add_argument("--output-dir", type=Path, default=Path("viz_outputs"))
    parser.add_argument("--imgsz", type=int, default=960)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))

    metrics = run_validation(model, args.data_cfg, args.val_split)
    print("Validation metrics:", metrics)

    if args.sample_dir:
        save_predictions(model, args.sample_dir, args.output_dir / "predictions", args.imgsz)

    if args.gradcam_image:
        save_gradcam(model, args.gradcam_image, args.output_dir / "gradcam.jpg")


if __name__ == "__main__":
    main()
