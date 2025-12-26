# VisDrone 目标检测与计数实战作业

本项目提供了从数据预处理、EDA、格式转换、训练到可视化与评估的完整流水线，针对 VisDrone 无人机视角下的小目标检测场景，基于 Ultralytics YOLOv8 构建基线与改进方案。

## 环境准备

- 建议使用 Python 3.10+，GPU 训练需 CUDA11+（Ultralytics/torch 会依据环境选择 CPU/GPU）。
- 创建隔离环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 可选：若存在 GPU，确保正确安装了匹配版本的 torch/torchvision。
```

## 数据准备

1. 一键下载（会自动更新 `configs/visdrone.yaml` 的 `path` 指向下载目录，并整理 train/val/test-dev 文件夹）：

```bash
python scripts/download_visdrone.py --data-root data
```

2. 将官方 txt 标注转为 YOLO 标准格式（忽略类别 10/11 的无效区域，类别下标从 0 开始）。转换后 `images/` 原样引用，`labels/` 输出为 YOLO txt：

```bash
python scripts/convert_visdrone_to_yolo.py \
  --visdrone-root data/VisDrone2019-DET \
  --output-root data/VisDrone2019-DET \
  --split train val
```

转换后目录结构示例（运行下载与转换后）：

```
VisDrone2019-DET/
├── train
│   ├── images/*.jpg
│   └── labels/*.txt  # YOLO 标注
├── val
│   ├── images/*.jpg
│   └── labels/*.txt
└── test-dev
    └── images/*.jpg   # 无标注，可用于公开评测提交
```

## 数据探索 (EDA)

统计目标尺寸、宽高比、类别分布并输出可视化（可先在 train 标注上运行）：

```bash
python scripts/visdrone_eda.py \
  --labels data/VisDrone2019-DET/train/annotations \
  --output eda_outputs/train
```

生成的 `stats.json` 与若干 png 图可用于分析小目标比例、类别不均衡等问题。

## 训练

默认同时运行基线与改进方案（使用 YOLOv8n），可根据机器修改 batch 与 img-size：

```bash
python scripts/train_visdrone.py \
  --data-cfg configs/visdrone.yaml \
  --project runs/visdrone \
  --epochs 50 \
  --img-size 960 \
  --batch 16
```

- **基线**：适度 Mosaic、无 MixUp/Copy-Paste。
- **改进**：更强的 Mosaic/MixUp/Copy-Paste、轻微透视与色彩扰动，开启 Focal Loss（`fl_gamma=1.5`）以缓解前景/背景不平衡与小目标召回问题。

可根据需要替换为更大模型（如 `yolov8m.pt`）或增加训练轮数。

## 评估与可视化

在验证集上评估、保存预测图、生成 Grad-CAM 热力图（同时支持推理可视化）：

```bash
python scripts/visualize_and_eval.py \
  --weights runs/visdrone/improved_yolov8n/weights/best.pt \
  --data-cfg configs/visdrone.yaml \
  --sample-dir data/VisDrone2019-DET/val/images \
  --gradcam-image data/VisDrone2019-DET/val/images/0000001_00000_d_0000001.jpg \
  --output-dir viz_outputs \
  --imgsz 960
```

脚本会输出 mAP@0.5 与 mAP@0.5:0.95 指标，并在 `viz_outputs/` 下保存预测可视化与 Grad-CAM 热力图，可用于 Bad Case 分析。若只需快速推理，可将 `--sample-dir` 指向任意包含 `.jpg` 的文件夹。

## 消融实验建议

- **数据增强消融**：对比基线与改进方案的 mAP/召回，验证 Mosaic+MixUp+Copy-Paste 对小目标的收益。
- **损失函数消融**：调整 `fl_gamma`（0 vs 1.5）检验 Focal Loss 对前景稀疏场景的影响。
- **后处理**：可在 `visualize_and_eval.py` 中扩展为 Soft-NMS/DIoU-NMS，比较遮挡场景下的误检率。

## 目录概览

- `configs/visdrone.yaml`：YOLO 数据集配置（10 类，小目标场景）。
- `scripts/convert_visdrone_to_yolo.py`：标注转换脚本。
- `scripts/visdrone_eda.py`：EDA 与分布可视化。
- `scripts/train_visdrone.py`：基线与改进版训练脚本。
- `scripts/visualize_and_eval.py`：验证、预测可视化、Grad-CAM。
- `requirements.txt`：依赖列表。
