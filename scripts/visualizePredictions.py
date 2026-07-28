"""预测结果可视化脚本

加载训练好的模型，对验证集进行 few-shot episodic 推理，
输出预测结果图和各类别 IoU 分析。

Usage:
    # 服务器上（无图形界面，保存图片）
    python scripts/visualizePredictions.py \
        --checkpoint checkpoints/best_model.pth \
        --dataRoot data/LoveDA \
        --device cuda \
        --nEpisodes 30 \
        --savePath ./results \
        --noShow

    # 本地（弹出图形界面）
    python scripts/visualizePredictions.py \
        --checkpoint checkpoints/best_model.pth \
        --dataRoot data/LoveDA \
        --device cuda
"""

import argparse
import sys
import random
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

projectRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projectRoot))

from data.lovedaDataset import LoveDADataset
from models.samLora import SAMLoRA
from models.promptLearner import SimplePromptLearner


# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

CLASS_NAMES = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural']

# ImageNet 归一化参数（DefaultTransform 使用的值）
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD  = np.array([0.229, 0.224, 0.225])

# 各类别颜色（用于可视化）
CLASS_COLORS = [
    [0,   0,   0],    # 0 background  黑
    [255, 0,   0],    # 1 building     红
    [255, 255, 0],    # 2 road         黄
    [0,   0,   255],  # 3 water        蓝
    [128, 128, 128],  # 4 barren       灰
    [0,   255, 0],    # 5 forest       绿
    [0,   128, 0],    # 6 agricultural 深绿
]


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """反归一化 (3, H, W) float tensor → (H, W, 3) uint8 numpy"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * IMG_STD + IMG_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def buildBinaryTargets(masks: torch.Tensor, classIds: torch.Tensor):
    """复现 trainer._build_binary_targets 的逻辑"""
    classIds = classIds.view(-1, 1, 1, 1).to(masks.device)
    validMask = masks != 255
    targets = ((masks == classIds) & validMask).float()
    return targets, validMask


def computeBinaryIou(pred: torch.Tensor, target: torch.Tensor,
                     validMask: torch.Tensor) -> float:
    """计算二值分割 IoU，忽略 validMask=False 的像素"""
    predBin   = (torch.sigmoid(pred) > 0.5) & validMask   # (B,1,H,W) bool
    targetBin = (target > 0.5) & validMask

    intersection = (predBin & targetBin).sum().item()
    union        = (predBin | targetBin).sum().item()

    return intersection / union if union > 0 else float("nan")


def buildClassIndex(dataset: LoveDADataset) -> dict:
    """按主导类别索引验证集样本 {class_id: [idx, ...]}"""
    index = defaultdict(list)
    for idx, (_, _, classId) in enumerate(dataset.samples):
        index[classId].append(idx)
    return index


# ──────────────────────────────────────────────
# 推理
# ──────────────────────────────────────────────

@torch.no_grad()
def runEpisodes(model, promptLearner, dataset, classIndex, device,
                nEpisodes=30, nWay=1, kShot=1):
    """运行 N 个 few-shot episode，返回每个 episode 的结果

    Returns:
        list of dict:
            classId, iou, image, gtMask, predMask, predProb
    """
    model.eval()
    promptLearner.eval()

    availableClasses = [c for c, idxs in classIndex.items()
                        if len(idxs) >= kShot + 1 and c != 0]  # 排除 background

    results = []
    for _ in range(nEpisodes):
        # 随机选一个类别
        classId = random.choice(availableClasses)
        idxs = classIndex[classId].copy()
        random.shuffle(idxs)

        # 取第一张作为 query，剩余作为 support（这里 nWay=1 kShot=1）
        queryIdx  = idxs[0]
        queryImg, queryMask, _ = dataset[queryIdx]  # (3,H,W), (H,W), int

        # 拼成 batch (B=1)
        images   = queryImg.unsqueeze(0).to(device)          # (1, 3, H, W)
        masks    = queryMask.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
        classIds = torch.tensor([classId], device=device)    # (1,)

        # 前向推理
        prompts      = promptLearner(classIds)       # (1, nPrompts, 256)
        promptEmbeds = prompts.mean(dim=1)           # (1, 256)
        predMasks    = model(images, promptEmbeds)   # (1, 1, H, W)

        # 构造二值目标
        targets, validMask = buildBinaryTargets(masks, classIds)
        iou = computeBinaryIou(predMasks, targets, validMask)

        results.append(dict(
            classId   = classId,
            iou       = iou,
            image     = denormalize(queryImg),
            gtMask    = targets[0, 0].cpu().numpy(),        # (H, W) 0/1
            validMask = validMask[0, 0].cpu().numpy(),      # (H, W) bool
            predProb  = torch.sigmoid(predMasks[0, 0]).cpu().numpy(),  # (H, W)
            predMask  = (torch.sigmoid(predMasks[0, 0]) > 0.5).cpu().numpy(),
        ))

    return results


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────

def visualizeGrid(results: list, savePath: str | None, show: bool,
                  maxSamples: int = 8):
    """可视化前 N 个 episode：原图 | GT | 预测概率 | 预测二值"""
    samples = results[:maxSamples]
    nRows   = len(samples)
    fig, axes = plt.subplots(nRows, 4, figsize=(16, nRows * 3.5))
    if nRows == 1:
        axes = axes[np.newaxis, :]

    colTitles = ["Original Image", "Ground Truth", "Pred Probability", "Pred Mask (>0.5)"]
    for col, title in enumerate(colTitles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, r in enumerate(samples):
        className = CLASS_NAMES[r["classId"]] if r["classId"] < len(CLASS_NAMES) else f"cls{r['classId']}"
        iouStr    = f"{r['iou']:.3f}" if not np.isnan(r["iou"]) else "N/A"

        # 行标签
        axes[row, 0].set_ylabel(f"cls={className}\nIoU={iouStr}", fontsize=9, rotation=0,
                                labelpad=60, va="center")

        # 原图
        axes[row, 0].imshow(r["image"])
        axes[row, 0].axis("off")

        # Ground truth（前景绿色，背景黑色，ignore=白色）
        gtVis = np.zeros((*r["gtMask"].shape, 3), dtype=np.uint8)
        gtVis[r["gtMask"] > 0.5]    = CLASS_COLORS[r["classId"]]
        gtVis[~r["validMask"]]       = [255, 255, 255]  # ignore 区域标白
        axes[row, 1].imshow(gtVis)
        axes[row, 1].axis("off")

        # 预测概率热图
        im = axes[row, 2].imshow(r["predProb"], cmap="RdYlGn", vmin=0, vmax=1)
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # 预测二值 mask 叠加在原图上
        overlay = r["image"].copy()
        predFg  = r["predMask"].astype(bool)
        overlay[predFg] = (overlay[predFg] * 0.4 +
                           np.array(CLASS_COLORS[r["classId"]]) * 0.6).astype(np.uint8)
        axes[row, 3].imshow(overlay)
        axes[row, 3].axis("off")

    plt.suptitle("SAM-RS Few-Shot: Prediction Visualization", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if savePath:
        p = Path(savePath) / "predictions.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=120, bbox_inches="tight")
        print(f"[OK] 预测可视化已保存: {p}")

    if show:
        plt.show()
    else:
        plt.close()


def visualizePerClassIou(results: list, savePath: str | None, show: bool):
    """柱状图：各类别平均 IoU"""
    classIous = defaultdict(list)
    for r in results:
        if not np.isnan(r["iou"]):
            classIous[r["classId"]].append(r["iou"])

    classIds   = sorted(classIous.keys())
    classNames = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls{c}" for c in classIds]
    meanIous   = [np.mean(classIous[c]) for c in classIds]
    stdIous    = [np.std(classIous[c])  for c in classIds]
    counts     = [len(classIous[c])     for c in classIds]

    colors = [np.array(CLASS_COLORS[c]) / 255.0 if c < len(CLASS_COLORS)
              else (0.5, 0.5, 0.5) for c in classIds]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classNames, meanIous, yerr=stdIous, color=colors,
                  edgecolor="black", linewidth=0.8, capsize=5, alpha=0.85)

    # 在柱子上方标注均值和样本数
    for bar, mean, cnt in zip(bars, meanIous, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.3f}\n(n={cnt})", ha="center", va="bottom", fontsize=9)

    overallMiou = np.mean([r["iou"] for r in results if not np.isnan(r["iou"])])
    ax.axhline(y=overallMiou, color="red", linestyle="--", linewidth=1.5,
               label=f"Overall mIoU={overallMiou:.4f}")

    ax.set_title("Per-Class IoU Analysis", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Mean Binary IoU")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if savePath:
        p = Path(savePath) / "perClassIou.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=120, bbox_inches="tight")
        print(f"[OK] 类别IoU图已保存: {p}")

    if show:
        plt.show()
    else:
        plt.close()


def printPerClassSummary(results: list):
    """打印各类别 IoU 统计表"""
    classIous = defaultdict(list)
    for r in results:
        if not np.isnan(r["iou"]):
            classIous[r["classId"]].append(r["iou"])

    print("\n" + "=" * 55)
    print(f"{'类别':<14} {'平均IoU':>8} {'标准差':>8} {'样本数':>6}")
    print("-" * 55)
    allIous = []
    for classId in sorted(classIous.keys()):
        name   = CLASS_NAMES[classId] if classId < len(CLASS_NAMES) else f"cls{classId}"
        ious   = classIous[classId]
        allIous.extend(ious)
        print(f"{name:<14} {np.mean(ious):>8.4f} {np.std(ious):>8.4f} {len(ious):>6}")
    print("-" * 55)
    print(f"{'Overall mIoU':<14} {np.mean(allIous):>8.4f} {np.std(allIous):>8.4f} {len(allIous):>6}")
    print("=" * 55)


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def parseArgs():
    parser = argparse.ArgumentParser(description="SAM-RS 预测结果可视化")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径（best_model.pth）")
    parser.add_argument("--dataRoot",   type=str, default="./data/LoveDA",
                        help="数据集根目录")
    parser.add_argument("--split",      type=str, default="Val",
                        choices=["Train", "Val"],
                        help="使用训练集还是验证集（默认 Val）")
    parser.add_argument("--device",     type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--nEpisodes",  type=int, default=60,
                        help="推理 episode 数量（越多类别统计越稳定）")
    parser.add_argument("--savePath",   type=str, default=None,
                        help="图片保存目录（为空则不保存）")
    parser.add_argument("--noShow",     action="store_true",
                        help="不弹出图形窗口（服务器环境）")
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


def main():
    args = parseArgs()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ── 加载 checkpoint ──
    print(f"[INFO] 加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    print(f"[INFO] Checkpoint epoch: {ckpt['epoch']}, best mIoU: {ckpt.get('best_miou', 'N/A'):.4f}")

    # ── 初始化模型 ──
    model = SAMLoRA(
        samCheckpoint=config.model.samCheckpoint,
        modelType=config.model.samModelType,
        loraRank=config.model.loraRank,
        loraAlpha=config.model.loraAlpha,
        loraDropout=config.model.loraDropout,
    )
    promptLearner = SimplePromptLearner(
        numClasses=config.data.numClasses,
        nPrompts=config.model.nPrompts,
        embedDim=256,
        initStd=config.model.promptInitStd,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    promptLearner.load_state_dict(ckpt["prompt_learner_state_dict"])
    model.to(device)
    promptLearner.to(device)
    print("[INFO] 模型权重加载完成")

    # ── 加载数据集 ──
    print(f"[INFO] 加载 {args.split} 集: {args.dataRoot}")
    dataset    = LoveDADataset(root=args.dataRoot, split=args.split, download=False)
    classIndex = buildClassIndex(dataset)
    print(f"[INFO] 样本总数: {len(dataset)}")
    for cid, idxs in sorted(classIndex.items()):
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        print(f"  class {cid} ({name}): {len(idxs)} 个样本")

    # ── 运行 episodes ──
    print(f"\n[INFO] 运行 {args.nEpisodes} 个 few-shot episodes...")
    results = runEpisodes(
        model, promptLearner, dataset, classIndex,
        device=device, nEpisodes=args.nEpisodes,
    )

    # ── 打印统计 ──
    printPerClassSummary(results)

    # ── 可视化 ──
    visualizeGrid(results, savePath=args.savePath, show=not args.noShow)
    visualizePerClassIou(results, savePath=args.savePath, show=not args.noShow)

    print("\n[DONE] 可视化完成")


if __name__ == "__main__":
    main()
