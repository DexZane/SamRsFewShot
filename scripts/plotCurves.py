"""训练曲线可视化脚本

从训练日志文件解析 loss/mIoU/lr，生成三合一曲线图。

Usage:
    python scripts/plotCurves.py --logFile ./runs/training.log
    python scripts/plotCurves.py --logFile ./runs/training.log --savePath ./results
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ──────────────────────────────────────────────
# 多格式保存辅助函数
# ──────────────────────────────────────────────

def _saveFig(fig, saveDir: str, stem: str):
    """将图表保存为 PNG / PDF / SVG 三种格式

    Args:
        fig:     matplotlib Figure 对象
        saveDir: 输出目录（不存在时自动创建）
        stem:    文件名（不含扩展名），如 "curves"
    """
    outDir = Path(saveDir)
    outDir.mkdir(parents=True, exist_ok=True)
    for fmt, dpi in [("png", 150), ("pdf", None), ("svg", None)]:
        path = outDir / f"{stem}.{fmt}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"[OK] 已保存: {path}")


# ──────────────────────────────────────────────
# 解析日志
# ──────────────────────────────────────────────

def parseLog(logPath: str) -> dict:
    """从日志文件中提取训练指标

    支持的日志格式（training/trainer.py 输出）：
        Epoch N: train_loss = X.XXXX
        Epoch N: val_mIoU = X.XXXX
        Epoch N: learning rate = X.XXXXXX

    Returns:
        dict with keys:
            epochs:    list[int]  — 所有出现过的 epoch
            trainLoss: list[(epoch, loss)]
            valMiou:   list[(epoch, miou)]
            lr:        list[(epoch, lr)]
    """
    trainLoss, valMiou, lrHistory = [], [], []

    patternLoss = re.compile(r"Epoch\s+(\d+):\s+train_loss\s*=\s*([\d.]+)")
    patternMiou = re.compile(r"Epoch\s+(\d+):\s+val_mIoU\s*=\s*([\d.]+)")
    patternLr   = re.compile(r"Epoch\s+(\d+):\s+learning rate\s*=\s*([\d.eE+\-]+)")

    with open(logPath, "r") as f:
        for line in f:
            m = patternLoss.search(line)
            if m:
                trainLoss.append((int(m.group(1)), float(m.group(2))))
                continue
            m = patternMiou.search(line)
            if m:
                valMiou.append((int(m.group(1)), float(m.group(2))))
                continue
            m = patternLr.search(line)
            if m:
                lrHistory.append((int(m.group(1)), float(m.group(2))))

    allEpochs = sorted({e for e, _ in trainLoss + valMiou + lrHistory})
    return dict(epochs=allEpochs, trainLoss=trainLoss, valMiou=valMiou, lr=lrHistory)


# ──────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────

def plotCurves(data: dict, savePath: str | None = None, show: bool = True):
    """绘制三合一曲线图：训练损失 / 验证 mIoU / 学习率"""

    trainEpochs  = [e for e, _ in data["trainLoss"]]
    trainLosses  = [v for _, v in data["trainLoss"]]
    valEpochs    = [e for e, _ in data["valMiou"]]
    valMious     = [v for _, v in data["valMiou"]]
    lrEpochs     = [e for e, _ in data["lr"]]
    lrValues     = [v for _, v in data["lr"]]

    if not trainLosses:
        print("[ERROR] 日志中没有找到训练数据，请检查 --logFile 路径")
        sys.exit(1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("SAM-RS Few-Shot Training Curves", fontsize=14, fontweight="bold")

    # ── 子图1: Train Loss ──
    ax = axes[0]
    ax.plot(trainEpochs, trainLosses, color="#2196F3", linewidth=1.8, label="Train Loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # 标注最低点
    minIdx = int(np.argmin(trainLosses))
    ax.annotate(f"min={trainLosses[minIdx]:.4f}\n@ep{trainEpochs[minIdx]}",
                xy=(trainEpochs[minIdx], trainLosses[minIdx]),
                xytext=(5, 10), textcoords="offset points",
                fontsize=8, color="#1565C0",
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.2))

    # ── 子图2: Val mIoU ──
    ax = axes[1]
    if valMious:
        ax.plot(valEpochs, valMious, color="#4CAF50", linewidth=1.8,
                marker="o", markersize=5, label="Val mIoU")
        ax.axhline(y=max(valMious), color="#E53935", linestyle="--",
                   linewidth=1.2, alpha=0.6, label=f"Best={max(valMious):.4f}")
        ax.set_title("Validation mIoU")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mIoU")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        # 标注最高点
        maxIdx = int(np.argmax(valMious))
        ax.annotate(f"best={valMious[maxIdx]:.4f}\n@ep{valEpochs[maxIdx]}",
                    xy=(valEpochs[maxIdx], valMious[maxIdx]),
                    xytext=(5, -20), textcoords="offset points",
                    fontsize=8, color="#2E7D32",
                    arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2))
    else:
        ax.text(0.5, 0.5, "No validation data yet", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Validation mIoU")

    # ── 子图3: Learning Rate ──
    ax = axes[2]
    if lrValues:
        ax.plot(lrEpochs, lrValues, color="#FF9800", linewidth=1.8, label="Learning Rate")
        ax.set_title("Learning Rate (Cosine Annealing)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No LR data found", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Learning Rate")

    plt.tight_layout()

    if savePath:
        _saveFig(fig, savePath, "curves")

    if show:
        plt.show()
    else:
        plt.close()


# ──────────────────────────────────────────────
# 打印统计摘要
# ──────────────────────────────────────────────

def printSummary(data: dict):
    trainLosses = [v for _, v in data["trainLoss"]]
    valMious    = [v for _, v in data["valMiou"]]
    lrValues    = [v for _, v in data["lr"]]

    print("=" * 50)
    print("训练统计摘要")
    print("=" * 50)

    if trainLosses:
        minIdx = int(np.argmin(trainLosses))
        print(f"总训练 epoch 数    : {len(trainLosses)}")
        print(f"初始 train_loss    : {trainLosses[0]:.4f}")
        print(f"最终 train_loss    : {trainLosses[-1]:.4f}")
        print(f"最低 train_loss    : {trainLosses[minIdx]:.4f} (epoch {data['trainLoss'][minIdx][0]})")
        print(f"下降幅度           : {(trainLosses[0] - trainLosses[-1]) / trainLosses[0] * 100:.1f}%")

    if valMious:
        maxIdx = int(np.argmax(valMious))
        print(f"\n验证次数           : {len(valMious)}")
        print(f"最佳 val_mIoU      : {valMious[maxIdx]:.4f} (epoch {data['valMiou'][maxIdx][0]})")
        print(f"最终 val_mIoU      : {valMious[-1]:.4f}")

    if lrValues:
        print(f"\n初始 LR            : {lrValues[0]:.6f}")
        print(f"最终 LR            : {lrValues[-1]:.8f}")

    print("=" * 50)


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def parseArgs():
    parser = argparse.ArgumentParser(description="绘制 SAM-RS 训练曲线")
    parser.add_argument("--logFile", type=str, required=True,
                        help="训练日志文件路径（trainer.py 输出的 .log 文件）")
    parser.add_argument("--savePath", type=str, default=None,
                        help="保存图片路径（默认只展示不保存）")
    parser.add_argument("--noShow", action="store_true",
                        help="不弹出图形窗口（服务器环境用）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()

    print(f"[INFO] 解析日志: {args.logFile}")
    data = parseLog(args.logFile)
    print(f"[INFO] 找到 {len(data['trainLoss'])} 个 epoch 的训练数据")
    print(f"[INFO] 找到 {len(data['valMiou'])} 次验证数据")

    printSummary(data)

    plotCurves(
        data,
        savePath=args.savePath,
        show=not args.noShow
    )
