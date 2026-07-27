"""
Evaluation script for SAM-RS Few-Shot Learning

Usage:
    python scripts/evaluate.py --checkpoint path/to/checkpoint.pth --dataRoot ./data/loveda
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
projectRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projectRoot))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from config.default import Config
from models.samLora import SAMLoRA
from models.promptLearner import SimplePromptLearner
from utils.metrics import compute_iou
from tests.fixtures.sampleData import create_sample_dataset


class SimpleDataset(Dataset):
    """
    Simple dataset wrapper for (image, mask) tuples

    Converts numpy arrays to DataLoader format with proper batching.
    """

    def __init__(self, samples, numClasses):
        """
        Args:
            samples: List of (image, mask) tuples
            numClasses: Number of semantic classes
        """
        self.samples = samples
        self.numClasses = numClasses

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = self.samples[idx]

        # Convert image: (H, W, 3) uint8 -> (3, H, W) float32, normalized to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Convert mask: (H, W) int64 -> (1, H, W) float32
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Assign class_id (cycle through classes for demonstration)
        classId = idx % self.numClasses

        return {
            'image': img,
            'mask': mask,
            'class_id': classId
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate SAM-RS Few-Shot model')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataRoot', type=str, default='./data/loveda',
                        help='Root directory of dataset')

    # Optional arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation')
    parser.add_argument('--batchSize', type=int, default=8,
                        help='Batch size for evaluation')

    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()

    print("=" * 60)
    print("SAM-RS Few-Shot Evaluation")
    print("=" * 60)

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"\nLoading checkpoint: {args.checkpoint}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Extract configuration
    config = checkpoint['config']
    epoch = checkpoint['epoch']
    bestMiou = checkpoint.get('best_miou', 0.0)

    print(f"  Checkpoint epoch: {epoch}")
    print(f"  Best mIoU (during training): {bestMiou:.4f}")
    print()

    print(f"Configuration:")
    print(f"  SAM Model Type: {config.model.samModelType}")
    print(f"  LoRA Rank: {config.model.loraRank}")
    print(f"  Number of Prompts: {config.model.nPrompts}")
    print(f"  Number of Classes: {config.data.numClasses}")
    print(f"  Device: {args.device}")
    print()

    # TODO: Replace with real LoveDA test dataset loader
    # Currently using test data for demonstration
    print("Creating test dataset...")
    testSamples = create_sample_dataset(num_samples=20, numClasses=config.data.numClasses)
    print(f"  Test samples: {len(testSamples)}")
    print()

    # Create dataset and dataloader
    testDataset = SimpleDataset(testSamples, config.data.numClasses)
    testLoader = DataLoader(
        testDataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=0
    )

    print("Initializing models...")

    # Initialize SAM with LoRA
    model = SAMLoRA(
        samCheckpoint=config.model.samCheckpoint,
        modelType=config.model.samModelType,
        loraRank=config.model.loraRank,
        loraAlpha=config.model.loraAlpha,
        loraDropout=config.model.loraDropout
    )

    # Initialize Prompt Learner
    promptLearner = SimplePromptLearner(
        numClasses=config.data.numClasses,
        nPrompts=config.model.nPrompts,
        embedDim=256,
        initStd=config.model.promptInitStd
    )

    # Load model weights
    print("Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    promptLearner.load_state_dict(checkpoint['prompt_learner_state_dict'])

    # Move models to device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    promptLearner.to(device)

    # Set to evaluation mode
    model.eval()
    promptLearner.eval()

    print(f"Models loaded and set to evaluation mode on {device}")
    print()

    # Evaluation loop
    print("Starting evaluation...")
    print()

    totalMiou = 0.0
    perClassIous = [[] for _ in range(config.data.numClasses)]
    numBatches = 0

    with torch.no_grad():
        for batch in tqdm(testLoader, desc="Evaluating"):
            # Unpack batch
            images = batch['image'].to(device)  # (B, 3, H, W)
            masks = batch['mask'].to(device)    # (B, 1, H, W)
            classIds = batch['class_id'].to(device)  # (B,)

            # Forward pass
            # 1. Get prompts from prompt learner
            prompts = promptLearner(classIds)  # (B, nPrompts, embedDim)

            # 2. Average prompts to get single embedding per sample
            promptEmbeds = prompts.mean(dim=1)  # (B, embedDim)

            # 3. Pass through SAM model
            predMasks = model(images, promptEmbeds)  # (B, 1, H, W)

            # Convert predictions to class labels
            # predMasks contains logits, apply sigmoid and threshold
            predLabels = (torch.sigmoid(predMasks) > 0.5).long().squeeze(1)  # (B, H, W)
            targetLabels = masks.long().squeeze(1)  # (B, H, W)

            # Compute IoU
            batchMiou, classIous = compute_iou(predLabels, targetLabels, config.data.numClasses)

            # Accumulate metrics
            totalMiou += batchMiou
            numBatches += 1

            # Store per-class IoUs
            for classIdx, iou in enumerate(classIous):
                if not np.isnan(iou):
                    perClassIous[classIdx].append(iou)

    # Compute final metrics
    avgMiou = totalMiou / numBatches

    # Compute per-class average IoU
    perClassAvgIous = []
    for classIdx in range(config.data.numClasses):
        if len(perClassIous[classIdx]) > 0:
            avgIou = np.mean(perClassIous[classIdx])
            perClassAvgIous.append(avgIou)
        else:
            perClassAvgIous.append(np.nan)

    # Print results
    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print()
    print(f"Mean IoU (mIoU): {avgMiou:.4f}")
    print()
    print("Per-class IoU:")
    print("-" * 40)
    for classIdx, avgIou in enumerate(perClassAvgIous):
        if not np.isnan(avgIou):
            print(f"  Class {classIdx}: {avgIou:.4f}")
        else:
            print(f"  Class {classIdx}: N/A (no samples)")
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
