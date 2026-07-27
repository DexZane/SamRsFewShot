"""
Training script for SAM-RS Few-Shot Learning

Usage:
    python scripts/train.py --dataRoot ./data/loveda --samCheckpoint path/to/sam_vit_b.pth
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

from config.default import Config, ModelConfig, DataConfig, TrainingConfig
from models.samLora import SAMLoRA
from models.promptLearner import SimplePromptLearner
from training.trainer import Trainer
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
    parser = argparse.ArgumentParser(description='Train SAM-RS Few-Shot model')

    # Data arguments
    parser.add_argument('--dataRoot', type=str, default='./data/loveda',
                        help='Root directory of dataset')
    parser.add_argument('--numClasses', type=int, default=7,
                        help='Number of semantic classes')

    # Model arguments
    parser.add_argument('--samCheckpoint', type=str, required=True,
                        help='Path to SAM checkpoint')
    parser.add_argument('--samModelType', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM model type')
    parser.add_argument('--loraRank', type=int, default=4,
                        help='LoRA rank')
    parser.add_argument('--nPrompts', type=int, default=5,
                        help='Number of learnable prompts per class')

    # Few-shot arguments
    parser.add_argument('--nWay', type=int, default=5,
                        help='Number of classes per episode')
    parser.add_argument('--kShot', type=int, default=5,
                        help='Number of support samples per class')

    # Training arguments
    parser.add_argument('--numEpochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batchSize', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--evalInterval', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--saveInterval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--checkpointDir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--logDir', type=str, default='./runs',
                        help='Directory for TensorBoard logs')

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    print("=" * 60)
    print("SAM-RS Few-Shot Training")
    print("=" * 60)

    # Initialize configuration
    config = Config()

    # Update config with command line arguments
    config.model.samCheckpoint = args.samCheckpoint
    config.model.samModelType = args.samModelType
    config.model.loraRank = args.loraRank
    config.model.nPrompts = args.nPrompts

    config.data.dataRoot = args.dataRoot
    config.data.numClasses = args.numClasses
    config.data.nWay = args.nWay
    config.data.kShot = args.kShot

    config.training.numEpochs = args.numEpochs
    config.training.batchSize = args.batchSize
    config.training.learningRate = args.lr
    config.training.device = args.device
    config.training.evalInterval = args.evalInterval
    config.training.saveInterval = args.saveInterval
    config.training.checkpointDir = args.checkpointDir
    config.training.logDir = args.logDir

    print(f"\nConfiguration:")
    print(f"  SAM Checkpoint: {config.model.samCheckpoint}")
    print(f"  SAM Model Type: {config.model.samModelType}")
    print(f"  LoRA Rank: {config.model.loraRank}")
    print(f"  Number of Prompts: {config.model.nPrompts}")
    print(f"  Number of Classes: {config.data.numClasses}")
    print(f"  N-way K-shot: {config.data.nWay}-way {config.data.kShot}-shot")
    print(f"  Batch Size: {config.training.batchSize}")
    print(f"  Learning Rate: {config.training.learningRate}")
    print(f"  Number of Epochs: {config.training.numEpochs}")
    print(f"  Device: {config.training.device}")
    print()

    # TODO: Replace with real LoveDA dataset loader
    # Currently using test data for demonstration
    print("Creating test datasets...")
    trainSamples = create_sample_dataset(num_samples=40, numClasses=config.data.numClasses)
    valSamples = create_sample_dataset(num_samples=10, numClasses=config.data.numClasses)
    print(f"  Train samples: {len(trainSamples)}")
    print(f"  Val samples: {len(valSamples)}")
    print()

    # Create datasets
    trainDataset = SimpleDataset(trainSamples, config.data.numClasses)
    valDataset = SimpleDataset(valSamples, config.data.numClasses)

    # Create dataloaders
    trainLoader = DataLoader(
        trainDataset,
        batch_size=config.training.batchSize,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    valLoader = DataLoader(
        valDataset,
        batch_size=config.training.batchSize,
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
        embedDim=256,  # SAM image encoder output dimension
        initStd=config.model.promptInitStd
    )

    # Print model statistics
    print("\nModel Statistics:")
    print(f"  Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable model parameters: {sum(p.numel() for p in model.get_trainable_params()):,}")
    print(f"  Prompt learner parameters: {sum(p.numel() for p in promptLearner.parameters()):,}")
    totalTrainable = sum(p.numel() for p in model.get_trainable_params()) + sum(p.numel() for p in promptLearner.parameters())
    print(f"  Total trainable parameters: {totalTrainable:,}")
    print()

    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        config=config,
        model=model,
        promptLearner=promptLearner,
        trainLoader=trainLoader,
        valLoader=valLoader
    )

    # Start training
    print("\nStarting training loop...\n")
    trainer.train()

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Best validation mIoU: {trainer.bestMiou:.4f}")
    print(f"Checkpoints saved to: {config.training.checkpointDir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
