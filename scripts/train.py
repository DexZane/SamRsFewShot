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
from data.lovedaDataset import LoveDADataset
from data.transforms import DefaultTransform
from data.augmentedTransform import AugmentedTransform
from data.fewshotSampler import FewShotSampler




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
    parser.add_argument('--device', type=str, default='mps',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to use for training')
    parser.add_argument('--evalInterval', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--saveInterval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--checkpointDir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--logDir', type=str, default='./runs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--numWorkers', type=int, default=4,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--trainEpisodes', type=int, default=100,
                        help='Number of episodes per training epoch')

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

    # Create LoveDA datasets with transforms
    print("Loading LoveDA datasets...")
    # Phase 3.1: 训练集用增强版 transform，验证集用默认 transform
    trainTransform = AugmentedTransform(target_size=1024, train=True)
    valTransform = AugmentedTransform(target_size=1024, train=False)

    trainDataset = LoveDADataset(
        root=config.data.dataRoot,
        split='Train',
        download=False,  # Assume dataset is already downloaded
        transform=trainTransform
    )
    valDataset = LoveDADataset(
        root=config.data.dataRoot,
        split='Val',
        download=False,
        transform=valTransform
    )

    print(f"  Train samples: {len(trainDataset)}")
    print(f"  Val samples: {len(valDataset)}")
    print()

    # Extract labels from dataset for FewShotSampler
    # 直接读samples里已缓存的class_id：走__getitem__会把每张图都resize到1024
    # 再归一化一遍，只为拿一个整数，纯浪费启动时间
    print("Extracting labels for few-shot sampling...")
    trainLabels = [trainDataset.samples[i][2] for i in range(len(trainDataset))]
    valLabels = [valDataset.samples[i][2] for i in range(len(valDataset))]

    # Collate function to convert tuple to dict format
    def collate_fn(batch):
        """Convert list of (image, mask, class_id) tuples to dict format."""
        images = torch.stack([item[0] for item in batch])  # (B, 3, H, W)
        masks = torch.stack([item[1] for item in batch])   # (B, H, W)
        masks = masks.unsqueeze(1)  # (B, 1, H, W) - add channel dimension
        class_ids = torch.tensor([item[2] for item in batch])
        return {
            'image': images,
            'mask': masks,
            'class_id': class_ids
        }

    # Create few-shot samplers
    print("Creating few-shot samplers...")
    trainSampler = FewShotSampler(
        labels=trainLabels,
        nWay=config.data.nWay,
        kShot=config.data.kShot,
        nEpisodes=args.trainEpisodes,
        seed=42
    )
    valSampler = FewShotSampler(
        labels=valLabels,
        nWay=config.data.nWay,
        kShot=config.data.kShot,
        nEpisodes=20,
        seed=123
    )

    # Create dataloaders with batch samplers
    useCuda = args.device == 'cuda'
    trainLoader = DataLoader(
        trainDataset,
        batch_sampler=trainSampler,
        num_workers=args.numWorkers,
        collate_fn=collate_fn,
        pin_memory=useCuda,
        persistent_workers=(args.numWorkers > 0)
    )
    valLoader = DataLoader(
        valDataset,
        batch_sampler=valSampler,
        num_workers=args.numWorkers,
        collate_fn=collate_fn,
        pin_memory=useCuda,
        persistent_workers=(args.numWorkers > 0)
    )

    print("Initializing models...")

    # Initialize SAM with LoRA
    model = SAMLoRA(
        samCheckpoint=config.model.samCheckpoint,
        model_type=config.model.samModelType,
        loraRank=config.model.loraRank,
        loraAlpha=config.model.loraAlpha,
        loraDropout=config.model.loraDropout
    )

    # Initialize Prompt Learner
    promptLearner = SimplePromptLearner(
        nClasses=config.data.numClasses,
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
