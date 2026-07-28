#!/usr/bin/env python
"""
Quick Phase 2 validation script - completes training pipeline with minimal iterations
"""
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from config.default import Config
from models.samLora import SAMLoRA
from models.promptLearner import SimplePromptLearner
from models.loss import CombinedLoss
from data.lovedaDataset import LoveDADataset
from data.transforms import DefaultTransform
from data.fewshotSampler import FewShotSampler
from utils.metrics import compute_iou

def collate_fn(batch):
    """Convert list of (image, mask, class_id) tuples to dict format."""
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    masks = masks.unsqueeze(1)  # Add channel dimension
    class_ids = torch.tensor([item[2] for item in batch])
    return {
        'image': images,
        'mask': masks,
        'class_id': class_ids
    }

def main():
    print("=" * 60)
    print("Phase 2 Quick Validation Training")
    print("=" * 60)

    # Minimal config
    config = Config()
    config.model.samCheckpoint = './checkpoints/sam_vit_b_01ec64.pth'
    config.model.loraRank = 2
    config.model.nPrompts = 2
    config.data.nWay = 3
    config.data.kShot = 3
    config.training.device = 'mps'
    config.training.learningRate = 1e-4

    # Load dataset
    print("\nLoading datasets...")
    transform = DefaultTransform()
    trainDataset = LoveDADataset(root='./data/LoveDA', split='Train', download=False, transform=transform)
    valDataset = LoveDADataset(root='./data/LoveDA', split='Val', download=False, transform=transform)
    print(f"  Train: {len(trainDataset)} samples")
    print(f"  Val: {len(valDataset)} samples")

    # Extract labels
    trainLabels = [trainDataset[i][2] for i in range(len(trainDataset))]
    valLabels = [valDataset[i][2] for i in range(len(valDataset))]

    # Create samplers with minimal episodes
    trainSampler = FewShotSampler(labels=trainLabels, nWay=3, kShot=3, nEpisodes=5, seed=42)
    valSampler = FewShotSampler(labels=valLabels, nWay=3, kShot=3, nEpisodes=2, seed=123)

    trainLoader = DataLoader(trainDataset, batch_sampler=trainSampler, num_workers=0, collate_fn=collate_fn)
    valLoader = DataLoader(valDataset, batch_sampler=valSampler, num_workers=0, collate_fn=collate_fn)

    # Initialize models
    print("\nInitializing models...")
    model = SAMLoRA(samCheckpoint=config.model.samCheckpoint, model_type='vit_b', loraRank=2)
    promptLearner = SimplePromptLearner(nClasses=7, nPrompts=2, embedDim=256)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    promptLearner.to(device)

    # Setup optimizer and loss
    params = list(model.get_trainable_params()) + list(promptLearner.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    criterion = CombinedLoss()

    print(f"  Device: {device}")
    print(f"  Trainable params: {sum(p.numel() for p in params):,}")

    # Training loop (2 epochs, 5 iterations each)
    print("\nStarting training...")
    num_epochs = 2

    for epoch in range(1, num_epochs + 1):
        model.train()
        promptLearner.train()

        print(f"\nEpoch {epoch}/{num_epochs}:")
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(trainLoader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            class_ids = batch['class_id'].to(device)

            # Forward
            prompts = promptLearner(class_ids)
            prompt_embeds = prompts.mean(dim=1)
            pred_masks = model(images, prompt_embeds)

            # Loss
            loss = criterion(pred_masks, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"  Batch {batch_idx + 1}/5: loss={loss.item():.4f}")

        avg_loss = epoch_loss / 5
        print(f"  Epoch {epoch} avg loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        promptLearner.eval()

        val_ious = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(valLoader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                class_ids = batch['class_id'].to(device)

                prompts = promptLearner(class_ids)
                prompt_embeds = prompts.mean(dim=1)
                pred_masks = model(images, prompt_embeds)

                # Compute IoU
                pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                iou = compute_iou(pred_binary, masks)
                val_ious.append(iou.item())

        avg_iou = sum(val_ious) / len(val_ious) if val_ious else 0.0
        print(f"  Validation mIoU: {avg_iou:.4f}")

    # Save final checkpoint
    checkpoint_path = './checkpoints/phase2_validation_complete.pth'
    torch.save({
        'model': model.state_dict(),
        'prompt_learner': promptLearner.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': num_epochs,
        'final_loss': avg_loss,
        'final_miou': avg_iou
    }, checkpoint_path)

    print("\n" + "=" * 60)
    print("Phase 2 Training Validation COMPLETE")
    print("=" * 60)
    print(f"Final checkpoint saved: {checkpoint_path}")
    print(f"Final training loss: {avg_loss:.4f}")
    print(f"Final validation mIoU: {avg_iou:.4f}")
    print("\nPhase 2 training pipeline successfully validated!")

if __name__ == '__main__':
    main()
