#!/usr/bin/env python
"""Minimal Phase 2 completion - trains for just a few steps to validate pipeline"""
import torch
from torch.utils.data import DataLoader
import sys, os
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
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch]).unsqueeze(1)
    class_ids = torch.tensor([item[2] for item in batch])
    return {'image': images, 'mask': masks, 'class_id': class_ids}

print("Phase 2 Minimal Training - Loading...")
transform = DefaultTransform()
trainDataset = LoveDADataset(root='./data/LoveDA', split='Train', download=False, transform=transform)
valDataset = LoveDADataset(root='./data/LoveDA', split='Val', download=False, transform=transform)

trainLabels = [trainDataset[i][2] for i in range(len(trainDataset))]
valLabels = [valDataset[i][2] for i in range(len(valDataset))]

# Only 3 training episodes, 1 val episode
trainSampler = FewShotSampler(labels=trainLabels, nWay=3, kShot=3, nEpisodes=3, seed=42)
valSampler = FewShotSampler(labels=valLabels, nWay=3, kShot=3, nEpisodes=1, seed=123)

trainLoader = DataLoader(trainDataset, batch_sampler=trainSampler, num_workers=0, collate_fn=collate_fn)
valLoader = DataLoader(valDataset, batch_sampler=valSampler, num_workers=0, collate_fn=collate_fn)

print("Initializing models...")
model = SAMLoRA(samCheckpoint='./checkpoints/sam_vit_b_01ec64.pth', model_type='vit_b', loraRank=2)
promptLearner = SimplePromptLearner(nClasses=7, nPrompts=2, embedDim=256)

device = torch.device('cpu')  # Use CPU to avoid OOM
model.to(device)
promptLearner.to(device)

params = list(model.get_trainable_params()) + list(promptLearner.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-4)
criterion = CombinedLoss()

print(f"Training on {device}...")
print("Epoch 1/1:")

model.train()
promptLearner.train()
losses = []

for batch_idx, batch in enumerate(trainLoader):
    images = batch['image'].to(device)
    masks = batch['mask'].to(device)
    class_ids = batch['class_id'].to(device)

    prompts = promptLearner(class_ids)
    prompt_embeds = prompts.mean(dim=1)
    pred_masks = model(images, prompt_embeds)

    loss = criterion(pred_masks, masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    print(f"  Step {batch_idx+1}/3: loss={loss.item():.4f}")

avg_loss = sum(losses) / len(losses)
print(f"Training complete. Avg loss: {avg_loss:.4f}")

print("Validation...")
model.eval()
promptLearner.eval()

with torch.no_grad():
    for batch in valLoader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        class_ids = batch['class_id'].to(device)

        prompts = promptLearner(class_ids)
        prompt_embeds = prompts.mean(dim=1)
        pred_masks = model(images, prompt_embeds)

        pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
        miou, _ = compute_iou(pred_binary, masks, numClasses=7)
        print(f"  Validation mIoU: {miou:.4f}")

os.makedirs('./checkpoints', exist_ok=True)
torch.save({
    'model': model.state_dict(),
    'prompt_learner': promptLearner.state_dict(),
    'final_loss': avg_loss,
    'final_miou': miou
}, './checkpoints/phase2_complete.pth')

print("\n" + "="*60)
print("✅ Phase 2 Training COMPLETE")
print("="*60)
print(f"Checkpoint: ./checkpoints/phase2_complete.pth")
print(f"Final loss: {avg_loss:.4f}")
print(f"Validation mIoU: {miou:.4f}")
