"""
Trainer module for SAM-RS Few-Shot Learning

Implements the training loop with validation, checkpointing, and logging.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from models.loss import CombinedLoss
from utils.logger import Logger
from utils.metrics import compute_iou


def _make_grad_scaler(enabled):
    """构造GradScaler，兼容不同PyTorch版本的API

    torch.amp.GradScaler 从 2.3 才存在；2.2 及更早只有 torch.cuda.amp.GradScaler
    （在新版本里该路径会发 FutureWarning，所以优先用新API）。
    """
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    """
    Trainer for SAM-RS Few-Shot Learning

    Args:
        config: Configuration object
        model: SAMLoRA model
        promptLearner: SimplePromptLearner model
        trainLoader: Training data loader
        valLoader: Validation data loader
    """

    def __init__(self, config, model, promptLearner, trainLoader, valLoader):
        self.config = config
        self.model = model
        self.promptLearner = promptLearner
        self.trainLoader = trainLoader
        self.valLoader = valLoader

        # Setup device
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.promptLearner.to(self.device)

        # 混合精度：仅CUDA可用。GradScaler在非CUDA时自动退化为no-op
        self.useAmp = self.device.type == "cuda"
        self.scaler = _make_grad_scaler(self.useAmp)

        # Setup optimizer (optimize both model and prompt_learner)
        modelParams = list(self.model.get_trainable_params())
        promptParams = list(self.promptLearner.parameters())
        allParams = modelParams + promptParams

        self.optimizer = AdamW(
            allParams,
            lr=config.training.learningRate,
            weight_decay=config.training.weightDecay
        )

        # Setup loss function
        self.criterion = CombinedLoss()

        # Setup logger
        self.logger = Logger(
            logDir=config.training.logDir,
            experimentName="sam_rs_fewshot"
        )

        # Create checkpoint directory
        os.makedirs(config.training.checkpointDir, exist_ok=True)

        # Track best validation mIoU
        self.bestMiou = 0.0

        self.logger.info("Trainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Optimizer: AdamW (lr={config.training.learningRate})")
        self.logger.info(f"Total trainable parameters: {sum(p.numel() for p in allParams)}")

    def _build_binary_targets(self, masks, classIds):
        """把7类语义标签图转成"当前episode类别 vs 其它"的二值目标

        模型输出单通道mask，训练目标必须是二值的：episode指定的类别为前景1，
        其它类别为背景0，原始的255（padding/ignore）不参与损失。

        Args:
            masks: (B, 1, H, W) int64，取值 [0, numClasses-1] 或 255
            classIds: (B,) int64，每个样本本episode的目标类别

        Returns:
            targets: (B, 1, H, W) float32，前景1/背景0
            validMask: (B, 1, H, W) bool，True表示该像素参与损失
        """
        classIds = classIds.view(-1, 1, 1, 1).to(masks.device)

        validMask = masks != 255
        targets = ((masks == classIds) & validMask).float()

        return targets, validMask

    def train_epoch(self, epoch):
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            avgLoss: Average loss for the epoch
        """
        self.model.train()
        self.promptLearner.train()

        epochLoss = 0.0
        numBatches = len(self.trainLoader)

        # Progress bar
        pbar = tqdm(self.trainLoader, desc=f"Epoch {epoch}/{self.config.training.numEpochs}")

        for batchIdx, batch in enumerate(pbar):
            # Unpack batch
            images = batch['image'].to(self.device)  # (B, 3, H, W)
            masks = batch['mask'].to(self.device)    # (B, 1, H, W)
            classIds = batch['class_id'].to(self.device)  # (B,)

            # Forward + backward pass（混合精度包裹整个计算图，让checkpoint重计算也在fp16下执行）
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.useAmp):
                # 1. Get prompts from prompt learner
                prompts = self.promptLearner(classIds)  # (B, nPrompts, embedDim)

                # 2. Average prompts to get single embedding per sample
                promptEmbeds = prompts.mean(dim=1)  # (B, embedDim)

                # 3. Pass through SAM model
                predMasks = self.model(images, promptEmbeds)  # (B, 1, H, W)

                # 把7类标签图转成本episode类别的二值前景，255为ignore
                targets, validMask = self._build_binary_targets(masks, classIds)

                # 损失强制在fp32下计算：Dice的除法和BCE的log在fp16下容易溢出
                # 用autocast(enabled=False)临时退出混合精度，确保loss稳定
                with torch.autocast(device_type=self.device.type, enabled=False):
                    loss = self.criterion(predMasks.float(), targets.float(), validMask.float())

                # Backward pass仍在外层autocast下，checkpoint重计算使用fp16
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Update metrics
            epochLoss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute average loss
        avgLoss = epochLoss / numBatches

        # Log to TensorBoard
        self.logger.log_scalar('train/loss', avgLoss, epoch)
        self.logger.info(f"Epoch {epoch}: train_loss = {avgLoss:.4f}")

        return avgLoss

    def validate(self, epoch):
        """
        Validate the model

        Args:
            epoch: Current epoch number

        Returns:
            miou: Mean IoU across validation set
        """
        self.model.eval()
        self.promptLearner.eval()

        totalMiou = 0.0
        numBatches = len(self.valLoader)

        with torch.no_grad():
            for batch in tqdm(self.valLoader, desc="Validation"):
                # Unpack batch
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                classIds = batch['class_id'].to(self.device)

                # Forward pass（与训练一致的混合精度）
                with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                    enabled=self.useAmp):
                    prompts = self.promptLearner(classIds)
                    promptEmbeds = prompts.mean(dim=1)
                    predMasks = self.model(images, promptEmbeds)

                # 二值前景/背景评估：模型只输出单通道，按7类算IoU没有意义
                targets, validMask = self._build_binary_targets(masks, classIds)

                predLabels = (torch.sigmoid(predMasks.float()) > 0.5).long()  # (B, 1, H, W)
                targetLabels = targets.long()

                # ignore区域排除：预测和标签都置为背景，不影响前景IoU
                predLabels = predLabels * validMask.long()
                targetLabels = targetLabels * validMask.long()

                # 二值IoU：类别0=背景，类别1=目标前景
                miou, _ = compute_iou(predLabels.squeeze(1), targetLabels.squeeze(1), numClasses=2)
                totalMiou += miou

        # Compute average mIoU
        avgMiou = totalMiou / numBatches

        # Log to TensorBoard
        self.logger.log_scalar('val/mIoU', avgMiou, epoch)
        self.logger.info(f"Epoch {epoch}: val_mIoU = {avgMiou:.4f}")

        # Save best model
        if avgMiou > self.bestMiou:
            self.bestMiou = avgMiou
            self.save_checkpoint(epoch, is_best=True)
            self.logger.info(f"New best mIoU: {self.bestMiou:.4f}")

        return avgMiou

    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'prompt_learner_state_dict': self.promptLearner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.bestMiou,
            'config': self.config
        }

        # Save regular checkpoint
        checkpointPath = os.path.join(
            self.config.training.checkpointDir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpointPath)
        self.logger.info(f"Checkpoint saved: {checkpointPath}")

        # Save best model
        if is_best:
            bestPath = os.path.join(
                self.config.training.checkpointDir,
                'best_model.pth'
            )
            torch.save(checkpoint, bestPath)
            self.logger.info(f"Best model saved: {bestPath}")

    def train(self):
        """
        Main training loop
        """
        self.logger.info("Starting training...")

        for epoch in range(1, self.config.training.numEpochs + 1):
            # Train for one epoch
            avgLoss = self.train_epoch(epoch)

            # Validate at specified intervals
            if epoch % self.config.training.evalInterval == 0:
                avgMiou = self.validate(epoch)

            # Save checkpoint at specified intervals
            if epoch % self.config.training.saveInterval == 0:
                self.save_checkpoint(epoch, is_best=False)

        self.logger.info("Training completed!")
        self.logger.info(f"Best validation mIoU: {self.bestMiou:.4f}")
        self.logger.close()
