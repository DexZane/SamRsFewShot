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

        # Setup learning rate scheduler (Cosine Annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.numEpochs,
            eta_min=config.training.learningRate * 0.01  # 最低降到初始lr的1%
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

        # Early stopping
        self.patience = getattr(config.training, 'patience', 10)  # 默认10个epoch不涨就停
        self.patienceCounter = 0

        self.logger.info("Trainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Optimizer: AdamW (lr={config.training.learningRate})")
        self.logger.info(f"Scheduler: CosineAnnealingLR (T_max={config.training.numEpochs}, eta_min={config.training.learningRate * 0.01:.6f})")
        self.logger.info(f"Early stopping: patience={self.patience}")
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

        # YOLO-style progress bar
        pbar = tqdm(
            self.trainLoader,
            desc=f"\033[1m{'Train':<8}\033[0m",
            bar_format='{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            ncols=120
        )

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
            avgLossSoFar = epochLoss / (batchIdx + 1)

            # YOLO-style postfix with key metrics highlighted
            currentLR = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(
                f"\033[36mEpoch {epoch}/{self.config.training.numEpochs}\033[0m | "
                f"\033[33mloss {loss.item():.4f}\033[0m | "
                f"\033[32mavg {avgLossSoFar:.4f}\033[0m | "
                f"lr {currentLR:.6f}"
            )

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

        # YOLO-style validation bar
        pbar = tqdm(
            self.valLoader,
            desc=f"\033[1m{'Val':<8}\033[0m",
            bar_format='{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
            ncols=120
        )

        with torch.no_grad():
            for batchIdx, batch in enumerate(pbar):
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

                # Update progress bar
                avgMiouSoFar = totalMiou / (batchIdx + 1)
                pbar.set_postfix_str(f"\033[35mmIoU {avgMiouSoFar:.4f}\033[0m")

        # Compute average mIoU
        avgMiou = totalMiou / numBatches

        # Log to TensorBoard
        self.logger.log_scalar('val/mIoU', avgMiou, epoch)
        self.logger.info(f"Epoch {epoch}: val_mIoU = {avgMiou:.4f}")

        # Save best model and update early stopping counter
        if avgMiou > self.bestMiou:
            self.bestMiou = avgMiou
            self.patienceCounter = 0  # 重置计数器
            self.save_checkpoint(epoch, is_best=True)
            self.logger.info(f"New best mIoU: {self.bestMiou:.4f}")
        else:
            self.patienceCounter += 1
            self.logger.info(f"No improvement for {self.patienceCounter} validation(s)")

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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.bestMiou,
            'patience_counter': self.patienceCounter,
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
        Main training loop with early stopping
        """
        print("\n" + "="*80)
        print(f"\033[1;36m{'SAM-RS Few-Shot Training':^80}\033[0m")
        print("="*80)
        print(f"  Device       : {self.device}")
        print(f"  Epochs       : {self.config.training.numEpochs}")
        print(f"  Batch Size   : {self.config.training.batchSize} ({self.config.data.nWay}-way {self.config.data.kShot}-shot)")
        print(f"  Learning Rate: {self.config.training.learningRate:.6f}")
        print(f"  LoRA Dropout : {self.config.model.loraDropout}")
        print(f"  Early Stop   : patience={self.patience}")
        print("="*80 + "\n")

        self.logger.info("Starting training...")

        for epoch in range(1, self.config.training.numEpochs + 1):
            # Train for one epoch
            avgLoss = self.train_epoch(epoch)

            # Step scheduler (update learning rate)
            self.scheduler.step()
            currentLr = self.scheduler.get_last_lr()[0]
            self.logger.log_scalar('train/lr', currentLr, epoch)
            self.logger.info(f"Epoch {epoch}: learning rate = {currentLr:.6f}")

            # Validate at specified intervals
            if epoch % self.config.training.evalInterval == 0:
                avgMiou = self.validate(epoch)

                # Early stopping check
                if self.patienceCounter >= self.patience:
                    print(f"\n\033[1;33m⚠ Early stopping triggered after {epoch} epochs\033[0m")
                    print(f"  No improvement for {self.patience} consecutive validations\n")
                    self.logger.info(f"Early stopping triggered after {epoch} epochs")
                    self.logger.info(f"No improvement for {self.patience} consecutive validations")
                    break

            # Save checkpoint at specified intervals
            if epoch % self.config.training.saveInterval == 0:
                self.save_checkpoint(epoch, is_best=False)

        print("\n" + "="*80)
        print(f"\033[1;32m{'Training Completed!':^80}\033[0m")
        print("="*80)
        print(f"  Best mIoU: \033[1;32m{self.bestMiou:.4f}\033[0m")
        print("="*80 + "\n")

        self.logger.info("Training completed!")
        self.logger.info(f"Best validation mIoU: {self.bestMiou:.4f}")
        self.logger.close()
