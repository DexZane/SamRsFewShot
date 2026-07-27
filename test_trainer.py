"""Test trainer implementation"""
import sys
sys.path.insert(0, '/Users/dexzane/Desktop/FindProject/samRsFewShot')

import torch
from config import Config
from models.samLora import SAMLoRA
from models.promptLearner import SimplePromptLearner
from utils.logger import Logger
from utils.metrics import compute_iou
from training.trainer import Trainer

def test_imports():
    """Test that all imports work"""
    print("OK All imports successful")

def test_logger():
    """Test Logger class"""
    import tempfile
    import shutil

    tmpDir = tempfile.mkdtemp()
    try:
        logger = Logger(tmpDir, "test_exp")
        logger.info("Test message")
        logger.log_scalar("test/metric", 0.5, 1)
        logger.close()
        print("OK Logger test passed")
    finally:
        shutil.rmtree(tmpDir)

def test_metrics():
    """Test compute_iou function"""
    # Create dummy predictions and targets
    pred = torch.tensor([
        [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [2, 2, 3, 3],
         [2, 2, 3, 3]]
    ])  # (1, 4, 4)

    target = torch.tensor([
        [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [2, 2, 3, 3],
         [2, 2, 3, 3]]
    ])  # (1, 4, 4)

    miou, perClassIou = compute_iou(pred, target, numClasses=4)
    print(f"  mIoU: {miou:.4f}")
    print(f"  Per-class IoU: {perClassIou}")
    assert miou == 1.0, "Perfect prediction should have mIoU=1.0"
    print("OK Metrics test passed")

def test_trainer_init():
    """Test Trainer initialization"""
    # Create config
    cfg = Config()

    # Create models (test mode, no checkpoint)
    model = SAMLoRA(samCheckpoint=None, loraRank=4)
    promptLearner = SimplePromptLearner(
        nClasses=cfg.data.numClasses,
        nPrompts=cfg.model.nPrompts,
        embedDim=model.embedDim
    )

    # Create dummy dataloaders
    class DummyDataset:
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 1024, 1024),
                'mask': torch.randint(0, 2, (1, 256, 256)).float(),
                'class_id': torch.tensor(0)
            }

    from torch.utils.data import DataLoader
    trainLoader = DataLoader(DummyDataset(), batch_size=2)
    valLoader = DataLoader(DummyDataset(), batch_size=2)

    # Initialize trainer
    import tempfile
    import shutil
    tmpDir = tempfile.mkdtemp()

    try:
        cfg.training.logDir = tmpDir
        cfg.training.checkpointDir = tmpDir
        cfg.training.device = "cpu"  # Use CPU for testing

        trainer = Trainer(cfg, model, promptLearner, trainLoader, valLoader)
        print(f"  Device: {trainer.device}")
        print(f"  Best mIoU: {trainer.bestMiou}")
        trainer.logger.close()
        print("OK Trainer initialization test passed")
    finally:
        shutil.rmtree(tmpDir)

if __name__ == "__main__":
    print("Testing trainer implementation...\n")

    test_imports()
    test_logger()
    test_metrics()
    test_trainer_init()

    print("\nAll tests passed!")
