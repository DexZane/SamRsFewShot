"""Default configuration for SAM-RS Few-Shot segmentation."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    samCheckpoint: Optional[str] = None
    samModelType: str = "vit_b"
    loraRank: int = 8  # Phase 3.2: 增大 LoRA rank
    loraAlpha: int = 16  # alpha = 2 * rank (保持比例)
    loraDropout: float = 0.2  # 保持 Phase 3.1 的 dropout
    nPrompts: int = 10  # Phase 3.2: 增加 prompt 数量
    promptInitStd: float = 0.02


@dataclass
class DataConfig:
    """Data configuration."""
    dataRoot: str = "./data/loveda"
    numClasses: int = 7
    imageSize: int = 1024
    nWay: int = 5
    kShot: int = 5
    nQuery: int = 5
    nEpisodes: int = 100


@dataclass
class TrainingConfig:
    """Training configuration."""
    batchSize: int = 8
    numEpochs: int = 50
    learningRate: float = 1e-4
    weightDecay: float = 1e-4
    warmupEpochs: int = 5
    evalInterval: int = 5
    saveInterval: int = 10
    checkpointDir: str = "./checkpoints"
    logDir: str = "./runs"
    device: str = "cuda"  # NVIDIA GPU；Apple Silicon 本地调试可传 --device mps
    numWorkers: int = 4


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        """Initialize sub-configs and validate configuration."""
        # Initialize sub-configs if not provided
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()

        # Validate configuration
        assert self.data.nWay <= self.data.numClasses, (
            f"nWay ({self.data.nWay}) must be <= numClasses ({self.data.numClasses})"
        )
        assert self.training.batchSize >= self.data.nWay, (
            f"batchSize ({self.training.batchSize}) must be >= nWay ({self.data.nWay})"
        )
