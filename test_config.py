"""Test configuration management."""
import sys
sys.path.insert(0, '/Users/dexzane/Desktop/FindProject/samRsFewShot')

from config import Config, ModelConfig, DataConfig, TrainingConfig


def test_default_config():
    """Test default configuration initialization."""
    cfg = Config()

    # Check model config defaults
    assert cfg.model.samCheckpoint is None
    assert cfg.model.samModelType == "vit_b"
    assert cfg.model.loraRank == 4
    assert cfg.model.loraAlpha == 8
    assert cfg.model.loraDropout == 0.1
    assert cfg.model.nPrompts == 5
    assert cfg.model.promptInitStd == 0.02

    # Check data config defaults
    assert cfg.data.dataRoot == "./data/loveda"
    assert cfg.data.numClasses == 7
    assert cfg.data.imageSize == 1024
    assert cfg.data.nWay == 5
    assert cfg.data.kShot == 5
    assert cfg.data.nQuery == 5
    assert cfg.data.nEpisodes == 100

    # Check training config defaults
    assert cfg.training.batchSize == 8
    assert cfg.training.numEpochs == 50
    assert cfg.training.learningRate == 1e-4
    assert cfg.training.weightDecay == 1e-4
    assert cfg.training.warmupEpochs == 5
    assert cfg.training.evalInterval == 5
    assert cfg.training.saveInterval == 10
    assert cfg.training.checkpointDir == "./checkpoints"
    assert cfg.training.logDir == "./runs"
    assert cfg.training.device == "cuda"
    assert cfg.training.numWorkers == 4

    print("OK Default config test passed")


def test_custom_config():
    """Test custom configuration."""
    model_cfg = ModelConfig(loraRank=8, nPrompts=10)
    data_cfg = DataConfig(numClasses=10, nWay=3, kShot=3)
    training_cfg = TrainingConfig(batchSize=8, numEpochs=100)

    cfg = Config(model=model_cfg, data=data_cfg, training=training_cfg)

    assert cfg.model.loraRank == 8
    assert cfg.model.nPrompts == 10
    assert cfg.data.numClasses == 10
    assert cfg.data.nWay == 3
    assert cfg.training.batchSize == 8
    assert cfg.training.numEpochs == 100

    print("OK Custom config test passed")


def test_validation_nway():
    """Test validation: nWay <= numClasses."""
    try:
        data_cfg = DataConfig(numClasses=5, nWay=10)
        cfg = Config(data=data_cfg)
        assert False, "Should have raised assertion error"
    except AssertionError as e:
        assert "nWay" in str(e) and "numClasses" in str(e)
        print(f"OK nWay validation test passed: {e}")


def test_validation_batch_size():
    """Test validation: batchSize >= nWay."""
    try:
        data_cfg = DataConfig(numClasses=10, nWay=8)
        training_cfg = TrainingConfig(batchSize=4)
        cfg = Config(data=data_cfg, training=training_cfg)
        assert False, "Should have raised assertion error"
    except AssertionError as e:
        assert "batchSize" in str(e) and "nWay" in str(e)
        print(f"OK Batch size validation test passed: {e}")


if __name__ == "__main__":
    test_default_config()
    test_custom_config()
    test_validation_nway()
    test_validation_batch_size()
    print("\nAll configuration tests passed!")
