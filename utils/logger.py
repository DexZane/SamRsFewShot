"""
Logger module for training and evaluation

Provides logging functionality with file output and TensorBoard integration.
"""

import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger class with file and TensorBoard support

    Args:
        logDir: Directory for log files
        experimentName: Name of the experiment
    """

    def __init__(self, logDir, experimentName="experiment"):
        self.logDir = logDir
        self.experimentName = experimentName

        # Create log directory
        os.makedirs(logDir, exist_ok=True)

        # Setup file logger
        logFile = os.path.join(logDir, "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logFile),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experimentName)

        # Setup TensorBoard writer
        tbDir = os.path.join(logDir, experimentName)
        self.writer = SummaryWriter(tbDir)

        self.info(f"Logger initialized. Log dir: {logDir}")
        self.info(f"TensorBoard dir: {tbDir}")

    def info(self, msg):
        """Log info message"""
        self.logger.info(msg)

    def log_scalar(self, tag, value, step):
        """
        Log scalar value to TensorBoard

        Args:
            tag: Tag name (e.g., 'train/loss', 'val/mIoU')
            value: Scalar value to log
            step: Global step (epoch or iteration)
        """
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """Close the logger and TensorBoard writer"""
        self.writer.close()
        self.info("Logger closed")
