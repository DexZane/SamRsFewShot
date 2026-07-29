"""CSV logger for structured training metrics storage"""
import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional


class CSVLogger:
    """记录训练指标到CSV文件，便于后续分析和对比实验"""

    def __init__(self, save_dir: str, experiment_name: str):
        """
        Args:
            save_dir: CSV文件保存目录
            experiment_name: 实验名称（用于文件名）
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(save_dir, f"{experiment_name}_{timestamp}.csv")

        # CSV字段
        self.fieldnames = [
            'epoch',
            'train_loss',
            'val_miou',
            'learning_rate',
            'best_miou',
            'patience_counter',
            'timestamp'
        ]

        # 创建CSV文件并写入表头
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

        print(f"[CSV Logger] Metrics will be saved to: {self.csv_path}")

    def log_epoch(self, metrics: Dict[str, Any]):
        """
        记录单个epoch的指标

        Args:
            metrics: 包含epoch指标的字典，必须包含'epoch'键
                    可选键: 'train_loss', 'val_miou', 'learning_rate', 'best_miou', 'patience_counter'
        """
        # 确保所有字段都存在（缺失的填None）
        row = {field: metrics.get(field, None) for field in self.fieldnames}

        # 添加时间戳
        if 'timestamp' not in metrics:
            row['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 追加到CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def save_summary(self, summary: Dict[str, Any]):
        """
        保存训练总结（最终指标）

        Args:
            summary: 总结信息字典
        """
        summary_path = self.csv_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"Training Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n")
            for key, value in summary.items():
                f.write(f"{key:30s}: {value}\n")
            f.write("="*60 + "\n")

        print(f"[CSV Logger] Summary saved to: {summary_path}")

    def get_csv_path(self) -> str:
        """返回CSV文件路径"""
        return self.csv_path
