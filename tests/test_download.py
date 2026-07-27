import os
import tempfile
import pytest
from data.download import _verify_dataset


def test_verify_dataset_missing():
    """测试验证函数对缺失目录的检测"""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert _verify_dataset(tmpdir) == False


def test_verify_dataset_incomplete():
    """测试验证函数对不完整目录的检测"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 只创建部分目录
        os.makedirs(os.path.join(tmpdir, 'Train/Urban/images_png'))
        assert _verify_dataset(tmpdir) == False
