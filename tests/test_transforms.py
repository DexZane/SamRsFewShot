import pytest
import torch
import numpy as np
from PIL import Image
from data.transforms import ResizePadding, DefaultTransform


def test_resize_padding_wide_image():
    """测试宽图padding"""
    transform = ResizePadding(target_size=1024)

    # 创建2048x1024的宽图
    image = Image.new('RGB', (2048, 1024), color=(255, 0, 0))
    mask = Image.new('L', (2048, 1024), color=128)

    img_out, mask_out = transform(image, mask)

    assert img_out.size == (1024, 1024)
    assert mask_out.size == (1024, 1024)

    # 检查padding是否正确（上下应该有黑边）
    img_array = np.array(img_out)
    # 上下应该有padding（黑色），中间应该是红色
    assert np.all(img_array[0, :, 0] == 0)  # 顶部是黑色
    assert np.all(img_array[512, :, 0] == 255)  # 中间是红色


def test_resize_padding_tall_image():
    """测试高图padding"""
    transform = ResizePadding(target_size=1024)

    # 创建512x2048的高图
    image = Image.new('RGB', (512, 2048), color=(0, 255, 0))
    mask = Image.new('L', (512, 2048), color=64)

    img_out, mask_out = transform(image, mask)

    assert img_out.size == (1024, 1024)
    assert mask_out.size == (1024, 1024)

    # 检查padding是否正确（左右应该有黑边）
    img_array = np.array(img_out)
    assert np.all(img_array[:, 0, 1] == 0)  # 左边是黑色
    assert np.all(img_array[:, 512, 1] == 255)  # 中间是绿色


def test_resize_padding_mask_fill_value():
    """测试mask的填充值为255"""
    transform = ResizePadding(target_size=1024, mask_fill_value=255)

    image = Image.new('RGB', (2048, 1024))
    mask = Image.new('L', (2048, 1024), color=1)

    img_out, mask_out = transform(image, mask)

    # mask的padding区域应该是255
    mask_array = np.array(mask_out)
    assert mask_array[0, 0] == 255  # 顶部padding


def test_default_transform_output_types():
    """测试DefaultTransform输出类型和形状"""
    transform = DefaultTransform(target_size=1024)

    image = Image.new('RGB', (512, 512), color=(128, 128, 128))
    mask = Image.new('L', (512, 512), color=5)

    img_tensor, mask_tensor = transform(image, mask)

    # 检查类型和形状
    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(mask_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 1024, 1024)
    assert mask_tensor.shape == (1024, 1024)
    assert img_tensor.dtype == torch.float32
    assert mask_tensor.dtype == torch.int64


def test_default_transform_normalization():
    """测试图像归一化"""
    transform = DefaultTransform(target_size=1024)

    # 创建白色图像
    image = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
    mask = Image.new('L', (1024, 1024), color=0)

    img_tensor, mask_tensor = transform(image, mask)

    # 白色经过归一化后应该约等于 (1 - mean) / std
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    expected_r = (1.0 - 0.485) / 0.229
    expected_g = (1.0 - 0.456) / 0.224
    expected_b = (1.0 - 0.406) / 0.225

    assert torch.abs(img_tensor[0, 512, 512] - expected_r) < 0.01
    assert torch.abs(img_tensor[1, 512, 512] - expected_g) < 0.01
    assert torch.abs(img_tensor[2, 512, 512] - expected_b) < 0.01
