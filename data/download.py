"""LoveDA数据集下载功能"""
import os
import urllib.request
import zipfile
from typing import Optional
from tqdm import tqdm


def _verify_dataset(root: str) -> bool:
    """验证数据集文件结构完整性

    Args:
        root: 数据集根目录

    Returns:
        True if all required directories exist, False otherwise
    """
    required_dirs = [
        'Train/Urban/images_png',
        'Train/Urban/masks_png',
        'Train/Rural/images_png',
        'Train/Rural/masks_png',
        'Val/Urban/images_png',
        'Val/Urban/masks_png',
        'Val/Rural/images_png',
        'Val/Rural/masks_png',
        'Test/Urban/images_png',
        'Test/Urban/masks_png',
        'Test/Rural/images_png',
        'Test/Rural/masks_png',
    ]

    for dir_path in required_dirs:
        full_path = os.path.join(root, dir_path)
        if not os.path.isdir(full_path):
            return False

    return True


class DownloadProgressBar(tqdm):
    """带进度条的下载器"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_loveda(root: str, url: Optional[str] = None) -> None:
    """自动下载LoveDA数据集

    Args:
        root: 下载目标目录
        url: 数据集URL（默认使用官方地址）

    Raises:
        RuntimeError: 下载或解压失败
    """
    if url is None:
        url = "https://zenodo.org/record/5706578/files/LoveDA.zip"

    zip_path = os.path.join(root, "LoveDA.zip")

    # 创建目录
    os.makedirs(root, exist_ok=True)

    # 下载（带进度条）
    print(f"Downloading LoveDA dataset from {url}...")

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="LoveDA") as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    except Exception as e:
        # 清理不完整的下载
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError(f"Failed to download dataset: {e}")

    # 解压
    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
    except Exception as e:
        raise RuntimeError(f"Failed to extract dataset: {e}")
    finally:
        # 清理zip文件
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # 验证文件结构
    if not _verify_dataset(root):
        raise RuntimeError("Dataset structure verification failed. Please check the downloaded files.")

    print("Download complete!")
