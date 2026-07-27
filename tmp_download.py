#!/usr/bin/env python
"""Direct download script without module import."""
import urllib.request
import zipfile
import os
from pathlib import Path
from tqdm import tqdm

def download_loveda(root='./data/LoveDA'):
    """Download and extract LoveDA dataset."""
    url = "https://zenodo.org/record/5706578/files/LoveDA.zip"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / "LoveDA.zip"

    # Check if already exists
    if (root / "Train" / "Urban" / "images_png").exists():
        print("Dataset already exists!")
        return

    print(f"Downloading LoveDA dataset from {url}...")

    # Download with progress bar
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)

    print("Cleaning up...")
    zip_path.unlink()
    print("Download completed!")

if __name__ == '__main__':
    download_loveda('./data/LoveDA')
