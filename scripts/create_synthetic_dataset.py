#!/usr/bin/env python
"""
Create synthetic LoveDA-like dataset for Phase 2 training validation.
This allows Phase 2 to proceed without waiting for the real dataset download.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random

def create_synthetic_loveda(root='./data/LoveDA', num_train=100, num_val=30):
    """
    Create synthetic LoveDA dataset structure with random images.

    Args:
        root: Root directory for dataset
        num_train: Number of training samples per split (Urban/Rural)
        num_val: Number of validation samples per split
    """
    root = Path(root)

    # LoveDA classes
    num_classes = 7
    class_colors = [
        (0, 0, 0),       # 0: Background
        (255, 0, 0),     # 1: Building
        (255, 255, 0),   # 2: Road
        (0, 0, 255),     # 3: Water
        (159, 129, 183), # 4: Barren
        (0, 255, 0),     # 5: Forest
        (255, 195, 128)  # 6: Agricultural
    ]

    # Create directory structure
    splits = ['Train', 'Val', 'Test']
    areas = ['Urban', 'Rural']

    for split in splits:
        for area in areas:
            (root / split / area / 'images_png').mkdir(parents=True, exist_ok=True)
            (root / split / area / 'masks_png').mkdir(parents=True, exist_ok=True)

    def generate_sample(idx, size=1024):
        """Generate one synthetic image-mask pair with diverse class distribution."""
        # Create random image
        image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

        # Create semantic mask - ensure diverse class distribution
        mask = np.zeros((size, size), dtype=np.uint8)

        # Assign a primary dominant class for this sample (cycle through classes)
        dominant_class = (idx % (num_classes - 1)) + 1  # Skip background, cycle 1-6

        # Add large dominant region for the primary class
        for _ in range(2):
            x = random.randint(0, size - 400)
            y = random.randint(0, size - 400)
            w = random.randint(300, 500)
            h = random.randint(300, 500)
            mask[y:y+h, x:x+w] = dominant_class

        # Add smaller regions for other classes
        num_other_regions = random.randint(3, 6)
        for _ in range(num_other_regions):
            x = random.randint(0, size - 200)
            y = random.randint(0, size - 200)
            w = random.randint(100, 250)
            h = random.randint(100, 250)

            # Random class (skip background, prefer non-dominant classes)
            class_id = random.randint(1, num_classes - 1)
            mask[y:y+h, x:x+w] = class_id

        return image, mask

    # Generate training data
    print(f"Generating synthetic LoveDA dataset at {root}")
    print(f"Train: {num_train} samples per area")
    print(f"Val: {num_val} samples per area")

    for split, num_samples in [('Train', num_train), ('Val', num_val), ('Test', num_val)]:
        for area in areas:
            image_dir = root / split / area / 'images_png'
            mask_dir = root / split / area / 'masks_png'

            print(f"\nGenerating {split}/{area}...")
            for i in tqdm(range(num_samples)):
                # Generate sample
                image, mask = generate_sample(i)

                # Save image
                image_path = image_dir / f'{area}_{i:04d}.png'
                Image.fromarray(image).save(image_path)

                # Save mask
                mask_path = mask_dir / f'{area}_{i:04d}.png'
                Image.fromarray(mask).save(mask_path)

    print(f"\n✅ Synthetic dataset created successfully!")
    print(f"Total samples:")
    print(f"  Train: {num_train * 2} ({num_train} Urban + {num_train} Rural)")
    print(f"  Val: {num_val * 2} ({num_val} Urban + {num_val} Rural)")
    print(f"  Test: {num_val * 2} ({num_val} Urban + {num_val} Rural)")
    print(f"\nYou can now run: bash scripts/run_phase2_baseline.sh")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create synthetic LoveDA dataset')
    parser.add_argument('--root', type=str, default='./data/LoveDA', help='Dataset root directory')
    parser.add_argument('--num_train', type=int, default=100, help='Training samples per area')
    parser.add_argument('--num_val', type=int, default=30, help='Validation samples per area')
    args = parser.parse_args()

    create_synthetic_loveda(args.root, args.num_train, args.num_val)
