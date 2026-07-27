from setuptools import setup, find_packages

setup(
    name="samRsFewShot",
    version="0.1.0",
    description="SAM-based Few-Shot Segmentation for Remote Sensing - Phase 1 Baseline",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "segment-anything",
        "peft>=0.5.0",
        "numpy>=1.24.0",
    ],
)
