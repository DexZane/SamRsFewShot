#!/usr/bin/env python
"""Temporary script to download LoveDA dataset."""
import sys
sys.path.insert(0, '/Users/dexzane/Desktop/FindProject/samRsFewShot')

from samRsFewShot.data.download import download_loveda

if __name__ == '__main__':
    download_loveda(root='./data/LoveDA')
    print("Dataset download completed!")
