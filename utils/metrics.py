"""
Metrics module for segmentation evaluation

Provides functions to compute IoU (Intersection over Union) metrics.
"""

import torch
import numpy as np


def compute_iou(pred, target, numClasses):
    """
    Compute mean IoU and per-class IoU

    Args:
        pred: Predicted segmentation masks (B, H, W), integer class labels
        target: Ground truth segmentation masks (B, H, W), integer class labels
        numClasses: Number of classes

    Returns:
        miou: Mean IoU across all classes (float)
        perClassIou: Per-class IoU values (list of floats)
    """
    # Convert to numpy if tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    perClassIou = []

    for cls in range(numClasses):
        # Binary masks for current class
        predMask = (pred == cls)
        targetMask = (target == cls)

        # Compute intersection and union
        intersection = np.logical_and(predMask, targetMask).sum()
        union = np.logical_or(predMask, targetMask).sum()

        # Handle union=0 case
        if union == 0:
            # If the class is not present in both pred and target, ignore it
            iou = np.nan
        else:
            iou = intersection / union

        perClassIou.append(iou)

    # Compute mean IoU (ignoring NaN values)
    perClassIou = np.array(perClassIou)
    validIous = perClassIou[~np.isnan(perClassIou)]

    if len(validIous) > 0:
        miou = validIous.mean()
    else:
        miou = 0.0

    return miou, perClassIou.tolist()
