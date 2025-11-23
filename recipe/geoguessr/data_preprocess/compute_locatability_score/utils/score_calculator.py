"""Score calculator for locatability."""

import json
import torch
import numpy as np
from typing import Dict, Tuple

from .semantic_config import SEMANTIC_CLASSES, CLASS_WEIGHTS


def compute_locatability_score(
    semantic_map: torch.Tensor,
    class_weights: torch.Tensor,
    class_names: np.ndarray
) -> Tuple[float, str]:
    """
    Compute locatability score from semantic segmentation map.

    Formula: score = Σ (weight[class] × percentage[class])

    Args:
        semantic_map: Tensor of shape (H, W) with class IDs for each pixel
        class_weights: Tensor of shape (num_classes,) with locatability weights
        class_names: Array of class name strings

    Returns:
        locatability_score: Float in [0, 1] range
        class_mapping_json: JSON string mapping class names to percentages
    """
    total_pixels = semantic_map.numel()

    if total_pixels == 0:
        return 0.0, json.dumps({})

    # Count pixels per class
    unique_classes, counts = torch.unique(semantic_map, return_counts=True)

    # Filter valid classes (within weight array bounds)
    valid_mask = unique_classes < len(class_weights)
    valid_classes = unique_classes[valid_mask]
    valid_counts = counts[valid_mask]

    if valid_classes.numel() == 0:
        return 0.0, json.dumps({})

    # Compute percentages for each class
    percentages = valid_counts.float() / total_pixels

    # Get corresponding weights and names
    class_indices = valid_classes.cpu().numpy()
    class_specific_weights = class_weights[valid_classes]
    names = class_names[class_indices]

    # Compute weighted sum (locatability score)
    locatability_score = torch.sum(class_specific_weights * percentages).item()

    # Create class mapping dictionary
    class_mapping = {
        str(name): float(pct)
        for name, pct in zip(names, percentages.cpu().numpy())
    }

    return locatability_score, json.dumps(class_mapping)
