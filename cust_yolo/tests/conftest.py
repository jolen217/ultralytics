import numpy as np
import pytest
import torch


@pytest.fixture
def bgr_image():
    """640×480 synthetic BGR uint8 image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def raw_prediction():
    """Synthetic YOLOv8 raw output: (1, 84, 6300) with two high-confidence boxes."""
    pred = torch.zeros(1, 84, 6300)
    # box 0: cx=100, cy=100, w=50, h=50, class 0 score=0.9
    pred[0, :4, 0] = torch.tensor([100.0, 100.0, 50.0, 50.0])
    pred[0, 4, 0] = 0.9
    # box 1: cx=400, cy=300, w=80, h=80, class 1 score=0.8 — well-separated
    pred[0, :4, 1] = torch.tensor([400.0, 300.0, 80.0, 80.0])
    pred[0, 5, 1] = 0.8
    return pred
