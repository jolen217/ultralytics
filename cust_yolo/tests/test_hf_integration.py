import numpy as np
import pytest
import torch

from deepbrain.cust_yolo.hf_integration import run_detection


class _FakeModel:
    """Deterministic stub: returns one high-confidence box at anchor index 0."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pred = torch.zeros(x.shape[0], 84, 6300)
        pred[:, :4, 0] = torch.tensor([100.0, 100.0, 50.0, 50.0])
        pred[:, 4, 0] = 0.95
        return pred


class _PILStub:
    """Minimal PIL Image stub backed by a numpy array."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __array__(self, dtype=None):
        return self._arr if dtype is None else self._arr.astype(dtype)


@pytest.fixture
def fake_batch():
    rng = np.random.default_rng(1)
    imgs = [_PILStub(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)) for _ in range(2)]
    return {"image": imgs}


def test_output_keys(fake_batch):
    result = run_detection(fake_batch, _FakeModel(), device="cpu")
    assert "detections" in result
    assert len(result["detections"]) == 2


def test_detection_structure(fake_batch):
    result = run_detection(fake_batch, _FakeModel(), device="cpu")
    det = result["detections"][0]
    assert "boxes" in det and "scores" in det and "classes" in det


def test_boxes_in_original_space(fake_batch):
    result = run_detection(fake_batch, _FakeModel(), device="cpu")
    for det in result["detections"]:
        for box in det["boxes"]:
            x1, y1, x2, y2 = box
            assert x2 >= x1 and y2 >= y1
            # coordinates should be in original image range, not letterbox range
            assert x2 <= 640 + 1 and y2 <= 480 + 1


def test_empty_when_no_detections(fake_batch):
    class _NoDetModel:
        def __call__(self, x):
            return torch.zeros(x.shape[0], 84, 6300)

    result = run_detection(fake_batch, _NoDetModel(), device="cpu")
    for det in result["detections"]:
        assert det["boxes"] == []
        assert det["scores"] == []
        assert det["classes"] == []


def test_conf_threshold_filters(fake_batch):
    result = run_detection(fake_batch, _FakeModel(), device="cpu", conf=0.99)
    for det in result["detections"]:
        assert det["boxes"] == []
