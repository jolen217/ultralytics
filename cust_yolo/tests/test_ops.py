import numpy as np
import pytest
import torch

from cust_yolo.ops import box_iou, clip_boxes, scale_boxes, xywh2xyxy


class TestXywh2Xyxy:
    def test_known_conversion(self):
        x = torch.tensor([[10.0, 20.0, 4.0, 6.0]])  # cx, cy, w, h
        y = xywh2xyxy(x)
        assert y.tolist() == [[8.0, 17.0, 12.0, 23.0]]

    def test_batch(self):
        x = torch.zeros(5, 4)
        y = xywh2xyxy(x)
        assert y.shape == (5, 4)

    def test_numpy_input(self):
        x = np.array([[0.5, 0.5, 1.0, 1.0]])
        y = xywh2xyxy(x)
        np.testing.assert_allclose(y, [[0.0, 0.0, 1.0, 1.0]])

    def test_wrong_last_dim_raises(self):
        with pytest.raises(AssertionError):
            xywh2xyxy(torch.zeros(3, 5))


class TestClipBoxes:
    def test_clamps_above(self):
        boxes = torch.tensor([[800.0, 900.0, 900.0, 1000.0]])
        clip_boxes(boxes, (640, 640))
        assert boxes[:, 0].max() <= 640
        assert boxes[:, 1].max() <= 640

    def test_clamps_below(self):
        boxes = torch.tensor([[-10.0, -5.0, 100.0, 100.0]])
        clip_boxes(boxes, (480, 640))
        assert boxes[:, 0].min() >= 0
        assert boxes[:, 1].min() >= 0

    def test_numpy_boxes(self):
        boxes = np.array([[800.0, 900.0, 900.0, 1000.0]])
        clip_boxes(boxes, (640, 640))
        assert boxes[:, [0, 2]].max() <= 640
        assert boxes[:, [1, 3]].max() <= 640

    def test_in_bounds_unchanged(self):
        boxes = torch.tensor([[10.0, 10.0, 100.0, 100.0]])
        original = boxes.clone()
        clip_boxes(boxes, (640, 640))
        assert torch.allclose(boxes, original)


class TestScaleBoxes:
    def test_identity(self):
        # Same source and target shape → boxes unchanged (minus clip)
        boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        out = scale_boxes((640, 640), boxes.clone(), (640, 640))
        assert torch.allclose(out, boxes)

    def test_half_scale(self):
        # letterbox image 640×640 contains a 320×320 original (scale=0.5, no pad)
        # A box at [0, 0, 320, 320] in letterbox space should become [0, 0, 640, 640]
        boxes = torch.tensor([[0.0, 0.0, 320.0, 320.0]])
        # ratio_pad: gain=0.5, pad=(0,0)
        out = scale_boxes((640, 640), boxes.clone(), (640, 640), ratio_pad=((0.5,), (0, 0)))
        assert torch.allclose(out, torch.tensor([[0.0, 0.0, 640.0, 640.0]]))

    def test_removes_padding(self):
        # 320×480 original letterboxed into 640×640: gain=640/480≈1.333, pad_x=80
        orig_hw = (320, 480)
        lb_hw = (640, 640)
        # A box in letterbox space that starts at the left-pad boundary
        boxes = torch.tensor([[80.0, 0.0, 640.0, 320.0]])
        out = scale_boxes(lb_hw, boxes.clone(), orig_hw)
        # After removing pad_x=80 and dividing by gain, x1 should be ~0
        assert out[0, 0].item() == pytest.approx(0.0, abs=2.0)


class TestBoxIou:
    def test_identical_boxes(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        iou = box_iou(boxes, boxes)
        assert iou.item() == pytest.approx(1.0, abs=1e-5)

    def test_non_overlapping(self):
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        box2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        iou = box_iou(box1, box2)
        assert iou.item() == pytest.approx(0.0, abs=1e-6)

    def test_half_overlap(self):
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        box2 = torch.tensor([[5.0, 0.0, 15.0, 10.0]])  # 5×10 intersection
        iou = box_iou(box1, box2)
        # intersection=50, union=150 → IoU=1/3
        assert iou.item() == pytest.approx(1 / 3, abs=1e-5)

    def test_output_shape(self):
        box1 = torch.rand(3, 4)
        box2 = torch.rand(5, 4)
        iou = box_iou(box1, box2)
        assert iou.shape == (3, 5)
