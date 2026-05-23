import numpy as np
import pytest
import torch

from deepbrain.cust_yolo.preprocess import letterbox, to_tensor


class TestLetterbox:
    def test_output_shape(self, bgr_image):
        out, _, _ = letterbox(bgr_image, new_shape=(640, 640))
        assert out.shape == (640, 640, 3)

    def test_scale_is_positive(self, bgr_image):
        _, scale, _ = letterbox(bgr_image)
        assert scale > 0

    def test_scale_limits_by_min_ratio(self):
        # 480×640 image → target 640×640; limiting ratio is 640/640=1.0 (width)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, scale, _ = letterbox(img, new_shape=(640, 640))
        assert scale == pytest.approx(1.0)

    def test_padding_is_symmetric(self):
        # Square image into square target → zero padding
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        _, _, (pad_x, pad_y) = letterbox(img, new_shape=(640, 640))
        assert pad_x == 0 and pad_y == 0

    def test_non_square_input_has_padding(self, bgr_image):
        # 480×640 → 640×640: height needs padding (480 < 640 after scale)
        _, _, (pad_x, pad_y) = letterbox(bgr_image, new_shape=(640, 640))
        assert pad_x == 0   # width already fills target
        assert pad_y > 0    # height is padded

    def test_custom_new_shape(self, bgr_image):
        out, _, _ = letterbox(bgr_image, new_shape=(320, 320))
        assert out.shape == (320, 320, 3)

    def test_padding_value(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out, _, (px, py) = letterbox(img, new_shape=(200, 200), padding_value=114)
        if py > 0:
            assert (out[:py] == 114).all()


class TestToTensor:
    def test_shape(self, bgr_image):
        lb, *_ = letterbox(bgr_image)
        t = to_tensor(lb)
        assert t.shape == (1, 3, 640, 640)

    def test_dtype(self, bgr_image):
        lb, *_ = letterbox(bgr_image)
        t = to_tensor(lb)
        assert t.dtype == torch.float32

    def test_range(self, bgr_image):
        lb, *_ = letterbox(bgr_image)
        t = to_tensor(lb)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_bgr_to_rgb(self):
        # Image with a known B channel: verify channel 0 in tensor = R = 0
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # R=255, G=0, B=0 in BGR convention means R channel is [2]
        t = to_tensor(img)
        assert t[0, 0].max() == pytest.approx(1.0)  # R → channel 0 in RGB tensor
        assert t[0, 1].max() == pytest.approx(0.0)
        assert t[0, 2].max() == pytest.approx(0.0)
