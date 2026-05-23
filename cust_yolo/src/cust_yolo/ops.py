import platform
import sys

import numpy as np
import torch

# macOS 14 has an MPS bug with in-place clamp on certain tensor shapes.
# Use non-in-place clamp on that platform to avoid silent wrong results.
_macos14 = sys.platform == "darwin" and platform.mac_ver()[0].startswith("14.")


def empty_like(x):
    """Return an uninitialised tensor/array with the same shape and dtype as x."""
    return torch.empty_like(x, dtype=x.dtype) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=x.dtype)


def xywh2xyxy(x):
    """Convert (cx, cy, w, h) boxes to (x1, y1, x2, y2).

    Args:
        x (torch.Tensor | np.ndarray): (..., 4) boxes in xywh format.

    Returns:
        (torch.Tensor | np.ndarray): Same shape, xyxy format.
    """
    assert x.shape[-1] == 4, f"expected last dim 4, got {x.shape}"
    y = empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def clip_boxes(boxes, shape):
    """Clip bounding boxes to image boundaries.

    Args:
        boxes (torch.Tensor | np.ndarray): (..., 4) in xyxy format.
        shape (tuple): (H, W[, C]) image shape.

    Returns:
        boxes clipped in-place and returned.
    """
    h, w = shape[:2]
    if isinstance(boxes, torch.Tensor):
        if not _macos14:
            boxes[..., 0].clamp_(0, w)
            boxes[..., 1].clamp_(0, h)
            boxes[..., 2].clamp_(0, w)
            boxes[..., 3].clamp_(0, h)
        else:
            boxes[..., 0] = boxes[..., 0].clamp(0, w)
            boxes[..., 1] = boxes[..., 1].clamp(0, h)
            boxes[..., 2] = boxes[..., 2].clamp(0, w)
            boxes[..., 3] = boxes[..., 3].clamp(0, h)
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)
    return boxes


def scale_boxes(
    img1_shape: tuple[int, int],
    boxes,
    img0_shape: tuple[int, int],
    ratio_pad=None,
    padding: bool = True,
    xywh: bool = False,
):
    """Rescale bounding boxes from letterboxed space back to original image space.

    Args:
        img1_shape: (H, W) of the letterboxed image (e.g. (640, 640)).
        boxes (torch.Tensor | np.ndarray): (N, 4) boxes, modified in-place.
        img0_shape: (H, W) of the original image.
        ratio_pad: pre-computed (gain, (pad_x, pad_y)); computed from shapes when None.
        padding: whether letterbox padding was applied (True for standard YOLO inference).
        xywh: True if boxes are in xywh format rather than xyxy.

    Returns:
        Rescaled boxes (same object, modified in-place).
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
        pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad_x
        boxes[..., 1] -= pad_y
        if not xywh:
            boxes[..., 2] -= pad_x
            boxes[..., 3] -= pad_y
    boxes[..., :4] /= gain
    return boxes if xywh else clip_boxes(boxes, img0_shape)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU between two sets of xyxy boxes.

    Args:
        box1: (N, 4)
        box2: (M, 4)
        eps: small value to avoid division by zero.

    Returns:
        (N, M) IoU matrix.
    """
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
