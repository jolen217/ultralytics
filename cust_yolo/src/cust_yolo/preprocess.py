import cv2
import numpy as np
import torch


def letterbox(img: np.ndarray, new_shape: tuple[int, int] = (640, 640), padding_value: int = 114):
    """Resize and pad image to new_shape with centered letterboxing.

    Args:
        img: HWC BGR uint8 numpy array.
        new_shape: (height, width) target size.
        padding_value: constant pad fill value.

    Returns:
        img: padded image (new_shape[0], new_shape[1], C)
        scale: ratio applied to original dimensions
        pad: (pad_x, pad_y) pixels added on left/top sides
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (round(w * r), round(h * r))  # (W, H)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(padding_value,) * 3)
    return img, r, (left, top)


def to_tensor(img: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    """Convert a letterboxed BGR HWC uint8 image to a batched CHW float32 tensor.

    Args:
        img: HWC BGR uint8 numpy array (output of letterbox).
        device: torch device string or object.

    Returns:
        Tensor of shape (1, 3, H, W), float32, values in [0, 1].
    """
    t = torch.from_numpy(img[..., ::-1].copy())  # BGR → RGB
    t = t.permute(2, 0, 1).float() / 255.0       # HWC → CHW, normalise
    return t.unsqueeze(0).to(device)
