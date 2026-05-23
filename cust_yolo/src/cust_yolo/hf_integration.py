"""HuggingFace Datasets integration for batch detection."""

from __future__ import annotations

import numpy as np
import torch

from .nms import non_max_suppression
from .ops import scale_boxes
from .preprocess import letterbox, to_tensor

_LETTERBOX_SHAPE = (640, 640)


def run_detection(
    batch: dict,
    model,
    device: str | torch.device,
    conf: float = 0.25,
    iou: float = 0.45,
) -> dict:
    """Run YOLOv8 detection on a HuggingFace Dataset batch.

    Designed for use with ``Dataset.map(batched=True)``. Each image in the
    batch is independently letterboxed, inferred, and NMS-filtered. Box
    coordinates in the output are in original image pixel space.

    Args:
        batch: Dict with an ``"image"`` key containing a list of PIL Images.
        model: Any callable that accepts a (1, 3, 640, 640) float32 tensor and
               returns a (1, 4+nc, num_anchors) tensor — a loaded .pt nn.Module,
               a TorchScript module, or an ONNXRuntime session wrapper.
        device: Torch device for inference (e.g. ``"cuda"`` or ``"cpu"``).
        conf: Confidence threshold passed to NMS.
        iou: IoU threshold passed to NMS.

    Returns:
        Dict with a ``"detections"`` key: a list (one entry per image) of
        ``{"boxes": [[x1,y1,x2,y2], ...], "scores": [...], "classes": [...]}``.
        Coordinates are in original image pixel space.
    """
    results = []
    for pil_img in batch["image"]:
        img_bgr = np.array(pil_img)[..., ::-1]  # RGB → BGR for letterbox
        orig_hw = img_bgr.shape[:2]

        lb_img, _scale, _pad = letterbox(img_bgr, new_shape=_LETTERBOX_SHAPE)
        img_t = to_tensor(lb_img, device=device)

        with torch.no_grad():
            raw = model(img_t)

        dets = non_max_suppression(raw, conf_thres=conf, iou_thres=iou)[0]  # (N, 6)
        if len(dets):
            dets[:, :4] = scale_boxes(_LETTERBOX_SHAPE, dets[:, :4], orig_hw)

        results.append(
            {
                "boxes": dets[:, :4].cpu().numpy().tolist(),
                "scores": dets[:, 4].cpu().numpy().tolist(),
                "classes": dets[:, 5].cpu().numpy().astype(int).tolist(),
            }
        )

    return {"detections": results}
