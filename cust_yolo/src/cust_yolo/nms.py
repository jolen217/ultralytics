import logging
import sys
import time

import torch

from .ops import box_iou, xywh2xyxy

logger = logging.getLogger(__name__)


class TorchNMS:
    """Pure-PyTorch NMS; used as fallback when torchvision is not available."""

    @staticmethod
    def fast_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        use_triu: bool = True,
        iou_func=box_iou,
        exit_early: bool = True,
    ) -> torch.Tensor:
        """Fast-NMS via upper-triangular IoU matrix (https://arxiv.org/pdf/1904.02689).

        Args:
            boxes: (N, 4) xyxy boxes.
            scores: (N,) confidence scores.
            iou_threshold: suppression threshold.
            use_triu: use torch.triu (exportable); False uses a manual mask.
            iou_func: callable(box1, box2) → IoU matrix.
            exit_early: return immediately on empty input.

        Returns:
            Indices of kept boxes.
        """
        if boxes.numel() == 0 and exit_early:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            scores_ = scores[sorted_idx]
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            scores[sorted_idx] = scores_
            pick = torch.topk(scores_, scores_.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Greedy NMS with early exit; matches torchvision output exactly.

        Args:
            boxes: (N, 4) xyxy boxes.
            scores: (N,) confidence scores.
            iou_threshold: suppression threshold.

        Returns:
            Indices of kept boxes.
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(0, descending=True)

        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0
        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            if inter.sum() == 0:
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter)
            order = rest[iou <= iou_threshold]

        return keep[:keep_idx]

    @staticmethod
    def batched_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        iou_threshold: float,
        use_fast_nms: bool = False,
    ) -> torch.Tensor:
        """Class-aware batched NMS.

        Args:
            boxes: (N, 4) xyxy boxes.
            scores: (N,) confidence scores.
            idxs: (N,) class indices.
            iou_threshold: suppression threshold.
            use_fast_nms: use Fast-NMS instead of greedy NMS.

        Returns:
            Indices of kept boxes.
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        return (
            TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold)
            if use_fast_nms
            else TorchNMS.nms(boxes_for_nms, scores, iou_threshold)
        )


def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """Non-maximum suppression for YOLOv8 detection output.

    Supports standard (non-rotated) detection only. For COCO-class YOLOv8 the raw
    model output has shape (batch, 84, 6300): first 4 channels are xywh box coords,
    remaining 80 are per-class logits.

    Args:
        prediction (torch.Tensor): Raw model output (B, 4+nc, num_anchors).
        conf_thres (float): Minimum confidence to keep a detection.
        iou_thres (float): IoU threshold for NMS suppression.
        classes (list[int] | None): If set, keep only these class ids.
        agnostic (bool): Class-agnostic NMS (ignore class offsets).
        multi_label (bool): Allow multiple labels per box.
        labels (list): A-priori labels for auto-labelling; leave empty for inference.
        max_det (int): Maximum detections per image.
        nc (int): Number of classes; inferred from prediction shape when 0.
        max_time_img (float): Per-image NMS time budget in seconds.
        max_nms (int): Cap on candidates before NMS (sorted by score).
        max_wh (int): Maximum box side length for class-offset trick.
        end2end (bool): Model already outputs (B, N, 6) — skip the NMS decode.
        return_idxs (bool): Also return the anchor indices of kept detections.

    Returns:
        list[torch.Tensor]: One (N, 6) tensor per image: [x1, y1, x2, y2, conf, cls].
        If return_idxs=True, returns (output_list, index_list).
    """
    assert 0 <= conf_thres <= 1, f"conf_thres={conf_thres} out of range [0, 1]"
    assert 0 <= iou_thres <= 1, f"iou_thres={iou_thres} out of range [0, 1]"

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # unwrap (inference, loss) tuple from val mode
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    # End-to-end models output (B, N, 6) directly
    if prediction.shape[-1] == 6 or end2end:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    extra = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]

    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    prediction = prediction.transpose(-1, -2)       # (B, num_anchors, 4+nc+extra)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs

    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        filt = xc[xi]
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]

        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        boxes = x[:, :4] + c

        if "torchvision" in sys.modules:
            import torchvision
            i = torchvision.ops.nms(boxes, scores, iou_thres)
        else:
            i = TorchNMS.nms(boxes, scores, iou_thres)

        i = i[:max_det]
        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)

        if (time.time() - t) > time_limit:
            logger.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break

    return (output, keepi) if return_idxs else output
