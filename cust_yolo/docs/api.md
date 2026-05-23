# API Reference

## `cust_yolo.preprocess`

### `letterbox(img, new_shape=(640, 640), padding_value=114)`

Resize and symmetrically pad an image to `new_shape` without distorting aspect ratio.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `img` | `np.ndarray` | — | HWC BGR uint8 input image |
| `new_shape` | `tuple[int, int]` | `(640, 640)` | Target `(height, width)` |
| `padding_value` | `int` | `114` | Constant fill value for padded pixels |

**Returns** `tuple[np.ndarray, float, tuple[int, int]]`

- `img` — padded image of shape `(new_shape[0], new_shape[1], C)`
- `scale` — ratio applied to original `(h, w)` dimensions
- `(pad_x, pad_y)` — pixels added on the **left** and **top** sides

---

### `to_tensor(img, device="cpu")`

Convert a letterboxed BGR HWC uint8 image to a batched CHW float32 tensor.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `img` | `np.ndarray` | — | HWC BGR uint8 image (output of `letterbox`) |
| `device` | `str \| torch.device` | `"cpu"` | Target device |

**Returns** `torch.Tensor` — shape `(1, 3, H, W)`, float32, values in `[0, 1]`

---

## `cust_yolo.nms`

### `non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, ...)`

Full NMS decode pipeline for YOLOv8 detection output.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `prediction` | `torch.Tensor` | — | Raw model output `(B, 4+nc, num_anchors)` |
| `conf_thres` | `float` | `0.25` | Minimum confidence score |
| `iou_thres` | `float` | `0.45` | IoU suppression threshold |
| `classes` | `list[int] \| None` | `None` | Whitelist of class ids; `None` keeps all |
| `agnostic` | `bool` | `False` | Class-agnostic NMS |
| `multi_label` | `bool` | `False` | Allow multiple labels per anchor |
| `labels` | `list` | `()` | A-priori labels for auto-labelling; empty for inference |
| `max_det` | `int` | `300` | Maximum detections per image |
| `nc` | `int` | `0` | Number of classes; inferred from `prediction` when `0` |
| `max_nms` | `int` | `30000` | Cap on candidates fed into NMS kernel |
| `max_wh` | `int` | `7680` | Max box side length (used for class-offset trick) |
| `end2end` | `bool` | `False` | Model already outputs `(B, N, 6)` — skip decode |
| `return_idxs` | `bool` | `False` | Also return anchor indices of kept detections |

**Returns**

- Default: `list[torch.Tensor]` — one `(N, 6)` tensor per image: `[x1, y1, x2, y2, conf, cls]`
- `return_idxs=True`: `tuple[list[torch.Tensor], list[torch.Tensor]]`

---

### `TorchNMS`

Pure-PyTorch NMS kernels; used automatically when `torchvision` is not installed.

| Method | Description |
|---|---|
| `TorchNMS.nms(boxes, scores, iou_threshold)` | Greedy NMS; matches torchvision exactly |
| `TorchNMS.fast_nms(boxes, scores, iou_threshold)` | Upper-triangular IoU matrix NMS (exportable) |
| `TorchNMS.batched_nms(boxes, scores, idxs, iou_threshold)` | Class-aware batched NMS |

---

## `cust_yolo.ops`

### `xywh2xyxy(x)`

Convert `(cx, cy, w, h)` boxes to `(x1, y1, x2, y2)`.

**Parameters** `x (torch.Tensor | np.ndarray)` — `(..., 4)`

**Returns** same shape and type as input

---

### `clip_boxes(boxes, shape)`

Clamp box coordinates to image boundaries in-place.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `boxes` | `torch.Tensor \| np.ndarray` | `(..., 4)` xyxy boxes |
| `shape` | `tuple` | `(H, W[, C])` image shape |

**Returns** `boxes` (same object, modified in-place)

---

### `scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False)`

Map bounding boxes from letterboxed space back to original image pixel coordinates.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `img1_shape` | `tuple[int, int]` | — | `(H, W)` of the letterboxed image |
| `boxes` | `torch.Tensor \| np.ndarray` | — | `(N, 4)` boxes, modified in-place |
| `img0_shape` | `tuple[int, int]` | — | `(H, W)` of the original image |
| `ratio_pad` | `tuple \| None` | `None` | Pre-computed `(gain, (pad_x, pad_y))`; computed when `None` |
| `padding` | `bool` | `True` | Whether letterbox padding was applied |
| `xywh` | `bool` | `False` | Boxes are in xywh format rather than xyxy |

**Returns** rescaled boxes (same object)

---

### `box_iou(box1, box2, eps=1e-7)`

Pairwise IoU between two sets of xyxy boxes.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `box1` | `torch.Tensor` | `(N, 4)` |
| `box2` | `torch.Tensor` | `(M, 4)` |

**Returns** `torch.Tensor` — `(N, M)` IoU matrix

---

## `cust_yolo.model`

### `load_model(path, device="cpu")`

Load a YOLOv8 `.pt` checkpoint as an `nn.Module`.

> **Note**: ultralytics must be importable at load time because the model graph is
> serialised as Python objects. Use `load_onnx` or `load_torchscript` to eliminate
> that dependency at inference time.

**Returns** `nn.Module` in `eval()` mode. Access `model.names` for `{int: str}` class labels.

---

### `load_onnx(path, device="cpu")`

Create an ONNXRuntime `InferenceSession`. Zero ultralytics dependency.

**Returns** `onnxruntime.InferenceSession`

---

### `load_torchscript(path, device="cpu")`

Load a `.torchscript` model exported via `model.export(format="torchscript")`.

**Returns** `torch.jit.ScriptModule` in `eval()` mode

---

## `cust_yolo.hf_integration`

### `run_detection(batch, model, device, conf=0.25, iou=0.45)`

HuggingFace Datasets batch-map function for YOLOv8 detection.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `batch` | `dict` | — | Dataset batch with `"image"` key (list of PIL Images) |
| `model` | callable | — | Any model callable: `.pt` nn.Module, TorchScript, or ONNX wrapper |
| `device` | `str \| torch.device` | — | Inference device |
| `conf` | `float` | `0.25` | Confidence threshold |
| `iou` | `float` | `0.45` | IoU NMS threshold |

**Returns** `{"detections": list[dict]}` — one dict per image:

```python
{
    "boxes":   [[x1, y1, x2, y2], ...],   # float, original pixel coords
    "scores":  [0.93, ...],
    "classes": [0, ...],                   # int
}
```
