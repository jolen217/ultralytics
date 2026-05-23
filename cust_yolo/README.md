# cust-yolo

Pure-PyTorch YOLOv8 detection inference with **zero ultralytics runtime dependency**.

Extracts the minimum set of preprocessing, NMS, and coordinate-rescaling utilities
needed to run a YOLOv8 `.pt` / `.onnx` / `.torchscript` model end-to-end — including
a drop-in helper for [HuggingFace Datasets](https://huggingface.co/docs/datasets).

---

## Installation

```bash
pip install -e ".[dev]"          # editable install + test/lint deps
pip install -e ".[dev,onnx,hf]"  # also pull in onnxruntime and datasets
```

---

## Quick start

### Load and run a `.pt` model

```python
import torch
from cust_yolo import load_model, letterbox, to_tensor, non_max_suppression, scale_boxes
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = load_model("model.pt", device=device)

img_bgr               = cv2.imread("image.jpg")           # HWC BGR uint8
lb_img, scale, pad    = letterbox(img_bgr)                 # → 640×640
img_t                 = to_tensor(lb_img, device=device)   # (1,3,640,640)

with torch.no_grad():
    raw = model(img_t)                                     # (1, 84, 6300)

dets = non_max_suppression(raw, conf_thres=0.25, iou_thres=0.45)[0]  # (N, 6)
if len(dets):
    dets[:, :4] = scale_boxes((640, 640), dets[:, :4], img_bgr.shape[:2])

# dets columns: [x1, y1, x2, y2, confidence, class_id]
```

### ONNX inference (no ultralytics at load time)

```python
import numpy as np
import torch
from cust_yolo import load_onnx, letterbox, non_max_suppression, scale_boxes

sess   = load_onnx("model.onnx", device="cpu")
inp    = sess.get_inputs()[0].name                         # usually "images"

img    = cv2.imread("image.jpg")
lb, *_ = letterbox(img)
x      = lb[..., ::-1].copy().astype(np.float32) / 255.0  # BGR→RGB, normalise
x      = x.transpose(2, 0, 1)[None]                        # NCHW

raw    = torch.from_numpy(sess.run(None, {inp: x})[0])     # (1, 84, 6300)
dets   = non_max_suppression(raw)[0]
```

### HuggingFace Datasets batch mapping

```python
from datasets import load_dataset
from cust_yolo import load_model
from cust_yolo.hf_integration import run_detection

device = "cuda"
model  = load_model("model.pt", device=device)

ds = load_dataset("org/dataset")
ds = ds.map(
    run_detection,
    batched=True,
    batch_size=8,
    fn_kwargs={"model": model, "device": device},
)
ds.push_to_hub("org/results")
```

---

## API reference

See [docs/api.md](docs/api.md) for full parameter documentation.

### Preprocessing

| Symbol | Description |
|---|---|
| `letterbox(img, new_shape, padding_value)` | Resize + pad to square; returns `(img, scale, (pad_x, pad_y))` |
| `to_tensor(img, device)` | BGR HWC uint8 → RGB CHW float32 tensor, batch dim added |

### NMS

| Symbol | Description |
|---|---|
| `non_max_suppression(prediction, ...)` | Full NMS pipeline; returns list of `(N, 6)` tensors per image |
| `TorchNMS.nms(boxes, scores, iou_threshold)` | Greedy NMS, matches torchvision output |
| `TorchNMS.fast_nms(boxes, scores, iou_threshold)` | Fast-NMS via upper-triangular IoU matrix |

### Coordinate utilities

| Symbol | Description |
|---|---|
| `xywh2xyxy(x)` | Convert center-format boxes to corner-format |
| `clip_boxes(boxes, shape)` | Clamp box coords to image boundaries |
| `scale_boxes(img1_shape, boxes, img0_shape)` | Map letterbox coords back to original image space |
| `box_iou(box1, box2)` | Pairwise IoU matrix `(N, M)` |

### Model loading

| Symbol | Description |
|---|---|
| `load_model(path, device)` | Load `.pt` checkpoint as `nn.Module` (requires ultralytics at load time) |
| `load_onnx(path, device)` | Create an ONNXRuntime session (no ultralytics) |
| `load_torchscript(path, device)` | Load exported `.torchscript` model (no ultralytics) |

---

## Development

```bash
pip install -e ".[dev]"
pytest                  # run test suite
ruff check src/ tests/  # lint
ruff format src/ tests/ # format
```

---

## Tensor flow reference

```
Input image (HWC, uint8, BGR)
    ↓ letterbox(new_shape=(640,640))
    ↓ /255, BGR→RGB, HWC→NCHW float32
    ↓ model(x)
Raw output: (batch, 84, 6300)
    ├─ [:, :4,  :] — box coords xywh (relative to 640×640 letterboxed space)
    └─ [:, 4:, :] — per-class logits (80 classes for COCO)
    ↓ non_max_suppression(conf_thres=0.25, iou_thres=0.45)
NMS output: list[Tensor(N, 6)] — one per image
    └─ [x1, y1, x2, y2, conf, class_id]  (still in 640×640 space)
    ↓ scale_boxes(img1_shape=(640,640), boxes, img0_shape=orig_hw)
Final: [x1, y1, x2, y2, conf, class_id] in original pixel coordinates
```
