# Inference Integration Digest Strategy

## Context

Task: detection. Model: `.pt` file, YOLOv8 architecture (closest to YOLOX — anchor-free, decoupled head, NMS-based, not end2end).
Goal: replace ultralytics APIs with pure PyTorch for production inference.
Constraints: no ultralytics dependency, HuggingFace dataset I/O, ONNX + TorchScript export, CPU + GPU, Triton serving.

---

## YOLOv8 Detection — Tensor Flow

```
Input image (HWC, uint8, BGR)
    ↓ LetterBox(new_shape=(640,640), center=True, padding_value=114)
    ↓ /255, HWC→NCHW, float32
    ↓ model(x)
Raw output: (batch, 84, 6300)
    ┌─ 4 cols: box coords in xywh format (relative to 640×640 letterboxed space)
    └─ 80 cols: per-class confidence scores (no sigmoid, raw logits until NMS)
    ↓ non_max_suppression(conf_thres=0.25, iou_thres=0.45)
NMS output: list[Tensor(N, 6)] — one per image
    └─ 6 cols: [x1, y1, x2, y2, conf, class_id]  (coords still in 640×640 space)
    ↓ scale_boxes(img1_shape=(640,640), boxes, img0_shape=orig_hw)
Final: [x1, y1, x2, y2, conf, class_id] in original image pixel coordinates
```

`84 = 4 + 80` for COCO. For custom-class models: `84 = 4 + num_classes`. Adjust accordingly.
`6300 = 80×80 + 40×40 + 20×20` — P3, P4, P5 anchor grids for a 640×640 input.

---

## Files to Extract (verbatim or near-verbatim)

### 1. Preprocessing — `ultralytics/data/augment.py:1602`

Class `LetterBox`. Key logic in `get_params()` (lines ~1697–1729):
- computes scale ratio `r = min(H_new/H_old, W_new/W_old)`
- computes symmetric padding `dw, dh` (halved when `center=True`)
- resizes with `cv2.INTER_LINEAR`, pads with value `114`

For inference only, you can skip `apply_instances()` and `apply_semantic()` — those handle training labels.

**Standalone reimplementation** (~30 lines of numpy/cv2, no ultralytics imports needed):

```python
import cv2, numpy as np

def letterbox(img, new_shape=(640, 640), padding_value=114):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (round(w * r), round(h * r))           # (W, H)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    top, bottom = dh // 2, dh - dh // 2
    left, right  = dw // 2, dw - dw // 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(padding_value,) * 3)
    return img, r, (left, top)                          # img, scale, padding offset
```

**To tensor**:
```python
img_t = torch.from_numpy(img[..., ::-1].copy())  # BGR→RGB
img_t = img_t.permute(2, 0, 1).float() / 255.0   # HWC→CHW, normalize
img_t = img_t.unsqueeze(0)                         # add batch dim
```

### 2. NMS — `ultralytics/utils/nms.py:13`

`non_max_suppression()` is pure PyTorch with no ultralytics state. Copy the whole file — only imports are `torch`, `time`, `box_iou` (torchvision or reimplement as 10 lines), and `xywh2xyxy` (4 lines below).

For detection (not OBB, not end2end): the only path exercised is lines 72–141.

**`xywh2xyxy`** — `ultralytics/utils/ops.py:231` (4 lines):
```python
def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y
```

### 3. Coordinate rescaling — `ultralytics/utils/ops.py:102`

`scale_boxes(img1_shape, boxes, img0_shape)` — maps NMS output coords back to original image space.
No dependencies. Copy verbatim (~40 lines including `clip_boxes`).

---

## Loading the .pt Model (no ultralytics)

YOLOv8 `.pt` files are standard PyTorch checkpoints. Load with:

```python
import torch
ckpt = torch.load("model.pt", map_location="cpu")
model = ckpt["model"].float().eval()   # nn.Module already instantiated
# model.names → dict of {int: str} class names
```

This works because the model graph is serialized as a Python object, not just weights. The only risk: if the checkpoint was saved with ultralytics classes in scope, you need ultralytics at load time. To break this dependency, export to ONNX or TorchScript first (see below).

---

## Export (run once, then discard ultralytics)

```python
# Run this once in a dev environment that has ultralytics installed
from ultralytics import YOLO
model = YOLO("model.pt")
model.export(format="onnx", opset=17, simplify=True)     # → model.onnx
model.export(format="torchscript")                        # → model.torchscript
```

After export, inference needs zero ultralytics code.

**ONNX inference** (preferred for Triton):
```python
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
outputs = sess.run(None, {"images": img_np})   # input name may vary; check sess.get_inputs()[0].name
raw = torch.from_numpy(outputs[0])             # (batch, 84, 6300)
```

**TorchScript inference**:
```python
model = torch.jit.load("model.torchscript")
raw = model(img_t)    # (batch, 84, 6300)
```

---

## HuggingFace Dataset Integration

```python
from datasets import Dataset
import PIL.Image

def run_detection(batch, model, conf=0.25, iou=0.45):
    results = []
    for pil_img in batch["image"]:
        img = np.array(pil_img)[..., ::-1]          # RGB→BGR for letterbox
        lb_img, scale, (pad_x, pad_y) = letterbox(img)
        img_t = torch.from_numpy(lb_img[..., ::-1].copy()).permute(2,0,1).float()/255
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            raw = model(img_t)                       # (1, 84, 6300)

        dets = non_max_suppression(raw, conf, iou)[0]  # (N, 6)
        if len(dets):
            dets[:, :4] = scale_boxes((640, 640), dets[:, :4], img.shape[:2])

        results.append({
            "boxes": dets[:, :4].cpu().numpy().tolist(),
            "scores": dets[:, 4].cpu().numpy().tolist(),
            "classes": dets[:, 5].cpu().numpy().astype(int).tolist(),
        })
    return {"detections": results}

ds = Dataset.from_hub("org/dataset")
ds = ds.map(run_detection, batched=True, batch_size=8, fn_kwargs={"model": model})
ds.push_to_hub("org/results")
```

---

## Triton Deployment

Use ONNX model artifact. Preprocessing (letterbox + normalize) is client-side or in a Triton ensemble.

```
triton-model-repo/
  yolov8_detect/
    1/
      model.onnx
    config.pbtxt        # input: "images" [1,3,640,640] FP32; output: "output0" [1,84,6300] FP32
```

NMS runs client-side (or as a second Triton model in the ensemble using the NMS Python backend).

---

## What to Skip

Irrelevant to inference: `engine/trainer.py`, `engine/tuner.py`, `models/yolo/*/train.py`,
`models/yolo/*/val.py`, `solutions/`, `trackers/`, `data/dataset.py`, `data/augment.py` (except LetterBox).
