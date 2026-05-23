# YOLO Version Deltas

## Naming Notes

**Missing versions (v4, v7):** YOLOv4 and YOLOv7 were published by different research groups and were never part of Ultralytics. Only versions with direct Ultralytics involvement are in this repo. v6 and v9/v10 were external contributions that Ultralytics integrated.

**RT-DETR:** Not YOLO. It's Baidu's DETR variant with a full transformer decoder (`RTDETRDecoder`), not a convolutional head. Included because it targets the same real-time detection use case and plugs into the same `engine/` framework, but architecturally distinct enough to keep a separate name.

**v11 → v12 → 26 (not v26):** YOLO12 exists (released Feb 2025). After that, Ultralytics switched from sequential version numbers to **year-based naming**: YOLO26 = released in 2026. The "26" is not version 26.

---

## Architecture Delta Table

| Version | Released | Backbone Block | Key Head | Anchor | NMS | Tasks | Scales | Standout Delta |
|---|---|---|---|---|---|---|---|---|
| **YOLOv3** | 2018 | Darknet-53 (`Bottleneck`) | Coupled detect | Anchor-based | Yes | detect | — | FPN multi-scale at P3/P4/P5; fixed anchors per scale |
| **YOLOv5** | 2020 | CSP-Darknet (`C3`) | Coupled detect | Anchor-based | Yes | detect | n/s/m/l/x | Compound scaling (`depth × width × max_channels`); `C3` replaces `Bottleneck`; `SPPF` replaces `SPP` |
| **YOLOv6** | 2022 | Plain `Conv` stack | Decoupled detect | Anchor-free | Yes | detect | n/s/m/l/x | Third-party (Meituan); anchor-free; `RepConv` reparameterization at inference |
| **YOLOv8** | 2023 | `C2f` | Decoupled + DFL loss | Anchor-free | Yes | detect, seg, cls, pose, obb | n/s/m/l/x | `C2f` cross-stage dense connections; Distribution Focal Loss; first to add OBB task |
| **YOLOv9** | 2024 | GELAN (`RepNCSPELAN4` + `ADown`) | Detect | Anchor-free | Yes | detect, seg | t/s/m/c/e | GELAN backbone; `v9e` adds PGI (Programmable Gradient Info via `CBLinear`/`CBFuse`) |
| **YOLOv10** | 2024 | `C2f` + `SCDown` + `PSA` | `v10Detect` (dual assignment) | Anchor-free | NMS-free | detect | n/s/m/b/l/x | NMS-free via one-to-one label assignment; `SCDown`; `PSA` attention at P5 |
| **RT-DETR** | 2024 | `HGStem` + `HGBlock` | `RTDETRDecoder` (transformer) | Anchor-free | NMS-free | detect | l/x | Transformer decoder (DETR-style); `AIFI` intra-scale attention; `RepC3` FPN; not YOLO |
| **YOLO11** | 2024 | `C3k2` + `C2PSA` | Detect | Anchor-free | Yes | detect, seg, cls, pose, obb | n/s/m/l/x | `C3k2` (kernel-aware) replaces `C2f`; `C2PSA` attention appended to backbone; smaller nano (2.6M vs 3.2M) |
| **YOLO12** | Feb 2025 | `C3k2` + `A2C2f` | Detect | Anchor-free | Yes | detect, seg, cls, pose, obb | n/s/m/l/x | `A2C2f` area-attention: feature map split into regions, self-attention within each; detect-only pretrained weights |
| **YOLO26** | Jan 2026 | `C3k2` + enhanced `SPPF` + `C2PSA` | `Detect` / `Pose26` / `Segment26` / `OBB26` | Anchor-free | NMS-free | detect, seg, cls, pose, obb, **semantic** | n/s/m/l/x | Year-based naming (2026); end-to-end by default (`end2end: True`); 26-specific task heads; adds semantic segmentation; `YOLOE` open-vocab variant |

## Key Progression

- **v3 → v5**: CSP blocks, compound scaling, `SPPF`
- **v5 → v6**: anchor-free, decoupled head
- **v6 → v8**: `C2f` dense connections, DFL loss, OBB task
- **v8 → v9**: GELAN backbone, PGI gradient paths (v9e)
- **v9 → v10**: NMS-free dual-assignment, `PSA` attention at backbone
- **v10 / RT-DETR**: parallel NMS-free tracks — CNN-based vs transformer-decoder-based
- **v8 → v11**: `C3k2` + `C2PSA`, smaller nano
- **v11 → v12**: `A2C2f` area-attention replaces `C2f` at deep stages
- **v12 → 26**: year-based naming; end-to-end by default; semantic seg task; 26-specific heads

## Appendix

### Term Definitions

| Term | What it is |
|---|---|
| **A2C2f** (Area-Attention C2f) | v12's main block. Divides the feature map into N non-overlapping regions and runs self-attention independently within each region. Provides spatial attention with lower memory cost than full self-attention. |
| **ADown** | v9's learned downsampling block. Averages a 2×2 neighborhood then applies a strided conv — blends information before downsampling to avoid aliasing. Replaces the strided `Conv` used in v5/v8. |
| **AIFI** (Adaptive Intra-scale Feature Interaction) | RT-DETR's transformer module applied within a single feature scale (P5). Unlike cross-scale attention, it fuses spatial positions at the same resolution using multi-head self-attention with learned positional encodings. |
| **Anchor-based** | The head predicts offsets relative to a set of fixed prior boxes ("anchors") of predefined aspect ratios and sizes — one set per pyramid level. Requires tuning anchors to the dataset. |
| **Anchor-free** | The head directly predicts box center + width/height from each grid cell, without anchor priors. Simpler to tune, removes the anchor-clustering step. Introduced in this family at v6. |
| **Backbone** | The part of the network that extracts features from the raw image. It runs the image through progressively downsampled feature maps (P1–P5). Weights are often pre-trained on ImageNet. |
| **Bottleneck** | A three-conv block (1×1 → 3×3 → 1×1) that reduces channels before the expensive 3×3 conv, then expands back. Standard ResNet building block; used as Darknet-53's core unit in v3. |
| **C2f** | v8's replacement for `C3`. "C2" = two convolutions, "f" = fast/dense. Splits channels into two halves, runs one half through N bottleneck sub-layers, concatenates all intermediate outputs (dense connections), then projects back. More gradient paths than `C3`. |
| **C3** | The CSP bottleneck block used in v5. Two paths: one through N bottleneck layers, one a direct 1×1 conv. Merged by concatenation then another 1×1. |
| **C3k2** | v11/v12/v26's replacement for `C2f`. "k" = kernel-flexible (supports variable kernel sizes in sub-layers). Runs two bottleneck sub-layers by default; the kernel size can differ per stage, allowing the block to adapt its receptive field. |
| **Compound Scaling** | A strategy (from EfficientNet) to scale a model's depth, width, and input resolution together using a single coefficient, rather than tuning each dimension independently. v5 introduced this to YOLO via the `scales:` block in YAMLs. |
| **Coupled head** | Classification and bounding-box regression share the same convolutional layers before their final outputs. Optimizing both tasks together can cause interference. Used in v3 and v5. |
| **CSP** (Cross-Stage Partial Network) | A block design that splits the input channels into two paths: one goes through the residual/bottleneck stack, the other bypasses it directly. They are concatenated at the end. Reduces computation while preserving gradient flow. Used in v5's `C3` block. |
| **Darknet** | The C-based deep learning framework written by Joseph Redmon (YOLO's original author). YOLOv3's backbone ("Darknet-53") is a 53-layer residual network defined in that framework. Ultralytics re-implemented it in PyTorch for v3 support. |
| **Decoupled head** | Classification and box regression use separate branch layers before their outputs. Reduces task interference; generally improves accuracy at a small cost in parameters. Introduced at v6, kept in all later versions. |
| **DETR** (Detection Transformer) | Object detection paradigm (Carion et al. 2020) that replaces the CNN head + NMS pipeline with a transformer decoder. A fixed set of learned "object queries" attend to image features and directly output box + class predictions — NMS-free by design. RT-DETR is a real-time variant. |
| **DFL** (Distribution Focal Loss) | Instead of predicting a single value for each box edge, the head predicts a discrete probability distribution over possible distances. The expected value is the final coordinate. Makes regression more robust to ambiguous boundaries. Introduced at v8. |
| **ELAN** (Efficient Layer Aggregation Network) | Predecessor to GELAN (used in YOLOv7, not in this repo directly). Aggregates outputs from multiple intermediate layers via concatenation, maximizing gradient reuse. `RepNCSPELAN4` in v9 nests ELAN sub-paths inside a CSP wrapper. |
| **End-to-end (E2E)** | Training and inference without a separate post-processing step (NMS). The model is trained with one-to-one matching so its raw outputs are final predictions. YOLO26 enables this by default. |
| **FPN** (Feature Pyramid Network) | A neck design (Lin et al. 2017) that adds top-down lateral connections: the deep, semantically-rich P5 features are upsampled and merged into P4 and P3. Gives each scale access to high-level context. |
| **GELAN** (Generalized Efficient Layer Aggregation Network) | v9's backbone design. Replaces CSP bottlenecks with `RepNCSPELAN4` — a block that nests multiple ELAN sub-paths inside a CSP wrapper, maximizing gradient paths and parameter efficiency. |
| **Head** | The final layers that produce predictions: bounding box coordinates, class probabilities, and (for seg/pose/obb) mask/keypoint outputs. |
| **Instance segmentation** | Predicts a per-pixel mask for each individual detected object instance. Two objects of the same class get separate masks. Implemented in YOLO via a prototype mask branch (`Proto`) + per-detection mask coefficients. |
| **IoU** (Intersection over Union) | Ratio of the overlap area to the union area between two boxes. Used both as a detection metric threshold (mAP) and as a training loss signal (box regression). Variants: GIoU, DIoU, CIoU add penalties for non-overlapping/distant/aspect-ratio-mismatched boxes. |
| **mAP** (mean Average Precision) | Standard detection metric. For each class, AP is the area under the precision-recall curve at a given IoU threshold. mAP50 averages over classes at IoU=0.5; mAP50-95 further averages over IoU thresholds 0.50–0.95 in 0.05 steps. |
| **Neck** | The part between backbone and head. Its job is to fuse features from multiple backbone scales so the head can detect objects of different sizes. FPN and PANet are common neck designs. |
| **NMS** (Non-Maximum Suppression) | Post-processing step that removes duplicate detections: for each object, keep only the box with the highest confidence and suppress overlapping boxes above an IoU threshold. Required when the head produces many overlapping candidates. |
| **NMS-free** | The head is trained with a one-to-one matching strategy so each object gets exactly one prediction. No suppression step needed at inference, reducing latency and eliminating the NMS hyperparameters. Introduced at v10. |
| **OBB** (Oriented Bounding Box) | A bounding box with a rotation angle, used for objects that are not axis-aligned (e.g. aerial/satellite imagery). First supported in v8. |
| **P3 / P4 / P5** | Shorthand for feature map pyramid levels. "P" = pyramid, number = log₂(downsampling factor). P3 = 8× smaller than input (detects small objects), P4 = 16×, P5 = 32× (detects large objects). YOLO heads attach to all three simultaneously. |
| **PANet** (Path Aggregation Network) | Extends FPN by adding a second bottom-up path after the top-down one. Information flows both up (FPN) and back down (PAN), so shallow detail also reaches deeper layers. Used in v5 and most later YOLO versions. |
| **PGI** (Programmable Gradient Information) | v9e auxiliary training mechanism. Adds a parallel "auxiliary backbone" (via `CBLinear`/`CBFuse`) that feeds gradient signals to intermediate layers, ensuring deep features retain complete information. Removed at inference. |
| **PSA** (Partial Self-Attention) | Splits channels in half; applies multi-head self-attention to only one half, then concatenates with the other. Adds global context at low cost. Used at the P5 stage in v10 and in `C2PSA` blocks in v11/v26. |
| **RepConv** (Re-parameterizable Conv) | During training, uses parallel branches (e.g. 3×3 + 1×1 + identity). At inference, all branches are algebraically merged into a single 3×3 conv — same output, fewer ops. Used in v6 and v9. |
| **SCDown** (Stride-Conv Downsampling) | v10's replacement for max-pooling between pyramid stages. A depthwise separable conv with stride 2 that learns the downsampling, rather than using a fixed max operation. Reduces spatial resolution with fewer parameters than a full strided conv. |
| **Semantic segmentation** | Assigns a class label to every pixel in the image without distinguishing individual instances. Added as a task in YOLO26 via `SemanticSegment` head, trained on datasets like Cityscapes. |
| **SimOTA** | Simplified Optimal Transport Assignment (YOLOX, 2021). Frames label assignment as an optimal transport problem: assign ground-truth boxes to predictions by minimizing a global cost (classification + regression). "Simplified" means it approximates the full OT solver with a top-k selection per ground-truth. Inspired the TAL assigner used from v8 onward. |
| **SPP** (Spatial Pyramid Pooling) | Applies max-pooling with several kernel sizes (e.g. 5, 9, 13) in parallel, then concatenates the results. Captures multi-scale context at a single feature map location without changing spatial resolution. |
| **SPPF** (Spatial Pyramid Pooling - Fast) | Equivalent to SPP but stacks a single small max-pool (5×5) repeatedly instead of using multiple large kernels in parallel. Same receptive field, lower compute. Replaces SPP from v5 onward. |
| **Task-aligned Assigner (TAL)** | The label-assignment strategy used from v8 onward. For each ground-truth box, it selects the top-k anchors/cells whose predictions are most aligned (jointly considering classification score and IoU), rather than using fixed IoU thresholds. |
| **YOLOE** | Open-vocabulary detection variant (available for v8, v11, v26). Adds a text-image contrastive head so the model can detect arbitrary classes described by text prompts at inference, not just the fixed classes it was trained on. |
| **YOLOWorld** | Open-vocabulary variant (v8-based). Uses `WorldDetect` head + `ImagePoolingAttn` to align visual features with CLIP-style text embeddings, enabling zero-shot detection of novel categories. |