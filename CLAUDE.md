# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

```bash
pip install -e ".[dev]"          # install in editable mode with dev deps
pip install -e ".[dev,export]"   # include export dependencies (ONNX, TensorRT, etc.)
```

## Common Commands

```bash
# Run all tests (includes doctests in source files)
pytest tests/

# Run a specific test file
pytest tests/test_python.py -k test_model_forward

# Run slow tests (skipped by default)
pytest tests/ --slow

# Lint
ruff check ultralytics/

# Format
ruff format ultralytics/

# CLI usage (two equivalent entrypoints)
yolo task=detect mode=train model=yolo26n.pt data=coco8.yaml epochs=100
ultralytics task=detect mode=train model=yolo26n.pt data=coco8.yaml
```

## Architecture Overview

### Package Structure

The `ultralytics/` package is organized around a clean separation of concerns:

- **`engine/`** — Base classes for the full ML lifecycle: `Model` (user-facing API), `BaseTrainer`, `BasePredictor`, `BaseValidator`, `Results`, `Exporter`, `Tuner`
- **`models/`** — Task-specific subclasses: `yolo/` (detect, segment, classify, pose, obb, semantic, world, yoloe), `sam/`, `fastsam/`, `rtdetr/`, `nas/`
- **`nn/`** — PyTorch building blocks: `modules/` (conv, block, head, transformer), `autobackend.py` (multi-framework inference), `tasks.py` (model parsing from YAML)
- **`data/`** — Dataset classes (`YOLODataset`, `ClassificationDataset`, etc.), augmentation pipeline, dataloaders
- **`cfg/`** — `default.yaml` (all hyperparameters), `models/` (architecture YAMLs: v8, v9, v10, 11, 26, rt-detr), `datasets/` (dataset YAMLs)
- **`utils/`** — Loss functions, metrics, callbacks, ops, plotting, torch utilities
- **`solutions/`** — High-level CV applications (ObjectCounter, Heatmap, SpeedEstimator, etc.)
- **`trackers/`** — BOTSORT and ByteTrack implementations

### Model Hierarchy

Every model task (detect, segment, classify, pose, obb, semantic) follows the same pattern: a task-specific `Model` subclass in `engine/` delegates to task-specific `Trainer`, `Predictor`, and `Validator` classes under `models/yolo/<task>/`. `engine/model.py:Model` is the single user-facing API that wraps all of this.

The `YOLO` class in `ultralytics/models/yolo/model.py` maps task strings to these implementation classes. New model families (SAM, RT-DETR, etc.) each have their own directory under `models/` with the same `train/predict/val` pattern.

### Configuration System

`ultralytics/cfg/__init__.py` defines:
- `TASKS = {"detect", "segment", "classify", "pose", "obb", "semantic"}`
- `MODES = {"train", "val", "predict", "export", "track", "benchmark"}`
- `TASK2MODEL`, `TASK2DATA`, `TASK2METRIC` mappings

`cfg/default.yaml` is the canonical source of all training/inference hyperparameters. Model architecture is defined separately in YAML files under `cfg/models/` (parsed at runtime by `nn/tasks.py`).

### Neural Network Parsing

Model architectures are specified declaratively in YAML (e.g., `cfg/models/26/yolo26.yaml`). `nn/tasks.py:parse_model()` reads these YAMLs and instantiates the corresponding `nn/modules/` building blocks. The latest generation is YOLO26; YOLO11 is the previous generation.

### Callbacks

Training/validation/prediction lifecycle events are handled via a callbacks dict (`utils/callbacks/base.py`). Integration callbacks (W&B, MLflow, TensorBoard, etc.) live in `utils/callbacks/`. Add custom behavior by passing callbacks to the `Model` API or trainer.

### Export

`engine/exporter.py` supports 20+ export formats (ONNX, TensorRT, CoreML, TFLite, OpenVINO, NCNN, etc.) via `model.export(format="onnx")`. Requires `pip install "ultralytics[export]"`.

## Code Style

- Line length: 120 characters (ruff, yapf, isort all configured to 120)
- Docstrings: Google-style with `Args:`, `Returns:`, `Examples:` sections; types in parentheses
- Doctests are run as part of the test suite (`--doctest-modules` in pytest config), so Examples blocks in docstrings must be runnable
- The `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` header is required in every source file

## Testing Notes

- Tests use `yolo26n.pt` / `yolo26n.yaml` as the default small model
- `tests/__init__.py` defines shared constants: `MODEL`, `CFG`, `SOURCE`, `TASK_MODEL_DATA`
- Tests marked `@pytest.mark.slow` are excluded unless `--slow` is passed
- `pytest.ini` enables `--doctest-modules`, so all module-level Examples are tested
