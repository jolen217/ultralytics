# Test & Analysis Report

**Date:** 2026-05-24  
**Python:** 3.11.15  
**pytest:** 9.0.3  
**Platform:** darwin

---

## Summary

| Metric | Result |
|---|---|
| Total tests | 61 |
| Passed | 61 |
| Failed | 0 |
| Warnings | 10 |
| Coverage | **100%** |
| Lint (ruff) | All checks passed |

---

## Test Results by Module

### `tests/test_hf_integration.py` — 5 tests

| Test | Status |
|---|---|
| `test_output_keys` | PASSED |
| `test_detection_structure` | PASSED |
| `test_boxes_in_original_space` | PASSED |
| `test_empty_when_no_detections` | PASSED |
| `test_conf_threshold_filters` | PASSED |

### `tests/test_model.py` — 4 tests

| Test | Status |
|---|---|
| `TestLoadModel::test_loads_and_returns_model` | PASSED |
| `TestLoadOnnx::test_cpu_uses_cpu_provider` | PASSED |
| `TestLoadOnnx::test_cuda_includes_cuda_provider` | PASSED |
| `TestLoadTorchscript::test_loads_and_evals` | PASSED |

### `tests/test_nms.py` — 25 tests

| Test | Status |
|---|---|
| `TestTorchNMS::test_nms_keeps_highest_score` | PASSED |
| `TestTorchNMS::test_nms_keeps_non_overlapping` | PASSED |
| `TestTorchNMS::test_nms_empty_input` | PASSED |
| `TestTorchNMS::test_fast_nms_matches_greedy` | PASSED |
| `TestTorchNMS::test_batched_nms_class_agnostic` | PASSED |
| `TestTorchNMS::test_fast_nms_empty_exit_early` | PASSED |
| `TestTorchNMS::test_fast_nms_use_triu_false` | PASSED |
| `TestTorchNMS::test_batched_nms_empty` | PASSED |
| `TestNonMaxSuppression::test_returns_two_detections` | PASSED |
| `TestNonMaxSuppression::test_output_columns` | PASSED |
| `TestNonMaxSuppression::test_high_conf_threshold_suppresses` | PASSED |
| `TestNonMaxSuppression::test_class_filter` | PASSED |
| `TestNonMaxSuppression::test_list_tuple_prediction` | PASSED |
| `TestNonMaxSuppression::test_end2end_format` | PASSED |
| `TestNonMaxSuppression::test_return_idxs` | PASSED |
| `TestNonMaxSuppression::test_max_det_limit` | PASSED |
| `TestNonMaxSuppression::test_empty_prediction` | PASSED |
| `TestNonMaxSuppression::test_end2end_class_filter` | PASSED |
| `TestNonMaxSuppression::test_labels_appended` | PASSED |
| `TestNonMaxSuppression::test_multi_label_with_return_idxs` | PASSED |
| `TestNonMaxSuppression::test_class_filter_with_return_idxs` | PASSED |
| `TestNonMaxSuppression::test_class_filter_eliminates_all` | PASSED |
| `TestNonMaxSuppression::test_max_nms_with_return_idxs` | PASSED |
| `TestNonMaxSuppression::test_torchvision_nms_branch` | PASSED |
| `TestNonMaxSuppression::test_time_limit_warning` | PASSED |

### `tests/test_ops.py` — 16 tests

| Test | Status |
|---|---|
| `TestXywh2Xyxy::test_known_conversion` | PASSED |
| `TestXywh2Xyxy::test_batch` | PASSED |
| `TestXywh2Xyxy::test_numpy_input` | PASSED |
| `TestXywh2Xyxy::test_wrong_last_dim_raises` | PASSED |
| `TestClipBoxes::test_clamps_above` | PASSED |
| `TestClipBoxes::test_clamps_below` | PASSED |
| `TestClipBoxes::test_numpy_boxes` | PASSED |
| `TestClipBoxes::test_in_bounds_unchanged` | PASSED |
| `TestClipBoxes::test_macos14_clamp_branch` | PASSED |
| `TestScaleBoxes::test_identity` | PASSED |
| `TestScaleBoxes::test_half_scale` | PASSED |
| `TestScaleBoxes::test_removes_padding` | PASSED |
| `TestBoxIou::test_identical_boxes` | PASSED |
| `TestBoxIou::test_non_overlapping` | PASSED |
| `TestBoxIou::test_half_overlap` | PASSED |
| `TestBoxIou::test_output_shape` | PASSED |

### `tests/test_preprocess.py` — 11 tests

| Test | Status |
|---|---|
| `TestLetterbox::test_output_shape` | PASSED |
| `TestLetterbox::test_scale_is_positive` | PASSED |
| `TestLetterbox::test_scale_limits_by_min_ratio` | PASSED |
| `TestLetterbox::test_padding_is_symmetric` | PASSED |
| `TestLetterbox::test_non_square_input_has_padding` | PASSED |
| `TestLetterbox::test_custom_new_shape` | PASSED |
| `TestLetterbox::test_padding_value` | PASSED |
| `TestToTensor::test_shape` | PASSED |
| `TestToTensor::test_dtype` | PASSED |
| `TestToTensor::test_range` | PASSED |
| `TestToTensor::test_bgr_to_rgb` | PASSED |

---

## Coverage

```
Name                                        Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------
src/deepbrain/cust_yolo/__init__.py             5      0   100%
src/deepbrain/cust_yolo/hf_integration.py      21      0   100%
src/deepbrain/cust_yolo/model.py               14      0   100%
src/deepbrain/cust_yolo/nms.py                141      0   100%
src/deepbrain/cust_yolo/ops.py                 49      0   100%
src/deepbrain/cust_yolo/preprocess.py          18      0   100%
-------------------------------------------------------------------------
TOTAL                                         248      0   100%
```

---

## Warnings

10 `DeprecationWarning` instances from `tests/test_hf_integration.py`, all from the same site:

```
src/deepbrain/cust_yolo/hf_integration.py:44:
  DeprecationWarning: __array__ implementation doesn't accept a copy keyword,
  so passing copy=False failed. __array__ must implement 'dtype' and 'copy'
  keyword arguments.
  (numpy 2.0 migration: adapting-to-changes-in-the-copy-keyword)
```

Each of the 5 HF integration tests triggers the warning twice (once per `np.array(pil_img)` call path). No impact on correctness; fix by upgrading the numpy array conversion in `hf_integration.py:44` to pass explicit `copy` and `dtype` kwargs when numpy ≥ 2.0.

---

## Lint

```
uv run ruff check src/ tests/
All checks passed!
```
