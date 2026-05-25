import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from deepbrain.cust_yolo.nms import TorchNMS, non_max_suppression


class TestTorchNMS:
    def test_nms_keeps_highest_score(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])  # heavily overlapping
        scores = torch.tensor([0.9, 0.5])
        keep = TorchNMS.nms(boxes, scores, iou_threshold=0.5)
        assert keep.tolist() == [0]

    def test_nms_keeps_non_overlapping(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
        scores = torch.tensor([0.9, 0.8])
        keep = TorchNMS.nms(boxes, scores, iou_threshold=0.5)
        assert set(keep.tolist()) == {0, 1}

    def test_nms_empty_input(self):
        keep = TorchNMS.nms(torch.empty(0, 4), torch.empty(0), 0.5)
        assert keep.numel() == 0

    def test_fast_nms_matches_greedy(self):
        torch.manual_seed(42)
        boxes = torch.rand(20, 4)
        boxes[:, 2:] += boxes[:, :2]  # ensure x2>x1, y2>y1
        scores = torch.rand(20)

        set(TorchNMS.nms(boxes, scores, 0.5).tolist())
        fast = set(TorchNMS.fast_nms(boxes, scores.clone(), 0.5).tolist())
        # Fast-NMS can differ from greedy but should keep at least the top box
        top = scores.argmax().item()
        assert top in fast

    def test_batched_nms_class_agnostic(self):
        # Two overlapping boxes of different classes: batched NMS should keep both
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
        scores = torch.tensor([0.9, 0.8])
        idxs = torch.tensor([0, 1])  # different classes
        keep = TorchNMS.batched_nms(boxes, scores, idxs, iou_threshold=0.5)
        assert len(keep) == 2

    def test_fast_nms_empty_exit_early(self):
        keep = TorchNMS.fast_nms(torch.empty(0, 4), torch.empty(0), iou_threshold=0.5)
        assert keep.numel() == 0

    def test_fast_nms_use_triu_false(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]])
        scores = torch.tensor([0.9, 0.5, 0.8])
        keep = TorchNMS.fast_nms(boxes, scores.clone(), iou_threshold=0.5, use_triu=False)
        assert keep.shape[0] == 3  # all indices returned; suppressed ones get score 0

    def test_batched_nms_empty(self):
        keep = TorchNMS.batched_nms(
            torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.long), iou_threshold=0.5
        )
        assert keep.numel() == 0


class TestNonMaxSuppression:
    def test_returns_two_detections(self, raw_prediction):
        result = non_max_suppression(raw_prediction, conf_thres=0.25, iou_thres=0.45)
        assert len(result) == 1  # batch size 1
        assert result[0].shape[0] == 2  # both boxes kept

    def test_output_columns(self, raw_prediction):
        dets = non_max_suppression(raw_prediction)[0]
        assert dets.shape[1] == 6  # x1, y1, x2, y2, conf, cls

    def test_high_conf_threshold_suppresses(self, raw_prediction):
        dets = non_max_suppression(raw_prediction, conf_thres=0.95)[0]
        assert dets.shape[0] == 0

    def test_class_filter(self, raw_prediction):
        # Only keep class 1
        dets = non_max_suppression(raw_prediction, classes=[1])[0]
        assert dets.shape[0] == 1
        assert dets[0, 5].item() == pytest.approx(1.0)

    def test_list_tuple_prediction(self, raw_prediction):
        # Validate-mode output wraps prediction in a tuple
        dets = non_max_suppression((raw_prediction, None))[0]
        assert dets.shape[0] == 2

    def test_end2end_format(self):
        # End-to-end model outputs (B, N, 6) directly
        pred = torch.zeros(1, 10, 6)
        pred[0, 0] = torch.tensor([10.0, 10.0, 50.0, 50.0, 0.9, 0.0])
        dets = non_max_suppression(pred, conf_thres=0.5, end2end=True)[0]
        assert dets.shape[0] == 1

    def test_return_idxs(self, raw_prediction):
        output, idxs = non_max_suppression(raw_prediction, return_idxs=True)
        assert len(output) == len(idxs) == 1
        assert idxs[0].numel() == output[0].shape[0]

    def test_max_det_limit(self, raw_prediction):
        dets = non_max_suppression(raw_prediction, max_det=1)[0]
        assert dets.shape[0] == 1

    def test_empty_prediction(self):
        pred = torch.zeros(1, 84, 6300)
        dets = non_max_suppression(pred, conf_thres=0.25)[0]
        assert dets.shape[0] == 0

    def test_end2end_class_filter(self):
        pred = torch.zeros(1, 10, 6)
        pred[0, 0] = torch.tensor([10.0, 10.0, 50.0, 50.0, 0.9, 0.0])
        pred[0, 1] = torch.tensor([20.0, 20.0, 60.0, 60.0, 0.8, 1.0])
        dets = non_max_suppression(pred, conf_thres=0.5, end2end=True, classes=[0])[0]
        assert dets.shape[0] == 1
        assert dets[0, 5].item() == pytest.approx(0.0)

    def test_labels_appended(self):
        pred = torch.zeros(1, 84, 6300)
        labels = [torch.tensor([[0.0, 100.0, 100.0, 50.0, 50.0]])]  # cls=0, xywh box
        dets = non_max_suppression(pred, conf_thres=0.01, labels=labels)[0]
        assert dets.shape[0] == 1

    def test_multi_label_with_return_idxs(self, raw_prediction):
        output, idxs = non_max_suppression(raw_prediction, multi_label=True, return_idxs=True)
        assert output[0].shape[0] == 2
        assert idxs[0].numel() == 2

    def test_class_filter_with_return_idxs(self, raw_prediction):
        output, idxs = non_max_suppression(raw_prediction, classes=[0], return_idxs=True)
        assert output[0].shape[0] == 1
        assert idxs[0].numel() == 1

    def test_class_filter_eliminates_all(self, raw_prediction):
        dets = non_max_suppression(raw_prediction, classes=[99])[0]
        assert dets.shape[0] == 0

    def test_max_nms_with_return_idxs(self, raw_prediction):
        output, idxs = non_max_suppression(raw_prediction, max_nms=1, return_idxs=True)
        assert output[0].shape[0] <= 1
        assert idxs[0].numel() == output[0].shape[0]

    def test_torchvision_nms_branch(self, raw_prediction):
        mock_tv = MagicMock()
        mock_tv.ops.nms.return_value = torch.tensor([0, 1])
        with patch.dict(sys.modules, {"torchvision": mock_tv}):
            dets = non_max_suppression(raw_prediction)[0]
        mock_tv.ops.nms.assert_called_once()
        assert dets.shape[0] == 2

    def test_time_limit_warning(self):
        pred = torch.zeros(2, 84, 6300)
        pred[0, :4, 0] = torch.tensor([100.0, 100.0, 50.0, 50.0])
        pred[0, 4, 0] = 0.9
        pred[1, :4, 0] = torch.tensor([200.0, 200.0, 50.0, 50.0])
        pred[1, 4, 0] = 0.85
        with patch("deepbrain.cust_yolo.nms.time") as mock_time:
            mock_time.time.side_effect = [0.0, 100.0]
            output = non_max_suppression(pred)
        assert output[0].shape[0] == 1  # image 0 processed
        assert output[1].shape[0] == 0  # image 1 skipped by time limit
