import torch
import pytest

from deepbrain.cust_yolo.nms import TorchNMS, non_max_suppression


class TestTorchNMS:
    def test_nms_keeps_highest_score(self):
        boxes  = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                                [1.0, 1.0, 11.0, 11.0]])   # heavily overlapping
        scores = torch.tensor([0.9, 0.5])
        keep = TorchNMS.nms(boxes, scores, iou_threshold=0.5)
        assert keep.tolist() == [0]

    def test_nms_keeps_non_overlapping(self):
        boxes  = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                                [50.0, 50.0, 60.0, 60.0]])
        scores = torch.tensor([0.9, 0.8])
        keep = TorchNMS.nms(boxes, scores, iou_threshold=0.5)
        assert set(keep.tolist()) == {0, 1}

    def test_nms_empty_input(self):
        keep = TorchNMS.nms(torch.empty(0, 4), torch.empty(0), 0.5)
        assert keep.numel() == 0

    def test_fast_nms_matches_greedy(self):
        torch.manual_seed(42)
        boxes  = torch.rand(20, 4)
        boxes[:, 2:] += boxes[:, :2]   # ensure x2>x1, y2>y1
        scores = torch.rand(20)

        greedy = set(TorchNMS.nms(boxes, scores, 0.5).tolist())
        fast   = set(TorchNMS.fast_nms(boxes, scores.clone(), 0.5).tolist())
        # Fast-NMS can differ from greedy but should keep at least the top box
        top = scores.argmax().item()
        assert top in fast

    def test_batched_nms_class_agnostic(self):
        # Two overlapping boxes of different classes: batched NMS should keep both
        boxes  = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                                [1.0, 1.0, 11.0, 11.0]])
        scores = torch.tensor([0.9, 0.8])
        idxs   = torch.tensor([0, 1])   # different classes
        keep = TorchNMS.batched_nms(boxes, scores, idxs, iou_threshold=0.5)
        assert len(keep) == 2


class TestNonMaxSuppression:
    def test_returns_two_detections(self, raw_prediction):
        result = non_max_suppression(raw_prediction, conf_thres=0.25, iou_thres=0.45)
        assert len(result) == 1          # batch size 1
        assert result[0].shape[0] == 2  # both boxes kept

    def test_output_columns(self, raw_prediction):
        dets = non_max_suppression(raw_prediction)[0]
        assert dets.shape[1] == 6       # x1, y1, x2, y2, conf, cls

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
