from unittest.mock import MagicMock, patch

from deepbrain.cust_yolo.model import load_model, load_onnx, load_torchscript


class TestLoadModel:
    def test_loads_and_returns_model(self):
        mock_model = MagicMock()
        ckpt = {"model": mock_model}
        with patch("deepbrain.cust_yolo.model.torch.load", return_value=ckpt):
            load_model("fake.pt", device="cpu")
        mock_model.float.assert_called_once()
        mock_model.float.return_value.eval.assert_called_once()


class TestLoadOnnx:
    def test_cpu_uses_cpu_provider(self):
        mock_ort = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            load_onnx("fake.onnx", device="cpu")
        mock_ort.InferenceSession.assert_called_once_with("fake.onnx", providers=["CPUExecutionProvider"])

    def test_cuda_includes_cuda_provider(self):
        mock_ort = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            load_onnx("fake.onnx", device="cuda")
        providers = mock_ort.InferenceSession.call_args[1]["providers"]
        assert "CUDAExecutionProvider" in providers


class TestLoadTorchscript:
    def test_loads_and_evals(self):
        mock_model = MagicMock()
        with patch("deepbrain.cust_yolo.model.torch.jit.load", return_value=mock_model):
            result = load_torchscript("fake.torchscript", device="cpu")
        mock_model.eval.assert_called_once()
        assert result is mock_model
