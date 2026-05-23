import torch
import torch.nn as nn


def load_model(path: str, device: str | torch.device = "cpu") -> nn.Module:
    """Load a YOLOv8 .pt checkpoint as a standalone nn.Module.

    The checkpoint must have been saved with ultralytics classes available, so
    ultralytics must be importable at load time. To remove that dependency
    permanently, export to ONNX or TorchScript instead (see README).

    Args:
        path: Path to the .pt file.
        device: Target device for the model.

    Returns:
        nn.Module in eval mode, on the requested device.
        Access model.names for the {int: str} class-name mapping.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model: nn.Module = ckpt["model"].float().eval()
    return model.to(device)


def load_onnx(path: str, device: str = "cpu"):
    """Create an ONNX Runtime inference session.

    Preferred for production / Triton deployments; zero ultralytics dependency.

    Args:
        path: Path to the .onnx file.
        device: "cpu" or "cuda".

    Returns:
        onnxruntime.InferenceSession ready for inference.
    """
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=providers)


def load_torchscript(path: str, device: str | torch.device = "cpu"):
    """Load a TorchScript model exported from ultralytics.

    Args:
        path: Path to the .torchscript file.
        device: Target device.

    Returns:
        torch.jit.ScriptModule in eval mode.
    """
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model
