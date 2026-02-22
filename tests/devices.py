import os
import shutil

def has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def has_npu() -> bool:
    # NPU is available typically if /dev/accel exists or openvino is in path in some systems,
    # but for OpenVINO with Intel Core Ultra we check for the OpenVINO package presence.
    return os.path.exists("/dev/accel") or shutil.which("openvino") is not None or _has_openvino()

def _has_openvino() -> bool:
    try:
        import openvino
        return True
    except ImportError:
        return False

# Base devices everyone has
DEVICES = ["cpu"]

if has_cuda():
    DEVICES.append("cuda")

if has_npu():
    # Let's use the exact string sherpa-onnx expects for its provider argument
    DEVICES.append("openvino")

print(f"Detected hardware providers: {DEVICES}")
