"""设备管理和显存工具。"""

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """返回最佳可用设备。

    Args:
        prefer: "auto" | "cuda" | "mps" | "cpu"

    Returns:
        torch.device
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_dtype(device: torch.device, half: bool = True) -> torch.dtype:
    """根据设备和设置返回推荐的数据类型。"""
    if half and device.type == "cuda":
        return torch.float16
    return torch.float32


def torch_gc():
    """清理显存缓存。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        import gc
        gc.collect()


def autocast(device: torch.device, enabled: bool = True):
    """混合精度上下文管理器。"""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.no_grad()
