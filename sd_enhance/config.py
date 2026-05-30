"""配置管理：模型路径解析、设备选择、自动下载开关。"""

import os
from pathlib import Path
from typing import Optional

import torch

from .utils.device import get_device, get_dtype


# 已知模型文件名 → 下载 URL 映射（仅作参考，不做自动下载）
MODEL_REGISTRY = {
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "size_mb": 340,
        "dir": "GFPGAN",
    },
    "codeformer-v0.1.0.pth": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "size_mb": 340,
        "dir": "Codeformer",
    },
}


class Config:
    """全局配置。"""

    def __init__(
        self,
        webui_path: Optional[str] = None,
        models_dir: Optional[str] = None,
        device_prefer: str = "auto",
        half: bool = True,
        output_dir: Optional[str] = None,
    ):
        # 设备
        self.device = get_device(device_prefer)
        self.dtype = get_dtype(self.device, half)

        # 模型搜索路径
        self._models_search_paths: list[str] = []

        # 如果指定了 webui 路径，添加其模型目录
        if webui_path:
            webui_path = os.path.abspath(webui_path)
            self.webui_path = webui_path
            models_root = os.path.join(webui_path, "models")
            if os.path.isdir(models_root):
                self._models_search_paths.append(models_root)
                # 添加各子目录
                for sub in os.listdir(models_root):
                    sub_path = os.path.join(models_root, sub)
                    if os.path.isdir(sub_path):
                        self._models_search_paths.append(sub_path)
        else:
            self.webui_path = None

        # 程序自身模型目录
        if models_dir:
            self.models_dir = os.path.abspath(models_dir)
        else:
            self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

        if os.path.isdir(self.models_dir):
            self._models_search_paths.append(self.models_dir)
            for sub in os.listdir(self.models_dir):
                sub_path = os.path.join(self.models_dir, sub)
                if os.path.isdir(sub_path):
                    self._models_search_paths.append(sub_path)

        # 输出目录
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def find_model(self, filename: str) -> Optional[str]:
        """在搜索路径中查找模型文件。

        搜索顺序：
        1. webui/models/<subdir>/<filename>
        2. webui/models/<filename>
        3. <models_dir>/<subdir>/<filename>
        4. <models_dir>/<filename>

        Returns:
            文件完整路径，或 None
        """
        # 先查已知映射
        info = MODEL_REGISTRY.get(filename)
        subdir = info["dir"] if info else None

        for base_path in self._models_search_paths:
            # 如果在子目录里
            candidate = os.path.join(base_path, filename)
            if os.path.isfile(candidate):
                return candidate

            # 如果在子目录的子目录里（如 models/GFPGAN/GFPGANv1.4.pth）
            if subdir:
                candidate = os.path.join(base_path, subdir, filename)
                if os.path.isfile(candidate):
                    return candidate

        # 遍历所有搜索路径下的所有 .pth 文件
        for base_path in self._models_search_paths:
            if os.path.isdir(base_path):
                for f in os.listdir(base_path):
                    if f.lower() == filename.lower():
                        return os.path.join(base_path, f)

        return None

    def find_all_pth_files(self) -> list[str]:
        """查找所有搜索路径中的 .pth 文件。"""
        results = []
        for base_path in self._models_search_paths:
            if os.path.isdir(base_path):
                for f in os.listdir(base_path):
                    if f.endswith(".pth"):
                        results.append(os.path.join(base_path, f))
        return results

    def list_available_face_models(self) -> list[str]:
        """返回可用的面部修复模型名称列表。"""
        available = []
        if self.find_model("GFPGANv1.4.pth"):
            available.append("GFPGAN")
        if self.find_model("codeformer-v0.1.0.pth") or self.find_model("codeformer.pth"):
            available.append("CodeFormer")
        return available
