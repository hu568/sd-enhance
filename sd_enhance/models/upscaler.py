"""放大器管理：内置放大器（None/Lanczos/Nearest）+ Spandrel 加载的模型。"""

import os
import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..config import Config
from ..utils.device import autocast

logger = logging.getLogger(__name__)

LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class Upscaler:
    """放大器基类。"""

    name: str = "Base"
    scale: int = 4  # 模型原生放大倍数

    def upscale(self, img: Image.Image, scale: float) -> Image.Image:
        """对图片执行放大。

        Args:
            img: 输入图片
            scale: 目标放大倍数

        Returns:
            放大后的图片
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 内置放大器（无需模型文件）
# ---------------------------------------------------------------------------

class UpscalerNone(Upscaler):
    name = "None"

    def upscale(self, img, scale):
        return img


class UpscalerLanczos(Upscaler):
    name = "Lanczos"

    def upscale(self, img, scale):
        w, h = img.size
        return img.resize((int(w * scale), int(h * scale)), LANCZOS)


class UpscalerNearest(Upscaler):
    name = "Nearest"

    def upscale(self, img, scale):
        w, h = img.size
        return img.resize((int(w * scale), int(h * scale)), NEAREST)


# ---------------------------------------------------------------------------
# Spandrel 加载的放大器
# ---------------------------------------------------------------------------

class UpscalerSpandrel(Upscaler):
    """通过 Spandrel 加载任意架构（ESRGAN / RealESRGAN / SwinIR / DAT …）的放大器。"""

    def __init__(self, name: str, model_path: str, device: torch.device, dtype: torch.dtype):
        self.name = name
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._model = None
        self.scale = 4  # 会被实际加载的模型覆盖

    def _load(self):
        if self._model is not None:
            return self._model

        import spandrel

        logger.info("加载放大器模型: %s", self.model_path)
        model = spandrel.ModelLoader().load_from_file(self.model_path)

        # 尝试获取模型的输入通道数并移动到目标设备
        model = model.to(self.device)
        if self.dtype == torch.float16 and self.device.type == "cuda":
            model = model.half()

        # 记录模型的放大倍数
        if hasattr(model, "scale") and model.scale:
            self.scale = model.scale

        self._model = model
        return model

    @torch.inference_mode()
    def upscale(self, img: Image.Image, scale: float) -> Image.Image:
        model = self._load()

        target_w = int(img.width * scale)
        target_h = int(img.height * scale)

        # 在循环中运行模型推理，处理 scale > native_scale 的情况
        current = img
        native = self.scale or 4
        for _ in range(3):
            if current.width >= target_w and current.height >= target_h:
                break

            # PIL → 张量 (1, C, H, W), RGB, [0,1]
            arr = np.array(current.convert("RGB")).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
            arr = np.expand_dims(arr, 0)  # add batch
            tensor = torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

            with autocast(self.device):
                output_tensor = model(tensor)

            # 张量 → PIL
            output_tensor = output_tensor.squeeze(0).cpu().float().clamp(0, 1)
            output_arr = (output_tensor * 255).byte().permute(1, 2, 0).numpy()
            current = Image.fromarray(output_arr, "RGB")

            # 如果模型没有放大图片（scale == 1），避免死循环
            if current.size == img.size:
                break

        # 如果结果尺寸不等于目标尺寸，用 Lanczos 缩放到目标尺寸
        if current.width != target_w or current.height != target_h:
            current = current.resize((target_w, target_h), LANCZOS)

        # 确保能被 8 整除
        final_w = target_w // 8 * 8
        final_h = target_h // 8 * 8
        if final_w != current.width or final_h != current.height:
            current = current.resize((final_w, final_h), LANCZOS)

        return current


# ---------------------------------------------------------------------------
# 管理器
# ---------------------------------------------------------------------------

class UpscalerManager:
    """管理所有可用的放大器。"""

    def __init__(self, config: Config):
        self.config = config
        self.upscalers: list[Upscaler] = []
        self._init()

    def _init(self):
        # 1. 内置放大器
        self.upscalers.append(UpscalerNone())
        self.upscalers.append(UpscalerLanczos())
        self.upscalers.append(UpscalerNearest())

        # 2. 扫描 .pth 文件
        for pth_path in self.config.find_all_pth_files():
            name = os.path.splitext(os.path.basename(pth_path))[0]
            try:
                upscaler = UpscalerSpandrel(
                    name, pth_path, self.config.device, self.config.dtype
                )
                self.upscalers.append(upscaler)
            except Exception as e:
                logger.warning("加载放大器 %s 失败: %s", pth_path, e)

    def get_names(self) -> list[str]:
        return [u.name for u in self.upscalers]

    def get_by_name(self, name: str) -> Optional[Upscaler]:
        for u in self.upscalers:
            if u.name == name:
                return u
        return None
