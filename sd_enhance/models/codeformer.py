"""CodeFormer 人脸修复模型封装。"""

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..config import Config
from ..utils.device import autocast
from .face_utils import create_face_helper, restore_with_face_helper

logger = logging.getLogger(__name__)


class CodeFormerModel:
    """CodeFormer 人脸修复模型。"""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self._model = None
        self._face_helper = None

    def is_available(self) -> bool:
        return (
            self.config.find_model("codeformer-v0.1.0.pth") is not None
            or self.config.find_model("codeformer.pth") is not None
        )

    def model_path(self) -> Optional[str]:
        return (
            self.config.find_model("codeformer-v0.1.0.pth")
            or self.config.find_model("codeformer.pth")
        )

    def _load(self):
        if self._model is not None:
            return self._model

        path = self.model_path()
        if not path:
            raise FileNotFoundError("未找到 CodeFormer 模型文件")

        import spandrel

        logger.info("加载 CodeFormer 模型: %s", path)
        model = spandrel.ModelLoader().load_from_file(path)
        model = model.to(self.device)
        if self.dtype == torch.float16 and self.device.type == "cuda":
            model = model.half()
        self._model = model
        return model

    def _get_face_helper(self):
        if self._face_helper is None:
            self._face_helper = create_face_helper(self.device)
        return self._face_helper

    @torch.inference_mode()
    def restore(
        self, img: Image.Image, visibility: float = 1.0, weight: float = 0.5
    ) -> Image.Image:
        """对图片执行 CodeFormer 人脸修复。

        Args:
            img: PIL 图片 (RGB)
            visibility: 与原图的混合比例 (0~1)
            weight: CodeFormer 权重 (0~1)，0=最大效果

        Returns:
            修复后的 PIL 图片 (RGB)
        """
        model = self._load()
        face_helper = self._get_face_helper()

        img_rgb = np.array(img.convert("RGB")).astype(np.float32)
        img_bgr = img_rgb[:, :, ::-1] / 255.0

        def restore_face_fn(cropped_face_rgb: np.ndarray):
            """修复单张裁剪后的人脸。"""
            # RGB [0,255] → BGR [0,1]
            face_bgr = cropped_face_rgb[:, :, ::-1].astype(np.float32) / 255.0
            face_tensor = torch.from_numpy(face_bgr).permute(2, 0, 1).unsqueeze(0).to(
                device=self.device, dtype=self.dtype
            )

            with autocast(self.device):
                output_tensor = model(face_tensor, w=weight)

            # back to numpy
            output = output_tensor.squeeze(0).cpu().float().clamp(0, 1)
            output = output.permute(1, 2, 0).numpy()
            # BGR [0,1] → RGB [0,255]
            output_rgb = (output[:, :, ::-1] * 255).astype(np.uint8)
            return output_rgb

        restored_bgr = restore_with_face_helper(
            img_bgr, face_helper, restore_face_fn, self.device
        )

        # BGR [0,1] → RGB PIL
        restored_rgb = (restored_bgr[:, :, ::-1] * 255).astype(np.uint8)
        restored_pil = Image.fromarray(restored_rgb, "RGB")

        if visibility < 1.0:
            original_rgb = img.convert("RGB")
            restored_pil = Image.blend(original_rgb, restored_pil, visibility)

        return restored_pil

    def to(self, device: torch.device):
        if self._model is not None:
            self._model = self._model.to(device)
        self.device = device
        return self

    def cpu(self):
        return self.to(torch.device("cpu"))
