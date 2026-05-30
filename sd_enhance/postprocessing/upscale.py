"""图片放大后期处理脚本。"""

import logging
from typing import Optional

from PIL import Image

from ..config import Config
from ..models.upscaler import UpscalerManager, UpscalerLanczos, UpscalerNearest
from .base import PostprocessedImage, ScriptPostprocessingBase

logger = logging.getLogger(__name__)


class ScriptUpscale(ScriptPostprocessingBase):
    """图片放大操作。

    支持两种模式:
    - Scale by: 按比例放大
    - Scale to: 缩放到指定尺寸
    支持两级放大器混合。
    """

    name = "Upscale"
    order = 1000

    def __init__(self, config: Config, upscaler_manager: UpscalerManager):
        self.config = config
        self.upscaler_manager = upscaler_manager

    def process(  # type: ignore[override]
        self,
        pp: PostprocessedImage,
        *,
        enabled: bool = False,
        mode: str = "scale_by",  # "scale_by" | "scale_to"
        scale: float = 2.0,
        width: int = 1024,
        height: int = 1024,
        crop: bool = False,
        upscaler_1_name: str = "None",
        upscaler_2_name: str = "None",
        upscaler_2_visibility: float = 0.0,
    ):
        if not enabled:
            return

        upscaler_1 = self.upscaler_manager.get_by_name(upscaler_1_name)
        if upscaler_1 is None:
            logger.warning("放大器 %s 不可用，跳过", upscaler_1_name)
            return

        upscaler_2 = self.upscaler_manager.get_by_name(upscaler_2_name)

        # 计算目标尺寸
        if mode == "scale_by":
            target_w = pp.image.width * scale
            target_h = pp.image.height * scale
        else:
            target_w = float(width)
            target_h = float(height)
            # 计算实际 scale（用于放大器的参数）
            scale = max(target_w / pp.image.width, target_h / pp.image.height)

        # 确保能被 8 整除
        target_w = int(target_w // 8 * 8)
        target_h = int(target_h // 8 * 8)

        if target_w < 8 or target_h < 8:
            logger.warning("目标尺寸太小: %dx%d", target_w, target_h)
            return

        # 执行第一级放大
        logger.info("放大: %s × %s -> %dx%d", upscaler_1.name, scale, target_w, target_h)
        result = upscaler_1.upscale(pp.image, max(scale, 1.0))

        # 如果达到或超过目标尺寸，裁剪
        if crop and (result.width > target_w or result.height > target_h):
            left = (result.width - target_w) // 2
            top = (result.height - target_h) // 2
            result = result.crop((left, top, left + target_w, top + target_h))
        else:
            # 缩放到精确目标尺寸
            if result.width != target_w or result.height != target_h:
                result = result.resize((target_w, target_h), Image.LANCZOS)

        # 第二级放大混合
        if upscaler_2 and upscaler_2_visibility > 0 and upscaler_2.name != "None":
            logger.info("二级放大混合: %s (%.2f)", upscaler_2.name, upscaler_2_visibility)
            result_2 = upscaler_2.upscale(pp.image, max(scale, 1.0))
            if result_2.size != result.size:
                result_2 = result_2.resize(result.size, Image.LANCZOS)
            result = Image.blend(result, result_2, upscaler_2_visibility)

        # 更新图片和元数据
        pp.image = result
        pp.nametags.append(f"x{scale:.1f}".replace(".", "_"))
        pp.info["upscaler_1"] = upscaler_1.name
        pp.info["upscale_mode"] = mode
        pp.info["upscale_scale"] = scale
        if upscaler_2 and upscaler_2_visibility > 0:
            pp.info["upscaler_2"] = upscaler_2.name
            pp.info["upscaler_2_visibility"] = upscaler_2_visibility
