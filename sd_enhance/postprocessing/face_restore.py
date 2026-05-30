"""人脸修复后期处理脚本（GFPGAN / CodeFormer）。"""

import logging
from typing import Optional

from ..config import Config
from ..models.gfpgan import GFPGANModel
from ..models.codeformer import CodeFormerModel
from .base import PostprocessedImage, ScriptPostprocessingBase

logger = logging.getLogger(__name__)


class ScriptFaceRestore(ScriptPostprocessingBase):
    """人脸修复操作。

    支持 GFPGAN 和 CodeFormer 两种模型。
    """

    name = "Face Restoration"
    order = 2000

    def __init__(self, config: Config):
        self.config = config
        self.gfpgan = GFPGANModel(config)
        self.codeformer = CodeFormerModel(config)

    def get_available_models(self) -> list[str]:
        """返回当前可用的面部修复模型列表。"""
        models = []
        if self.gfpgan.is_available():
            models.append("GFPGAN")
        if self.codeformer.is_available():
            models.append("CodeFormer")
        return models

    def process(  # type: ignore[override]
        self,
        pp: PostprocessedImage,
        *,
        enabled: bool = False,
        model_name: str = "GFPGAN",
        visibility: float = 1.0,
        codeformer_weight: float = 0.5,
    ):
        if not enabled:
            return

        if model_name == "GFPGAN":
            if not self.gfpgan.is_available():
                logger.warning("GFPGAN 模型不可用")
                return
            logger.info("执行 GFPGAN 人脸修复")
            pp.image = self.gfpgan.restore(pp.image, visibility=visibility)
            pp.info["face_restoration"] = "GFPGAN"
            pp.info["face_restoration_visibility"] = visibility
            pp.nametags.append("gfpgan")

        elif model_name == "CodeFormer":
            if not self.codeformer.is_available():
                logger.warning("CodeFormer 模型不可用")
                return
            logger.info("执行 CodeFormer 人脸修复 (weight=%.2f)", codeformer_weight)
            pp.image = self.codeformer.restore(
                pp.image, visibility=visibility, weight=codeformer_weight
            )
            pp.info["face_restoration"] = "CodeFormer"
            pp.info["face_restoration_visibility"] = visibility
            pp.info["codeformer_weight"] = codeformer_weight
            pp.nametags.append("codeformer")
