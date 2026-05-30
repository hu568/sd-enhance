"""核心处理管线：加载图片 → 执行脚本 → 保存输出。"""

import logging
import os
from typing import Optional

from PIL import Image

from .config import Config
from .utils.image import (
    read_image,
    fix_image,
    save_image,
    read_info_from_image,
    is_image_file,
)
from .utils.state import State
from .utils.device import torch_gc
from .models.upscaler import UpscalerManager
from .postprocessing.base import PostprocessedImage
from .postprocessing.upscale import ScriptUpscale
from .postprocessing.face_restore import ScriptFaceRestore

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """核心处理管线。"""

    def __init__(self, config: Config):
        self.config = config
        self.state = State()

        # 模型管理器
        self.upscaler_manager = UpscalerManager(config)

        # 后期处理脚本
        self.scripts = [
            ScriptUpscale(config, self.upscaler_manager),
            ScriptFaceRestore(config),
        ]

    def process(
        self,
        input_mode: int,  # 0=单张, 1=批量文件, 2=目录批量
        single_image: Optional[Image.Image],
        batch_files: Optional[list],
        input_dir: str,
        output_dir: str,
        show_results: bool,
        save_output: bool = True,
        # 放大参数
        upscale_enabled: bool = False,
        upscale_mode: str = "scale_by",
        upscale_scale: float = 2.0,
        upscale_width: int = 1024,
        upscale_height: int = 1024,
        upscale_crop: bool = False,
        upscaler_1_name: str = "None",
        upscaler_2_name: str = "None",
        upscaler_2_visibility: float = 0.0,
        # 人脸修复参数
        face_enabled: bool = False,
        face_model: str = "GFPGAN",
        face_visibility: float = 1.0,
        codeformer_weight: float = 0.5,
    ) -> tuple[list[Image.Image], str]:
        """执行后期处理管线。

        Returns:
            (输出图片列表, 信息文本)
        """
        self.state.begin(job="postprocessing")
        outputs = []
        infotext_lines = []

        try:
            # 1. 加载图片
            images_to_process = self._load_images(
                input_mode, single_image, batch_files, input_dir
            )

            if not images_to_process:
                return [], "没有找到可处理的图片"

            self.state.job_count = len(images_to_process)

            # 2. 确定输出目录
            if output_dir:
                outpath = output_dir
            else:
                outpath = self.config.output_dir
            os.makedirs(outpath, exist_ok=True)

            # 3. 逐张处理
            for img, name in images_to_process:
                if self.state.interrupted:
                    break

                self.state.nextjob()
                self.state.textinfo = name or f"图片 {self.state.job_no}"

                logger.info(
                    "处理 [%d/%d]: %s", self.state.job_no, self.state.job_count, name
                )

                try:
                    pp = PostprocessedImage(fix_image(img))

                    # 执行所有后期处理脚本
                    for script in self.scripts:
                        if self.state.interrupted:
                            break
                        if script.name == "Upscale":
                            script.process(
                                pp,
                                enabled=upscale_enabled,
                                mode=upscale_mode,
                                scale=upscale_scale,
                                width=upscale_width,
                                height=upscale_height,
                                crop=upscale_crop,
                                upscaler_1_name=upscaler_1_name,
                                upscaler_2_name=upscaler_2_name,
                                upscaler_2_visibility=upscaler_2_visibility,
                            )
                        elif script.name == "Face Restoration":
                            script.process(
                                pp,
                                enabled=face_enabled,
                                model_name=face_model,
                                visibility=face_visibility,
                                codeformer_weight=codeformer_weight,
                            )

                    # 收集信息文本
                    info_str = ", ".join(
                        f"{k}: {v}"
                        for k, v in pp.info.items()
                        if v is not None
                    )
                    if info_str:
                        infotext_lines.append(info_str)

                    # 保存输出
                    if save_output:
                        basename = os.path.splitext(os.path.basename(name))[0] if name else ""
                        suffix = "_".join(pp.nametags) if pp.nametags else ""
                        save_image(
                            pp.image,
                            outpath,
                            filename=basename,
                            suffix=f"_{suffix}" if suffix else "",
                            metadata=pp.info,
                        )
                        logger.info("已保存: %s", outpath)

                    # 收集结果
                    if input_mode != 2 or show_results:
                        outputs.append(pp.image)

                except Exception as e:
                    logger.error("处理图片 %s 失败: %s", name or "unknown", e)
                    continue

        finally:
            self.state.end()
            torch_gc()

        info_text = "<br>\n".join(infotext_lines) if infotext_lines else ""
        return outputs, info_text

    def _load_images(
        self,
        input_mode: int,
        single_image: Optional[Image.Image],
        batch_files: Optional[list],
        input_dir: str,
    ) -> list[tuple[Image.Image, Optional[str]]]:
        """根据输入模式加载图片列表。

        Returns:
            [(PIL Image, 文件名或无), ...]
        """
        images = []

        if input_mode == 0:
            # 单张图片
            if single_image is None:
                raise ValueError("未选择图片")
            images.append((single_image, None))

        elif input_mode == 1:
            # 批量文件
            if not batch_files:
                raise ValueError("未选择批量文件")
            for f in batch_files:
                try:
                    if hasattr(f, "name"):
                        img = read_image(f.name)
                        name = os.path.splitext(os.path.basename(f.orig_name))[0]
                        images.append((img, name))
                    else:
                        img = read_image(f)
                        name = os.path.splitext(os.path.basename(f))[0]
                        images.append((img, name))
                except Exception as e:
                    logger.warning("读取文件失败: %s", e)
                    continue

        elif input_mode == 2:
            # 目录批量
            if not input_dir or not os.path.isdir(input_dir):
                raise ValueError(f"输入目录无效: {input_dir}")
            for filename in sorted(os.listdir(input_dir)):
                if is_image_file(filename):
                    try:
                        img = read_image(os.path.join(input_dir, filename))
                        name = os.path.splitext(filename)[0]
                        images.append((img, name))
                    except Exception as e:
                        logger.warning("读取文件 %s 失败: %s", filename, e)
                        continue

        return images

    def get_upscaler_names(self) -> list[str]:
        return self.upscaler_manager.get_names()

    def get_face_models(self) -> list[str]:
        """返回可用的面部修复模型。"""
        face_script = self.scripts[1]
        return face_script.get_available_models()
