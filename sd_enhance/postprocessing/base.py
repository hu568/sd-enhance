"""后期处理基类：PostprocessedImage + ScriptPostprocessingBase。"""

from __future__ import annotations

from typing import Any, Optional

from PIL import Image


class PostprocessedImage:
    """包装一张图片及其处理信息。

    Attributes:
        image: 当前的 PIL 图片
        info: 元数据字典（处理参数等）
        extra_images: 额外输出的图片列表
        caption: 可选的图片描述文本
        nametags: 用于生成文件后缀的标签
        disable_processing: 是否跳过后续处理
    """

    def __init__(self, image: Image.Image):
        self.image = image
        self.info: dict[str, Any] = {}
        self.extra_images: list[PostprocessedImage] = []
        self.caption: Optional[str] = None
        self.nametags: list[str] = []
        self.disable_processing: bool = False

    def create_copy(
        self, image: Image.Image, *, extra_nametags: Optional[list[str]] = None
    ) -> PostprocessedImage:
        """创建当前实例的副本，用于持有额外输出的图片。"""
        pp = PostprocessedImage(image)
        pp.info = dict(self.info)
        pp.nametags = list(self.nametags)
        if extra_nametags:
            pp.nametags.extend(extra_nametags)
        return pp


class ScriptPostprocessingBase:
    """后期处理脚本的基类。

    子类需要实现:
    - name: 脚本名称
    - order: 执行顺序（数值越小越先执行）
    """

    name: str = ""
    order: int = 1000

    def process(self, pp: PostprocessedImage, **kwargs):
        """对图片执行处理。

        Args:
            pp: 待处理的图片包装器
            **kwargs: UI 控件传来的参数
        """
        raise NotImplementedError
