"""图片读写、缩放、保存工具函数。"""

import os
import io
import json
from PIL import Image
from typing import Optional

# 支持的图片扩展名
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".avif"}


def read_image(path: str) -> Image.Image:
    """读取图片，自动修正 EXIF 方向。"""
    img = Image.open(path)
    # 修正 EXIF 方向
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img


def read_image_from_bytes(data: bytes) -> Image.Image:
    """从字节数据读取图片。"""
    img = Image.open(io.BytesIO(data))
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img


def fix_image(img: Image.Image) -> Image.Image:
    """修复图片：处理 RGBA 和 EXIF。"""
    if img.mode not in ("RGBA", "RGB"):
        img = img.convert("RGB")
    return img


def resize_image(
    img: Image.Image,
    mode: str = "scale_by",
    scale: float = 1.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    upscaler_name: str = "Lanczos",
) -> Image.Image:
    """缩放图片。

    Args:
        img: 输入图片
        mode: "scale_by" 按比例 | "scale_to" 按目标尺寸
        scale: 缩放倍数
        width: 目标宽度
        height: 目标高度
        upscaler_name: 使用的放大器（仅作为标记）
    """
    if mode == "scale_by":
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
    else:
        new_w = width or img.width
        new_h = height or img.height

    # 确保能被 8 整除（SD 模型的要求）
    new_w = new_w // 8 * 8
    new_h = new_h // 8 * 8

    if new_w < 8 or new_h < 8:
        raise ValueError(f"目标尺寸太小: {new_w}x{new_h}")

    return img.resize((new_w, new_h), Image.LANCZOS)


def save_image(
    img: Image.Image,
    output_dir: str,
    filename: str = "",
    suffix: str = "",
    metadata: Optional[dict] = None,
    extension: str = "png",
) -> str:
    """保存图片到磁盘。

    Args:
        img: 要保存的图片
        output_dir: 输出目录
        filename: 文件名（不含扩展名）
        suffix: 文件名后缀
        metadata: PNG 元数据字典，会被写入 PNG 文本块
        extension: 文件扩展名

    Returns:
        完整的保存路径
    """
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{suffix}" if suffix else timestamp
    else:
        filename = f"{filename}{suffix}"

    save_path = os.path.join(output_dir, f"{filename}.{extension}")

    if extension.lower() == "png" and metadata:
        from PIL.PngImagePlugin import PngInfo
        png_info = PngInfo()
        for k, v in metadata.items():
            if v is not None:
                png_info.add_text(k, str(v))
        img.save(save_path, "PNG", pnginfo=png_info)
    else:
        img.save(save_path)

    return save_path


def read_info_from_image(img: Image.Image) -> tuple[str, dict]:
    """从图片中读取生成参数。

    Returns:
        (参数字符串, 参数字典)
    """
    geninfo = ""
    items = {}

    if hasattr(img, "text") and img.text:
        # PNG 文本块
        parameters = img.text.get("parameters", "")
        if parameters:
            geninfo = parameters
            items["parameters"] = parameters

        # 后处理信息
        postprocessing = img.text.get("postprocessing", "")
        if postprocessing:
            items["postprocessing"] = postprocessing

    # 尝试从 EXIF 读取
    try:
        exif_data = img.info.get("exif", b"")
        if exif_data and not geninfo:
            # 某些工具将参数写入 EXIF UserComment
            import struct
            # 简单处理，仅做参考
            pass
    except Exception:
        pass

    return geninfo, items


def split_grid(img, tile_w, tile_h, overlap):
    """将图片分割为重叠的图块，用于分块放大。"""
    from collections import namedtuple
    Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap", "tile_count"])
    Row = namedtuple("Row", ["y", "h", "tiles"])
    Tile = namedtuple("Tile", ["x", "w", "tile"])

    w, h = img.size
    non_overlap_w = tile_w - overlap
    non_overlap_h = tile_h - overlap

    cols = max(1, (w - overlap + non_overlap_w - 1) // non_overlap_w)
    rows = max(1, (h - overlap + non_overlap_h - 1) // non_overlap_h)

    tiles = []
    for row_idx in range(rows):
        row_tiles = []
        y = row_idx * non_overlap_h
        if y + tile_h >= h:
            y = h - tile_h

        for col_idx in range(cols):
            x = col_idx * non_overlap_w
            if x + tile_w >= w:
                x = w - tile_w

            tile = img.crop((x, y, x + tile_w, y + tile_h))
            row_tiles.append(Tile(x, tile_w, tile))

        tiles.append(Row(y, tile_h, row_tiles))

    return Grid(tiles, tile_w, tile_h, w, h, overlap, rows * cols)


def combine_grid(grid):
    """将分块放大的图块合并回完整图片。"""
    img = Image.new("RGB", (grid.image_w, grid.image_h))

    for row in grid.tiles:
        for tile in row.tiles:
            img.paste(tile.tile, (tile.x, row.y))

    return img


def is_image_file(filename: str) -> bool:
    """检查文件名是否为支持的图片格式。"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS
