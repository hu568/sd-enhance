"""人脸检测与修复工具函数（基于 facexlib）。"""

import logging
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

from ..utils.device import autocast

logger = logging.getLogger(__name__)


def create_face_helper(device: torch.device):
    """初始化 facexlib 的人脸修复助手。

    Args:
        device: torch 设备

    Returns:
        FaceRestoreHelper 实例
    """
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    from facexlib.detection import retinaface

    # 保证 retinaface 使用正确的设备
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
    )
    return face_helper


def restore_with_face_helper(
    img: np.ndarray,
    face_helper,
    restore_face_fn: Callable[[np.ndarray], np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """对图片进行人脸检测和修复。

    Args:
        img: BGR numpy 数组 (H, W, 3), [0,1] 范围
        face_helper: FaceRestoreHelper 实例
        restore_face_fn: 对裁剪后的人脸进行修复的函数
        device: torch 设备

    Returns:
        修复后的 BGR numpy 数组 (H, W, 3), [0,1] 范围
    """
    face_helper.clean_all()

    # 检测人脸
    face_helper.read_image(img)
    num_detected = face_helper.get_face_landmarks_5(
        only_keep_largest=False,
        pose_threshold=None,
    )

    if num_detected == 0:
        logger.info("未检测到人脸，跳过修复")
        return img

    # 对齐和裁剪人脸
    face_helper.align_warp_face()

    # 对每张检测到的人脸进行修复
    for cropped_face in face_helper.cropped_faces:
        # cropped_face 是 numpy 数组 (H, W, 3) RGB [0,255]
        restored = restore_face_fn(cropped_face)

        # 如果 restore_face_fn 返回了处理结果
        if restored is not None:
            face_helper.add_restored_face(restored)

    # 将修复后的人脸拼回原图
    face_helper.get_inverse_affine(None)

    # 修复后的完整图片 (BGR, [0,1])
    restored_img = face_helper.paste_faces_to_input_image()
    return restored_img
