#!/usr/bin/env python
"""SD Enhance - 独立的 Stable Diffusion 图片后期处理工具。

从 AUTOMATIC1111/stable-diffusion-webui 中分离的后期处理模块。
支持图片放大（ESRGAN/RealESRGAN/Lanczos/Nearest）和人脸修复（GFPGAN/CodeFormer）。

用法:
    python main.py
    python main.py --webui-path "D:\\sd-webui-aki-..."
    python main.py --port 7860 --device cpu
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def setup_logging():
    """配置日志输出。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 屏蔽第三方库的 DEBUG 日志
    logging.getLogger("spandrel").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("facexlib").setLevel(logging.WARNING)


def load_env_config() -> dict:
    """读取 sd_enhance_config.json，设置 CUDA_PATH 环境变量（必须在 import torch 之前）。

    Returns:
        配置 dict（可能为空）
    """
    config_path = Path(__file__).resolve().parent / "sd_enhance_config.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    # 设置 CUDA_PATH 环境变量（让 torch 找到指定的 CUDA 工具包）
    cuda_path = cfg.get("cuda_path")
    if cuda_path and os.path.isdir(cuda_path):
        os.environ["CUDA_PATH"] = cuda_path
        # 同时加入 PATH，确保运行时能找到 cudart64_*.dll
        bin_path = os.path.join(cuda_path, "bin")
        if os.path.isdir(bin_path):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
        logger = logging.getLogger(__name__)
        logger.info("已设置 CUDA_PATH = %s", cuda_path)

    return cfg


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="SD Enhance - 独立的 Stable Diffusion 图片后期处理工具"
    )
    parser.add_argument(
        "--webui-path",
        type=str,
        default=None,
        help="指向已有的 stable-diffusion-webui 安装路径，复用其模型文件",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="模型文件存放目录（默认: ./models）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="运行设备（默认: 自动选择）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio 服务端口（默认: 7860）",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公开分享链接（通过 Gradio Share）",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="禁用 half-precision（FP16）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: ./output）",
    )
    return parser.parse_args()


def main():
    """程序入口。"""
    setup_logging()
    logger = logging.getLogger(__name__)

    # 读取配置文件（在 import torch 之前设置环境变量）
    env_cfg = load_env_config()

    args = parse_args()

    # 配置文件与 CLI 参数合并：CLI 参数优先
    # device
    device = args.device
    if device == "auto" and "device" in env_cfg:
        device = env_cfg["device"]

    # half
    no_half = args.no_half
    if not no_half and "half" in env_cfg:
        no_half = not env_cfg["half"]

    # port
    port = args.port
    if port == 7860 and "port" in env_cfg:
        port = env_cfg["port"]

    # models_dir
    models_dir = args.models_dir
    if not models_dir and "models_dir" in env_cfg:
        models_dir = env_cfg["models_dir"]

    # output_dir
    output_dir = args.output_dir
    if not output_dir and "output_dir" in env_cfg:
        output_dir = env_cfg["output_dir"]

    # 解析 WebUI 路径
    webui_path = args.webui_path
    if webui_path:
        webui_path = os.path.abspath(webui_path)
        if not os.path.isdir(webui_path):
            logger.error("WebUI 路径不存在: %s", webui_path)
            sys.exit(1)
        logger.info("使用 WebUI 模型路径: %s", webui_path)

    # 初始化配置
    from sd_enhance.config import Config

    config = Config(
        webui_path=webui_path,
        models_dir=models_dir,
        device_prefer=device,
        half=not no_half,
        output_dir=output_dir,
    )

    logger.info("设备: %s  |  精度: %s", config.device, config.dtype)
    logger.info("模型搜索路径: %d 个目录", len(config._models_search_paths))
    logger.info("输出目录: %s", config.output_dir)

    # 打印可用模型摘要
    upscaler_count = len(config.find_all_pth_files())
    face_models = config.list_available_face_models()
    logger.info("找到 %d 个放大器模型文件", upscaler_count)
    logger.info("面部修复模型: %s", face_models if face_models else "无")

    # 创建并启动 UI
    from sd_enhance.app import create_ui, get_launch_kwargs

    demo = create_ui(config)
    launch_kwargs = get_launch_kwargs()

    # 如果端口被占用，自动尝试下一个端口
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            logger.info("启动 Gradio 服务: http://127.0.0.1:%d", port)
            demo.launch(
                server_name="127.0.0.1",
                server_port=port,
                share=args.share,
                show_error=True,
                **launch_kwargs,
            )
            break  # 启动成功
        except OSError as e:
            if "Cannot find empty port" in str(e) and attempt < max_attempts - 1:
                logger.warning("端口 %d 被占用，尝试 %d", port, port + 1)
                port += 1
            else:
                raise


if __name__ == "__main__":
    main()
