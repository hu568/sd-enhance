"""Gradio 前端 UI。"""

import logging
from typing import Optional

import gradio as gr

from .config import Config
from .pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄

# Gradio 版本检测（兼容 3.x/4.x 和 6.x）
GRADIO_MAJOR = int(gr.__version__.split(".")[0])

APP_CSS = """
    .infotext p { margin: 0.5em 0; }
    footer { display: none !important; }
"""


def get_blocks_kwargs() -> dict:
    """返回传给 gr.Blocks 的参数，取决于 Gradio 版本。

    Gradio 6+: css/theme 传给 launch()，Blocks() 不接受这些参数。
    Gradio ≤5: css 传给 Blocks()，launch() 不支持 css。
    """
    kwargs = {"title": "SD Enhance - 图片后期处理"}
    if GRADIO_MAJOR < 6:
        kwargs["css"] = APP_CSS
    return kwargs


def get_launch_kwargs() -> dict:
    """返回传给 demo.launch 的参数，取决于 Gradio 版本。"""
    kwargs = {}
    if GRADIO_MAJOR >= 6:
        kwargs["css"] = APP_CSS
        kwargs["theme"] = "soft"
    return kwargs


def create_ui(config: Config) -> gr.Blocks:
    """创建 Gradio 界面。

    Args:
        config: 配置实例

    Returns:
        gr.Blocks 实例
    """
    pipeline = ProcessingPipeline(config)
    upscaler_names = pipeline.get_upscaler_names()
    face_models = pipeline.get_face_models()

    with gr.Blocks(**get_blocks_kwargs()) as demo:

        gr.Markdown(
            "# 🎨 SD Enhance\n"
            "独立的 Stable Diffusion 图片后期处理工具 — 放大 + 人脸修复"
        )

        # ---------- 状态组件（不可见） ----------
        tab_index = gr.State(value=0)
        dummy_component = gr.State(value=None)

        # ---------- 主布局 ----------
        with gr.Row(equal_height=False):
            # 左侧：输入 + 控制
            with gr.Column(scale=1, variant="panel"):
                # 输入模式标签页
                with gr.Tabs(elem_id="input_tabs") as input_tabs:
                    with gr.TabItem("单张图片", id=0) as tab_single:
                        input_image = gr.Image(
                            label="上传图片",
                            type="pil",
                            image_mode="RGBA",
                            height=400,
                        )

                    with gr.TabItem("批量处理", id=1) as tab_batch:
                        input_batch = gr.Files(
                            label="选择多张图片",
                            file_types=["image"],
                            interactive=True,
                        )

                    with gr.TabItem("目录批量", id=2) as tab_dir:
                        input_dir = gr.Textbox(
                            label="输入目录",
                            placeholder="图片所在目录路径...",
                        )
                        output_dir = gr.Textbox(
                            label="输出目录（留空使用默认）",
                            placeholder="留空则保存到默认输出目录",
                        )
                        show_results = gr.Checkbox(
                            label="显示结果图片", value=True
                        )

                # ---------- 放大设置 ----------
                with gr.Accordion("📐 图片放大", open=True):
                    upscale_enabled = gr.Checkbox(
                        label="启用放大", value=True
                    )
                    with gr.Row():
                        upscale_mode = gr.Radio(
                            choices=["按比例", "按尺寸"],
                            value="按比例",
                            label="模式",
                        )
                    with gr.Row():
                        upscale_scale = gr.Slider(
                            minimum=1.0,
                            maximum=8.0,
                            value=2.0,
                            step=0.5,
                            label="放大倍数",
                            visible=True,
                        )
                        upscale_width = gr.Number(
                            value=1024, label="宽度", visible=False
                        )
                        upscale_height = gr.Number(
                            value=1024, label="高度", visible=False
                        )
                    upscale_crop = gr.Checkbox(
                        label="裁剪到目标尺寸", value=False, visible=False
                    )
                    with gr.Row():
                        upscaler_1_name = gr.Dropdown(
                            choices=upscaler_names,
                            value=upscaler_names[1] if len(upscaler_names) > 1 else "None",
                            label="放大器 1",
                        )
                        upscaler_2_name = gr.Dropdown(
                            choices=["None"] + upscaler_names[1:],
                            value="None",
                            label="放大器 2（混合）",
                        )
                    upscaler_2_visibility = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        label="二级放大器混合度",
                    )

                # ---------- 人脸修复 ----------
                with gr.Accordion("👤 人脸修复", open=False):
                    face_notice = ""
                    if not face_models:
                        face_notice = (
                            "⚠️ 未找到人脸修复模型文件。\n\n"
                            "需要将以下文件放入对应目录：\n"
                            "- models/GFPGAN/GFPGANv1.4.pth\n"
                            "- models/Codeformer/codeformer-v0.1.0.pth\n"
                            "或通过 --webui-path 指向已有 WebUI 安装。"
                        )
                        gr.Markdown(face_notice)

                    face_enabled = gr.Checkbox(
                        label="启用面部修复",
                        value=False,
                        interactive=bool(face_models),
                    )
                    with gr.Row():
                        face_model = gr.Dropdown(
                            choices=face_models if face_models else ["（无可用模型）"],
                            value=face_models[0] if face_models else "（无可用模型）",
                            label="模型",
                            interactive=bool(face_models),
                        )
                        face_visibility = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            label="修复强度（与原图混合）",
                        )
                    codeformer_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="CodeFormer 权重（0=最大效果）",
                    )

                # ---------- 提交按钮 ----------
                submit_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")

            # 右侧：输出
            with gr.Column(scale=1, variant="panel"):
                output_gallery = gr.Gallery(
                    label="处理结果",
                    columns=3,
                    height=500,
                    object_fit="contain",
                    show_label=True,
                )
                output_info = gr.HTML(label="处理信息", elem_classes="infotext")
                output_status = gr.Markdown("就绪")

        # ---------- 事件绑定 ----------

        # 模式切换时更新 UI
        def on_tab_change(tab):
            return tab

        tab_single.select(fn=lambda: 0, outputs=[tab_index])
        tab_batch.select(fn=lambda: 1, outputs=[tab_index])
        tab_dir.select(fn=lambda: 2, outputs=[tab_index])

        # 放大模式切换
        def on_mode_change(mode):
            is_scale_by = mode == "按比例"
            return {
                upscale_scale: gr.update(visible=is_scale_by),
                upscale_width: gr.update(visible=not is_scale_by),
                upscale_height: gr.update(visible=not is_scale_by),
                upscale_crop: gr.update(visible=not is_scale_by),
            }

        upscale_mode.change(
            fn=on_mode_change,
            inputs=[upscale_mode],
            outputs=[upscale_scale, upscale_width, upscale_height, upscale_crop],
        )

        # 提交按钮
        def run_pipeline(
            tab_idx,
            s_image,
            b_files,
            i_dir,
            o_dir,
            s_results,
            u_enabled,
            u_mode,
            u_scale,
            u_w,
            u_h,
            u_crop,
            u1_name,
            u2_name,
            u2_vis,
            f_enabled,
            f_model,
            f_vis,
            cf_weight,
        ):
            mode = "scale_by" if u_mode == "按比例" else "scale_to"
            try:
                outputs, info = pipeline.process(
                    input_mode=tab_idx,
                    single_image=s_image,
                    batch_files=b_files,
                    input_dir=i_dir,
                    output_dir=o_dir,
                    show_results=s_results,
                    upscale_enabled=u_enabled,
                    upscale_mode=mode,
                    upscale_scale=u_scale,
                    upscale_width=int(u_w),
                    upscale_height=int(u_h),
                    upscale_crop=u_crop,
                    upscaler_1_name=u1_name,
                    upscaler_2_name=u2_name,
                    upscaler_2_visibility=u2_vis,
                    face_enabled=f_enabled,
                    face_model=f_model,
                    face_visibility=f_vis,
                    codeformer_weight=cf_weight,
                )
                status = f"✅ 处理完成，共 {len(outputs)} 张图片"
                return outputs, (
                    f"<div class='infotext'>{info}</div>" if info else ""
                ), status
            except Exception as e:
                logger.exception("处理失败")
                return [], f"<p style='color:red'>错误: {e}</p>", f"❌ 处理失败: {e}"

        submit_btn.click(
            fn=run_pipeline,
            inputs=[
                tab_index,
                input_image,
                input_batch,
                input_dir,
                output_dir,
                show_results,
                upscale_enabled,
                upscale_mode,
                upscale_scale,
                upscale_width,
                upscale_height,
                upscale_crop,
                upscaler_1_name,
                upscaler_2_name,
                upscaler_2_visibility,
                face_enabled,
                face_model,
                face_visibility,
                codeformer_weight,
            ],
            outputs=[output_gallery, output_info, output_status],
            show_progress="full",
        )

        # 在启动日志中打印可用模型
        logger.info("可用放大器: %d 个", len(upscaler_names))
        logger.info("可用面部修复模型: %s", face_models or "无")
        if config.webui_path:
            logger.info("已关联 WebUI 路径: %s", config.webui_path)

    return demo
