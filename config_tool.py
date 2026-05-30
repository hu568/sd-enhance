#!/usr/bin/env python
"""SD Enhance 配置工具 — 管理 CUDA、PyTorch、设备、模型路径等设置。

用法:
    python config_tool.py
    python config_tool.py --port 7861
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)

# Gradio 版本检测（兼容 3.x~6.x）
GRADIO_MAJOR = int(gr.__version__.split(".")[0])

# ---------- 路径常量 ----------
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "sd_enhance_config.json"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------- 配置读写 ----------


def load_config() -> dict:
    """读取 sd_enhance_config.json，不存在则返回空 dict。"""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("配置文件读取失败: %s", e)
    return {}


def save_config(data: dict) -> str:
    """保存配置到 sd_enhance_config.json，返回状态信息。"""
    try:
        # 保留现有配置中未修改的字段
        existing = load_config()
        existing.update(data)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        return f"✅ 配置已保存到 {CONFIG_PATH}"
    except Exception as e:
        return f"❌ 保存失败: {e}"


# ---------- 系统检测函数 ----------


def get_nvidia_smi_output() -> str:
    """运行 nvidia-smi 获取 GPU 信息，失败返回空字符串。"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def parse_cuda_version(smi_output: str) -> str:
    """从 nvidia-smi 输出中提取 CUDA 版本。"""
    for line in smi_output.splitlines():
        if "CUDA Version" in line:
            # "NVIDIA-SMI 595.79  Driver Version: 595.79  CUDA Version: 13.2"
            parts = line.split("CUDA Version:")
            if len(parts) > 1:
                return parts[1].strip()
    return ""


def parse_gpu_info(smi_output: str) -> str:
    """从 nvidia-smi 输出中提取 GPU 名称和显存。"""
    # 从 GPU Summary 区域找 GPU 名称
    for line in smi_output.splitlines():
        if "NVIDIA GeForce" in line or "NVIDIA" in line and "GB" in line:
            return line.strip()
    return ""


def get_gpu_memory() -> str:
    """通过 nvidia-smi 查询显存信息。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 3:
                    name, mem_total, mem_used = parts[0], parts[1], parts[2]
                    return f"{name} | 显存: {mem_total}MB 总计 / {mem_used}MB 使用中"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "未检测到 NVIDIA GPU"


def check_torch_in_process() -> dict:
    """在当前 Python 进程中检测 PyTorch 状态。"""
    info = {
        "torch_version": "未安装",
        "cuda_available": False,
        "cuda_version": "",
        "gpu_name": "",
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda or ""
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def check_python_path(path: str) -> str:
    """检测指定 Python 可执行文件中的 PyTorch 和 CUDA 状态。

    Args:
        path: Python 可执行文件路径

    Returns:
        状态描述字符串
    """
    if not path or not path.strip():
        return "未指定"

    path = path.strip()

    # 如果是目录，自动补全 python.exe
    p = Path(path)
    if p.is_dir():
        candidate = p / "Scripts" / "python.exe"
        if candidate.exists():
            path = str(candidate)
        else:
            return "❌ 目录中未找到 python.exe"
    elif not p.exists():
        return "❌ 路径不存在"

    # 检查 Python 版本
    try:
        ver_result = subprocess.run(
            [path, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if ver_result.returncode != 0:
            return f"❌ 无法执行: {ver_result.stderr.strip()[:100]}"
        python_version = ver_result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"❌ 无法执行: {e}"

    # 检查 PyTorch 和 CUDA
    try:
        result = subprocess.run(
            [
                path, "-c",
                "import torch; "
                "print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.version.cuda or \"\"}|{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\"}')"
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split("|")
            if len(parts) >= 2:
                ver, cuda = parts[0], parts[1]
                cuda_ver = parts[2] if len(parts) > 2 else ""
                gpu = parts[3] if len(parts) > 3 else ""

                status = f"{python_version}"
                status += f"\nPyTorch: {ver}"
                if cuda == "True":
                    status += f"\n✅ CUDA 可用 (版本 {cuda_ver})"
                    if gpu:
                        status += f"\nGPU: {gpu}"
                    return status
                else:
                    status += "\n⚠️  仅 CPU 模式（CUDA 不可用）"
                    return status
            return f"{python_version}\nPyTorch 检测异常: {result.stdout.strip()[:200]}"
        else:
            # 可能是 PyTorch 没安装
            stderr = result.stderr.strip()[:200]
            if "No module named" in stderr or "ModuleNotFoundError" in stderr:
                return f"{python_version}\n⚠️  未安装 PyTorch"
            return f"{python_version}\n⚠️  检测错误: {stderr}"
    except subprocess.TimeoutExpired:
        return f"{python_version}\n⚠️  检测超时"
    except OSError as e:
        return f"❌ 执行失败: {e}"


def check_cuda_path(path: str) -> str:
    """检测指定 CUDA 目录是否有效。"""
    if not path or not path.strip():
        return "未指定"

    p = Path(path.strip())
    if not p.exists():
        return "❌ 路径不存在"
    if not p.is_dir():
        return "❌ 不是目录"

    # 检查关键文件
    findings = []

    # nvcc
    nvcc = p / "bin" / "nvcc.exe"
    if nvcc.exists():
        try:
            result = subprocess.run(
                [str(nvcc), "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "release" in line:
                        findings.append(f"nvcc: {line.strip()}")
                        break
        except (OSError, subprocess.TimeoutExpired):
            findings.append("nvcc: 存在但无法执行")
    else:
        findings.append("nvcc: 未找到")

    # cudart
    cudart = p / "bin" / "cudart64_*.dll"
    cudart_files = list(p.glob("bin/cudart64_*.dll"))
    if cudart_files:
        findings.append(f"CUDA Runtime: 找到 {len(cudart_files)} 个")
    else:
        # 也检查 lib 目录
        cudart_files = list(p.glob("lib/**/cudart*.lib"))
        if cudart_files:
            findings.append(f"CUDA Runtime lib: 找到")
        else:
            findings.append("CUDA Runtime: 未找到")

    # version.txt
    version_file = p / "version.txt"
    if version_file.exists():
        findings.append(f"版本文件存在")

    if not findings:
        return "⚠️  目录内容不完整（未找到 CUDA 组件）"

    return "\n".join(findings)


def scan_model_files(models_dir: str = None) -> list[dict]:
    """扫描模型文件。"""
    if not models_dir:
        models_dir = str(DEFAULT_MODELS_DIR)

    scan_path = Path(models_dir)
    if not scan_path.exists():
        return []

    models = []
    for f in sorted(scan_path.rglob("*.pth")):
        try:
            size_mb = f.stat().st_size / (1024 * 1024)
            models.append({
                "path": str(f.relative_to(scan_path)),
                "size": f"{size_mb:.1f} MB",
            })
        except OSError:
            pass
    return models


# ---------- Gradio UI 回调函数 ----------


def on_load():
    """页面加载时获取所有标签页的信息（一次性返回，避免多个 demo.load 冲突）。"""
    config = load_config()

    # ---- Tab 1: 系统诊断 ----
    smi = get_nvidia_smi_output()
    cuda_ver = parse_cuda_version(smi)
    gpu_mem = get_gpu_memory()
    torch_info = check_torch_in_process()

    # 如果配置了 python_path，也检测目标环境的 CUDA 状态
    python_path = config.get("python_path", "")
    target_cuda_line = ""
    if python_path:
        py_check = check_python_path(python_path)
        has_target_cuda = "✅ CUDA 可用" in py_check
        target_cuda_line = (
            f"\n| **目标环境 CUDA** | {'✅ 是' if has_target_cuda else '❌ 否'} |"
        )

    diag_text = f"""## 系统诊断

| 项目 | 状态 |
|------|------|
| **GPU** | {gpu_mem} |
| **CUDA 驱动版本** | {cuda_ver or '未检测到'} |
| **PyTorch 版本** | {torch_info['torch_version']} |
| **当前环境 CUDA** | {'✅ 是 - ' + torch_info['gpu_name'] if torch_info['cuda_available'] else '❌ 否（CPU 版 PyTorch）'} |
| **CUDA_PATH** | {os.environ.get('CUDA_PATH', '(未设置)')}{target_cuda_line} |
"""

    # ---- Tab 2: 配置表单值 ----
    device = config.get("device", "auto")
    half = config.get("half", True)
    cuda_path = config.get("cuda_path", "")
    python_path = config.get("python_path", "")
    models_dir = config.get("models_dir", "")
    output_dir = config.get("output_dir", "")
    port = config.get("port", 7860)

    # ---- Tab 3: PyTorch 状态 ----
    torch_lines = [
        f"**PyTorch 版本**: {torch_info['torch_version']}",
        f"**CUDA 可用**: {'✅ 是' if torch_info['cuda_available'] else '❌ 否（CPU 版 PyTorch）'}",
    ]
    if torch_info["cuda_version"]:
        torch_lines.append(f"**CUDA 版本**: {torch_info['cuda_version']}")
    if torch_info["gpu_name"]:
        torch_lines.append(f"**GPU**: {torch_info['gpu_name']}")

    # 如果配置了 python_path，追加目标环境信息
    if python_path:
        py_check = check_python_path(python_path)
        has_target_cuda = "✅ CUDA 可用" in py_check
        torch_lines.append("")
        torch_lines.append("---")
        torch_lines.append("**配置的目标 Python 环境:**")
        torch_lines.append(f"**路径**: `{python_path}`")
        torch_lines.append(f"**CUDA 可用**: {'✅ 是' if has_target_cuda else '❌ 否'}")
        if has_target_cuda:
            # 从检测结果中提取 PyTorch 版本
            for line in py_check.split("\n"):
                if "PyTorch:" in line:
                    torch_lines.append(f"**版本**: {line.strip()}")
                elif "GPU:" in line:
                    torch_lines.append(f"**GPU**: {line.strip()}")

    torch_status_text = "\n".join(torch_lines)

    # ---- Tab 4: 模型列表 ----
    scan_dir = models_dir if models_dir else None
    model_list_text = on_scan_models(scan_dir)

    return (
        diag_text,             # Tab 1
        device, half,          # Tab 2 表单
        cuda_path, python_path,
        models_dir, output_dir, port,
        torch_status_text,     # Tab 3
        model_list_text,       # Tab 4
    )


def on_refresh():
    """刷新系统诊断信息。"""
    return on_load()[0]  # 只返回诊断文本


def on_save_config(device, half, cuda_path, python_path, models_dir, output_dir, port):
    """保存所有配置。"""
    data = {
        "device": device,
        "half": half,
        "cuda_path": cuda_path.strip() if cuda_path else None,
        "python_path": python_path.strip() if python_path else None,
        "models_dir": models_dir.strip() if models_dir else None,
        "output_dir": output_dir.strip() if output_dir else None,
        "port": int(port),
    }
    return save_config(data)


def on_check_python(python_path):
    """检测指定 Python 路径。"""
    return check_python_path(python_path)


def on_check_cuda(cuda_path):
    """检测指定 CUDA 目录。"""
    return check_cuda_path(cuda_path)


def on_check_torch_now():
    """在当前进程重新检测 PyTorch。"""
    info = check_torch_in_process()
    lines = [
        f"**PyTorch 版本**: {info['torch_version']}",
        f"**CUDA 可用**: {'✅ 是' if info['cuda_available'] else '❌ 否'}",
    ]
    if info["cuda_version"]:
        lines.append(f"**CUDA 版本**: {info['cuda_version']}")
    if info["gpu_name"]:
        lines.append(f"**GPU**: {info['gpu_name']}")
    return "\n".join(lines)


def on_test_cuda():
    """运行 CUDA 小测试。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return "❌ CUDA 不可用，无法测试。"

        device = torch.device("cuda")
        # 创建一个小张量做简单运算
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.mm(a, b)
        result = torch.sum(c).item()

        gpu_name = torch.cuda.get_device_name(0)
        mem_allocated = torch.cuda.memory_allocated(0) / 1024 / 1024

        return (
            f"✅ CUDA 测试通过！\n\n"
            f"**GPU**: {gpu_name}\n"
            f"**矩阵乘法结果**: {result:.2f}\n"
            f"**显存占用**: {mem_allocated:.1f} MB\n"
            f"**运算设备**: {c.device}"
        )
    except ImportError:
        return "❌ PyTorch 未安装"
    except Exception as e:
        return f"❌ 测试失败: {e}"


def on_scan_models(models_dir):
    """扫描模型文件。"""
    models = scan_model_files(models_dir if models_dir else None)

    if not models:
        return "未找到模型文件"

    # 检测面部模型
    gfpgan_path = Path(models_dir if models_dir else DEFAULT_MODELS_DIR) / "GFPGAN" / "GFPGANv1.4.pth"
    codeformer_path = Path(models_dir if models_dir else DEFAULT_MODELS_DIR) / "Codeformer" / "codeformer-v0.1.0.pth"

    face_status = []
    if gfpgan_path.exists():
        face_status.append("✅ GFPGAN: 已安装")
    else:
        face_status.append("❌ GFPGAN: 未找到")
    if codeformer_path.exists():
        face_status.append("✅ CodeFormer: 已安装")
    else:
        face_status.append("❌ CodeFormer: 未找到")

    lines = [
        f"共找到 **{len(models)}** 个模型文件（{sum(float(m['size'].replace(' MB','')) for m in models):.0f} MB）\n",
        "### 面部修复模型",
        *face_status,
        "",
        "### 放大器模型",
    ]

    upscaler_categories = {}
    for m in models:
        path = m["path"]
        size = m["size"]
        category = Path(path).parent.name if "\\" in path or "/" in path else "根目录"
        if category not in upscaler_categories:
            upscaler_categories[category] = []
        upscaler_categories[category].append((Path(path).name, size))

    for cat, files in sorted(upscaler_categories.items()):
        lines.append(f"\n**{cat}/**")
        for name, size in files:
            lines.append(f"  - {name} ({size})")

    return "\n".join(lines)


# ---------- 构建 UI ----------

APP_CSS = """
    footer { display: none !important; }
    table { width: 100%; border-collapse: collapse; }
    table td, table th { padding: 6px 12px; border: 1px solid #ddd; }
    .status-ok { color: #22c55e; font-weight: bold; }
    .status-warn { color: #f59e0b; font-weight: bold; }
    .status-err { color: #ef4444; font-weight: bold; }
"""


def create_ui():
    """创建 Gradio 配置界面。"""
    # Gradio ≤5: css 传给 Blocks()；Gradio 6+: css/theme 传给 launch()
    blocks_kw = {"title": "SD Enhance - 配置工具"}
    if GRADIO_MAJOR < 6:
        blocks_kw["css"] = APP_CSS

    with gr.Blocks(**blocks_kw) as demo:
        gr.Markdown(
            "# ⚙️ SD Enhance 配置工具\n"
            "管理 CUDA、PyTorch、设备和模型路径设置。"
        )

        # ---------- 标签页 1: 系统诊断 ----------
        with gr.TabItem("🖥️  系统诊断"):
            diag_output = gr.Markdown("正在检测...")
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新检测", variant="secondary", scale=1)

        # ---------- 标签页 2: 设备 & 目录配置 ----------
        with gr.TabItem("⚙️  设备 & 目录配置"):
            gr.Markdown("### 设备设置")
            device_dropdown = gr.Dropdown(
                choices=["auto", "cuda", "cpu"],
                value="auto",
                label="计算设备",
                info="auto = 自动优先 CUDA",
            )
            half_checkbox = gr.Checkbox(
                value=True,
                label="启用 FP16 半精度",
                info="仅在 CUDA 模式下生效，可降低显存占用",
            )

            gr.Markdown("---\n### 目录配置")
            gr.Markdown(
                "可手动指定 CUDA 安装路径和 Python 环境路径。\n"
                "留空则使用自动检测或默认值。"
            )

            cuda_path_input = gr.Textbox(
                label="CUDA 安装目录",
                placeholder="例如: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2",
                info="设置后 main.py 会在启动时配置 CUDA_PATH 环境变量",
            )
            with gr.Row():
                check_cuda_btn = gr.Button("🔍 检测 CUDA 目录", scale=1)
            cuda_path_status = gr.Markdown("")

            python_path_input = gr.Textbox(
                label="Python 环境路径",
                placeholder="例如: D:\\other_env\\Scripts\\python.exe",
                info="指定后 start.bat 优先使用此 Python 启动（可指向已有 CUDA 版 PyTorch 的环境）",
            )
            with gr.Row():
                check_python_btn = gr.Button("🔍 检测 Python 环境", scale=1)
            python_path_status = gr.Markdown("")

            gr.Markdown("---\n### 其他设置")
            models_dir_input = gr.Textbox(
                label="模型文件目录（可选）",
                placeholder="默认: ./models",
            )
            output_dir_input = gr.Textbox(
                label="输出目录（可选）",
                placeholder="默认: ./output",
            )
            port_input = gr.Number(
                value=7860,
                label="服务端口",
                minimum=1024,
                maximum=65535,
            )

            gr.Markdown("---")
            with gr.Row():
                save_config_btn = gr.Button("💾 保存配置", variant="primary", scale=2)
            save_status = gr.Markdown("")

        # ---------- 标签页 3: PyTorch 管理 ----------
        with gr.TabItem("📦  PyTorch 管理"):
            gr.Markdown("### 当前环境")
            torch_status = gr.Markdown("正在检测...")

            with gr.Row():
                refresh_torch_btn = gr.Button("🔄 重新检测", variant="secondary", scale=1)
                test_cuda_btn = gr.Button("🧪 测试 CUDA 运算", variant="secondary", scale=1)

            gr.Markdown("---\n### 安装 CUDA 版 PyTorch")
            gr.Markdown(
                "如果当前 PyTorch 仅支持 CPU，可以在终端中运行以下命令来安装 CUDA 版：\n\n"
                "```bash\n"
                "# 1. 卸载 CPU 版 PyTorch\n"
                "venv\\Scripts\\pip uninstall -y torch torchvision\n\n"
                "# 2. 安装 CUDA 版（推荐 cu126，向下兼容 CUDA 12.x）\n"
                "venv\\Scripts\\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126\n"
                "```\n\n"
                "> 你的 GPU（RTX 5070）驱动版本 595.79 支持 CUDA 13.2，\n"
                "> cu126 版 PyTorch 可兼容运行（PyTorch 运行时仅需 CUDA driver ≥ cu126）。"
            )

            gr.Markdown("---\n### 使用外部 Python 环境的 PyTorch")
            gr.Markdown(
                "如果其他 Python 环境已安装 CUDA 版 PyTorch（如 WebUI 的 venv），\n"
                "可以在「设备 & 目录配置」标签页中指定该环境的 `python.exe` 路径，\n"
                "`start.bat` 将优先使用该环境启动。"
            )

        # ---------- 标签页 4: 模型状态 ----------
        with gr.TabItem("📋  模型状态"):
            model_output = gr.Markdown("正在扫描模型...")
            with gr.Row():
                refresh_models_btn = gr.Button("🔄 刷新模型列表", variant="secondary", scale=1)

        # ========== 事件绑定 ==========

        # 页面加载 — 一次性初始化所有标签页（单个 demo.load 避免冲突）
        demo.load(
            fn=on_load,
            outputs=[
                diag_output,                    # Tab 1
                device_dropdown, half_checkbox,  # Tab 2
                cuda_path_input, python_path_input,
                models_dir_input, output_dir_input, port_input,
                torch_status,                   # Tab 3
                model_output,                   # Tab 4
            ],
        )

        # Tab 1: 刷新
        refresh_btn.click(fn=on_refresh, outputs=diag_output)

        # Tab 2: 检测 CUDA 目录
        check_cuda_btn.click(
            fn=on_check_cuda,
            inputs=cuda_path_input,
            outputs=cuda_path_status,
        )

        # Tab 2: 检测 Python 环境
        check_python_btn.click(
            fn=on_check_python,
            inputs=python_path_input,
            outputs=python_path_status,
        )

        # Tab 2: 保存配置
        save_config_btn.click(
            fn=on_save_config,
            inputs=[
                device_dropdown, half_checkbox,
                cuda_path_input, python_path_input,
                models_dir_input, output_dir_input, port_input,
            ],
            outputs=save_status,
        )

        # Tab 3: 刷新 PyTorch 状态
        refresh_torch_btn.click(fn=on_check_torch_now, outputs=torch_status)

        # Tab 3: 测试 CUDA
        test_cuda_btn.click(fn=on_test_cuda, outputs=torch_status)

        # Tab 4: 刷新模型列表
        refresh_models_btn.click(
            fn=on_scan_models,
            inputs=models_dir_input,
            outputs=model_output,
        )

    return demo


def main():
    """配置工具入口。"""
    parser = argparse.ArgumentParser(description="SD Enhance 配置工具")
    parser.add_argument(
        "--port", type=int, default=7861,
        help="配置工具端口（默认 7861，避免与主程序冲突）",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="创建公开分享链接",
    )
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 确保必要目录存在
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    demo = create_ui()
    logger.info("配置工具启动: http://127.0.0.1:%d", args.port)

    # launch 参数（Gradio 6+ 才支持 css/theme）
    launch_kw = {}
    if GRADIO_MAJOR >= 6:
        launch_kw["css"] = APP_CSS
        launch_kw["theme"] = "soft"

    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        show_error=True,
        **launch_kw,
    )


if __name__ == "__main__":
    main()
