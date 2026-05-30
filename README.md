# SD Enhance

独立的 Stable Diffusion 图片后期处理工具，内置常用放大器模型。

从 [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 中分离的后期处理模块，使用 Gradio 实现前端界面。

## 功能

- **图片放大**：内置 ESRGAN / RealESRGAN / SwinIR / Lanczos / Nearest 等多种放大器，开箱即用
- **人脸修复**：支持 GFPGAN / CodeFormer（需手动下载模型文件）
- 单张处理、批量文件、目录批量三种输入模式
- 两级放大器混合

## 系统要求

- Python 3.10+
- Windows / Linux / macOS
- NVIDIA GPU（推荐）或 CPU

## 获取

### 方式一：下载 Release 压缩包（推荐）

从 [Releases](https://github.com/hu568/sd-enhance/releases) 页面下载最新版 `sd-enhance-v*.zip`，解压即可使用。

压缩包包含全部源码和模型文件（约 410MB），无需额外下载。

### 方式二：克隆仓库 + 自行准备模型

```bash
git clone https://github.com/你的用户名/sd-enhance.git
cd sd-enhance
```

模型文件可从已有 stable-diffusion-webui 目录复制。

## 安装

```bash
# 1. 创建虚拟环境
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 2. （NVIDIA GPU 用户）安装 CUDA 版 PyTorch
#    如果只需 CPU 版，跳过此步直接执行第 3 步
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. 安装其余依赖
pip install -r requirements.txt
```

> **已有 PyTorch 环境？** 如果你有其他 Python 环境（如 stable-diffusion-webui 的 venv）已经装了 CUDA 版 PyTorch，可以跳过第 2 步和第 3 步的 torch 安装。
> 之后打开配置工具（`config_tool.bat`），在"设备 & 目录配置"标签页中指定该环境的 `python.exe` 路径，
> `start.bat` 启动时会自动使用该环境的 PyTorch，无需重复安装。

## 使用

### 一键启动（Windows）

**双击 `start.bat`**，然后浏览器访问 `http://127.0.0.1:7860`

### 配置工具（推荐）

首次使用或需要修改 CUDA / PyTorch 设置时，运行配置工具：

**双击 `config_tool.bat`**，然后浏览器访问 `http://127.0.0.1:7861`

配置工具提供图形化界面，包含：
- **系统诊断**：查看 GPU、CUDA 驱动、PyTorch 状态
- **设备 & 目录配置**：切换 CPU / GPU 模式、指定 CUDA 安装目录、指定 Python 环境路径
- **PyTorch 管理**：检测 CUDA 可用性、查看安装命令
- **模型状态**：扫描并列出所有已安装的放大/修复模型

设置保存后，`start.bat` 启动时会自动应用。

### 命令行启动

```bash
python main.py
```

### 更多选项

```bash
# 指定端口
python main.py --port 7860

# CPU 模式
python main.py --device cpu

# 创建公开分享链接
python main.py --share

# 指定输出目录
python main.py --output-dir "D:\my_outputs"

# 复用外部 WebUI 的模型文件
python main.py --webui-path "D:\sd-webui-aki-..."
```

## 模型文件

### 已内置的放大模型

| 类型 | 位置 | 文件 |
|------|------|------|
| ESRGAN | `models/ESRGAN/` | ESRGAN_4x.pth, BSRGAN.pth, 4x-AnimeSharp.pth |
| RealESRGAN | `models/RealESRGAN/` | RealESRGAN_x4plus.pth, RealESRGAN_x4plus_anime_6B.pth |
| SwinIR | `models/SwinIR/` | SwinIR_4x.pth |

### 人脸修复模型（如需要）

面部修复功能需要手动下载以下模型文件，放入对应目录后重启即可：

1. **GFPGANv1.4.pth**（~340MB）
   - 下载：https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
   - 放到：`models/GFPGAN/GFPGANv1.4.pth`

2. **codeformer-v0.1.0.pth**（~340MB）
   - 下载：https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
   - 放到：`models/Codeformer/codeformer-v0.1.0.pth`

## 项目结构

```
sd-enhance/
├── start.bat             # 一键启动（Windows）
├── config_tool.bat       # 配置工具启动脚本（Windows）
├── release_package.bat   # Release 打包脚本（生成含模型的 zip）
├── main.py               # 程序入口
├── config_tool.py        # 图形化配置工具
├── sd_enhance_config.json # 持久化配置文件（自动生成，本地使用）
├── requirements.txt      # Python 依赖
├── LICENSE               # AGPL-3.0
├── README.md             # 本文件
├── models/               # 模型文件（Release zip 内含，不从 git 拉取）
│   ├── ESRGAN/
│   ├── RealESRGAN/
│   └── SwinIR/
├── sd_enhance/           # 核心代码
│   ├── config.py         # 配置管理
│   ├── app.py            # Gradio UI
│   ├── pipeline.py       # 处理管线
│   ├── postprocessing/   # 后期处理操作
│   ├── models/           # 模型封装
│   └── utils/            # 工具函数
├── output/               # 输出目录（自动创建）
└── venv/                 # 虚拟环境（.gitignore 忽略）
```

## 许可

AGPL-3.0

本项目基于 [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 的后期处理模块分离而来，遵循其 AGPL-3.0 许可。


## 配置文件

| 文件 | 说明 |
|------|------|
| `sd_enhance_config.json` | 持久化配置文件（由 config_tool.py 生成和修改） |
| `config_tool.py` | 图形化配置工具（Port 7861） |
| `config_tool.bat` | 配置工具一键启动脚本 |

### sd_enhance_config.json 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | str | `"auto"` | 计算设备：`auto` / `cuda` / `cpu` |
| `half` | bool | `true` | FP16 半精度（仅 CUDA 下生效） |
| `port` | int | `7860` | Gradio 服务端口 |
| `models_dir` | str | `null` | 自定义模型目录（默认 `./models`） |
| `output_dir` | str | `null` | 自定义输出目录（默认 `./output`） |
| `cuda_path` | str | `null` | 手动指定 CUDA Toolkit 路径 |
| `python_path` | str | `null` | 手动指定 Python 环境路径 |

> CLI 参数优先级高于配置文件。例如 `--device cpu` 会覆盖配置中的 `"cuda"`。
