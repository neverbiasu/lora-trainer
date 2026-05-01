#!/usr/bin/env python3
"""
Colab 训练自动化指南 - 完整流程清单

本文档提供了完整的自动化 Colab LoRA 训练工作流程。
由于浏览器自动化遇到 API key 限制，下面是手动执行的清晰步骤。
"""

import os
import json
from pathlib import Path
from datetime import datetime


def print_header(text: str, char: str = "="):
    """打印格式化的标题"""
    width = 80
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}\n")


def print_step(num: int, title: str, substeps: list = None):
    """打印步骤"""
    print(f"\n{'█' * 3} 步骤 {num}: {title}")
    if substeps:
        for i, step in enumerate(substeps, 1):
            print(f"    {i}. {step}")


def generate_colab_notebook_cells():
    """生成完整的 Colab 笔记本单元格"""
    
    cells = {}
    
    cells['cell_0'] = {
        'title': 'LoRA Trainer on Google Colab',
        'type': 'markdown',
        'content': '''# LoRA Trainer on Google Colab - 完整工作流

这是一个完全自动化的 LoRA 训练笔记本。

**需要的配置:**
- 确保已启用 GPU (Runtime -> Change runtime type -> GPU)
- 本地 HTTP 服务器运行在 `http://172.30.189.137:8765`
- 数据文件: `fern_new.zip` (~150 MB)

**总预计时间:** 35-50 分钟
'''
    }
    
    cells['cell_1'] = {
        'title': '1. GPU 检查',
        'type': 'code',
        'content': '''import os
import platform
import torch

print('=' * 60)
print('GPU 检查')
print('=' * 60)
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')

gpu_name = os.popen('nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null').read().strip()
if gpu_name:
    print(f'GPU: {gpu_name} ✓')
else:
    raise RuntimeError('❌ 未找到 GPU。请切换到 GPU 运行时。')

# 其他信息
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')
'''
    }
    
    cells['cell_2'] = {
        'title': '2. 克隆仓库',
        'type': 'code',
        'content': '''import subprocess

repo_url = 'https://github.com/neverbiasu/lora-trainer.git'
branch = 'master'
workdir = '/content/lora-trainer'

print('克隆仓库...')
subprocess.run(f'rm -rf {workdir}', shell=True)
subprocess.run(f'git clone --depth 1 -b {branch} {repo_url} {workdir}', shell=True)

print(f'✓ 仓库已克隆到: {workdir}')

# 获取最新提交
result = subprocess.run(f'git -C {workdir} rev-parse --short HEAD', shell=True, capture_output=True, text=True)
print(f'✓ 最新提交: {result.stdout.strip()}')
'''
    }
    
    cells['cell_3'] = {
        'title': '3. 安装依赖',
        'type': 'code',
        'content': '''import subprocess
import sys

print('安装依赖...')

# 升级基础工具
subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 'pip', 'setuptools', 'wheel'], check=True)

# 安装项目
print('\\n安装 lora-trainer...')
subprocess.run('cd /content/lora-trainer && pip install -e .', shell=True, check=True)

# 可选: xformers (可能失败，不影响主体功能)
print('\\n安装 xformers (可选)...')
subprocess.run('cd /content/lora-trainer && pip install -e ".[xformers]" 2>&1 | head -20', shell=True)

print('\\n✓ 依赖安装完成')
'''
    }
    
    cells['cell_4'] = {
        'title': '4. 下载 + 训练 (完整工作流)',
        'type': 'code',
        'content': '''# 完整的下载、提取、验证、训练工作流
import subprocess
import os
import json
import time
from pathlib import Path

# 配置
DOWNLOAD_URL = "http://172.30.189.137:8765/fern_new.zip"
TARGET_ZIP = "/content/fern_new.zip"
DATASET_DIR = "/content/dataset"
WORKDIR = "/content/lora-trainer"
CONFIG_PATH = "examples/config_fern_test.yaml"
RUN_DIR = "/content/runs/test_fern"
TRIGGER_TOKEN = "f3rn_char"

print("=" * 70)
print(" LoRA 训练完整工作流 - 数据下载到模型训练")
print("=" * 70)

# ========== 1. 下载 ==========
print("\\n[1/7] 📥 下载数据集...")
if not Path(TARGET_ZIP).exists():
    result = subprocess.run(
        f'wget {DOWNLOAD_URL} -O {TARGET_ZIP} --quiet --show-progress',
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f'❌ 下载失败: {result.stderr[:100]}')
        raise RuntimeError('Download failed')

zip_size = Path(TARGET_ZIP).stat().st_size / 1024 / 1024
print(f'✓ 文件大小: {zip_size:.1f} MB')

# ========== 2. 提取 ==========
print("\\n[2/7] 📂 提取数据集...")
os.makedirs(DATASET_DIR, exist_ok=True)
subprocess.run(f'rm -rf {DATASET_DIR}/*', shell=True)
subprocess.run(f'unzip -q {TARGET_ZIP} -d {DATASET_DIR}', shell=True, check=True)

extracted_path = Path(DATASET_DIR) / "fern_new"
if not extracted_path.exists():
    raise RuntimeError(f'提取失败: {extracted_path} 不存在')

print(f'✓ 提取路径: {extracted_path}')

# ========== 3. 验证 ==========
print("\\n[3/7] ✓ 验证数据集...")
images = list(extracted_path.glob("*.png")) + list(extracted_path.glob("*.jpg"))
print(f'  图像数量: {len(images)}')

missing = []
for img in images:
    if not img.with_suffix(".txt").exists():
        missing.append(str(img))

if missing:
    print(f'❌ 缺少 {len(missing)} 个标题文件')
    raise RuntimeError('Dataset validation failed')

print(f'✓ 所有图像都有对应的标题')

# ========== 4. 注入令牌 ==========
print("\\n[4/7] 🔤 注入触发令牌...")
caption_files = sorted(extracted_path.glob("*.txt"))
updated = 0
for txt_path in caption_files:
    text = txt_path.read_text(encoding="utf-8").strip()
    if not text.startswith(TRIGGER_TOKEN):
        txt_path.write_text(
            f"{TRIGGER_TOKEN}, {text}" if text else TRIGGER_TOKEN,
            encoding="utf-8"
        )
        updated += 1

print(f'✓ 更新了 {updated} 个标题文件')
print(f'  触发令牌: "{TRIGGER_TOKEN}"')
print(f'  示例提示词: "{TRIGGER_TOKEN}, 1girl"')

# ========== 5. 干运行 ==========
print("\\n[5/7] 🧪 运行干运行检查...")
result = subprocess.run(
    f'cd {WORKDIR} && PYTHONPATH={WORKDIR} python -m src.lora_trainer.cli '
    f'--config {CONFIG_PATH} --dataset {extracted_path} --run-dir {RUN_DIR} --dry-run',
    shell=True, capture_output=True, text=True
)

if result.returncode != 0:
    print(f'❌ 干运行失败')
    print(f'错误: {result.stderr[:200]}')
    raise RuntimeError('Dry-run failed')

print('✓ 配置有效')

# ========== 6. 训练 ==========
print("\\n[6/7] 🚀 启动训练...")
print("⏱ 这需要 5-15 分钟，请耐心等待...")

start_time = time.time()

result = subprocess.run(
    f'cd {WORKDIR} && PYTHONPATH={WORKDIR} python -m src.lora_trainer.cli '
    f'--config {CONFIG_PATH} --dataset {extracted_path} --run-dir {RUN_DIR}',
    shell=True, capture_output=True, text=True
)

elapsed = int(time.time() - start_time)

if result.returncode != 0:
    print(f'❌ 训练失败')
    print(f'日志: {result.stdout[-500:]}')
    raise RuntimeError('Training failed')

print(f'✓ 训练完成 (耗时: {elapsed // 60}m {elapsed % 60}s)')

# ========== 7. 收集结果 ==========
print("\\n[7/7] 📊 收集和分析结果...")

runs = sorted(Path(RUN_DIR).glob("run_*"))
if not runs:
    raise RuntimeError('No run directory found')

latest_run = runs[-1]
print(f'✓ 运行目录: {latest_run.name}')

# 读取元数据
meta_path = latest_run / "metadata.json"
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    
    print(f"\\n📈 训练指标:")
    print(f"   总步数: {meta.get('total_steps', 'N/A')}")
    print(f"   初始 loss: {meta.get('first_loss', 0):.4f}")
    print(f"   最终 loss: {meta.get('final_loss', 0):.4f}")
    
    if 'effectiveness' in meta:
        eff = meta['effectiveness']
        status = "✓ 通过" if eff.get('passed') else "✗ 未通过"
        print(f"   有效性: {status}")
        if eff.get('reasons'):
            for reason in eff['reasons']:
                print(f"      - {reason}")

# LoRA 文件
lora_path = latest_run / "export" / "lora_final.safetensors"
if lora_path.exists():
    size_mb = lora_path.stat().st_size / 1024 / 1024
    print(f"   LoRA 文件: ✓ ({size_mb:.1f} MB)")

print("\\n" + "=" * 70)
print("✓ 训练工作流全部完成!")
print("=" * 70)

# 显示日志摘要
log_path = latest_run / "logs" / "train.log"
if log_path.exists():
    print("\\n📋 训练日志 (最后 15 行):")
    with open(log_path) as f:
        lines = f.readlines()
        for line in lines[-15:]:
            print(f"  {line.rstrip()}")
'''
    }
    
    cells['cell_5'] = {
        'title': '5. 可视化对比 (Base vs LoRA)',
        'type': 'code',
        'content': '''# 生成对比图像来验证 LoRA 的有效性
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
import numpy as np

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
TRIGGER_TOKEN = "f3rn_char"
LORA_SCALE = 0.8

print("=" * 60)
print(" 可视化对比: Base vs LoRA")
print("=" * 60)

# 查找 LoRA
runs = sorted(Path("/content/runs/test_fern").glob("run_*/export/lora_final.safetensors"))
if not runs:
    print("❌ 找不到 LoRA 文件")
    raise RuntimeError("LoRA file not found")

lora_path = runs[-1]
print(f"\\n✓ LoRA 文件: {lora_path}")
print(f"  文件大小: {lora_path.stat().st_size / 1024 / 1024:.1f} MB")

# 加载模型
print("\\n加载基础模型...")
pipe_base = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipe_base.set_progress_bar_config(disable=True)

print("加载 LoRA 模型...")
pipe_lora = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe_lora.load_lora_weights(lora_path.parent.as_posix(), weight_name=lora_path.name)
pipe_lora.fuse_lora(lora_scale=LORA_SCALE)
pipe_lora = pipe_lora.to("cuda")
pipe_lora.set_progress_bar_config(disable=True)

# 测试提示词
PROMPTS = [
    f"{TRIGGER_TOKEN}",
    f"portrait of {TRIGGER_TOKEN}",
    f"{TRIGGER_TOKEN}, 1girl",
]
SEEDS = [42, 123]

print(f"\\n生成对比图像...")
print(f"  提示词数: {len(PROMPTS)}")
print(f"  Seed 数: {len(SEEDS)}")
print(f"  总图像数: {len(PROMPTS) * len(SEEDS) * 2} (Base + LoRA)")

for i, prompt in enumerate(PROMPTS):
    print(f"\\n[提示词 {i+1}] {prompt}")
    
    for seed in SEEDS:
        # Base 模型
        gen = torch.Generator("cuda").manual_seed(seed)
        img_base = pipe_base(prompt, generator=gen, num_inference_steps=28, guidance_scale=7.5).images[0]
        
        # LoRA 模型
        gen = torch.Generator("cuda").manual_seed(seed)
        img_lora = pipe_lora(prompt, generator=gen, num_inference_steps=28, guidance_scale=7.5).images[0]
        
        # 创建并排图
        w, h = img_base.size
        combined = Image.new("RGB", (w * 2 + 10, h + 30))
        combined.paste((255, 255, 255), (0, 0, w * 2 + 10, h + 30))
        combined.paste(img_base, (5, 25))
        combined.paste(img_lora, (w + 5, 25))
        
        # 添加标签
        draw = ImageDraw.Draw(combined)
        draw.text((10, 5), "Base", fill="black")
        draw.text((w + 10, 5), "LoRA", fill="black")
        
        display(combined)
        print(f"  seed={seed}: ✓ Generated")

print("\\n✓ 可视化对比完成")
print("\\n📊 评估指标:")
print("  - 差异度: LoRA 和 Base 输出的差异程度")
print("  - 一致性: 同 seed 下的可复现性")
print("  - 学习效果: LoRA 是否学到了新的特征")
'''
    }
    
    cells['cell_6'] = {
        'title': '6. 日志分析',
        'type': 'code',
        'content': '''# 详细的日志分析
import re
from pathlib import Path
import json

print("=" * 60)
print(" 训练日志分析")
print("=" * 60)

# 查找运行
runs = sorted(Path("/content/runs/test_fern").glob("run_*"))
if not runs:
    print("❌ 找不到运行目录")
else:
    latest_run = runs[-1]
    log_path = latest_run / "logs" / "train.log"
    
    if log_path.exists():
        print(f"\\n📋 日志文件: {log_path}")
        
        with open(log_path) as f:
            content = f.read()
            lines = content.split("\\n")
        
        # 提取指标
        print("\\n📊 提取指标...")
        losses = []
        steps = []
        
        for line in lines:
            # 查找 step 和 loss 信息
            if "step=" in line and "loss=" in line:
                m = re.search(r'step=(\\d+).*loss=([0-9.e-]+)', line)
                if m:
                    step = int(m.group(1))
                    loss = float(m.group(2))
                    steps.append(step)
                    losses.append(loss)
        
        if losses:
            print(f"  总步数: {len(losses)}")
            print(f"  初始 loss: {losses[0]:.6f}")
            print(f"  最终 loss: {losses[-1]:.6f}")
            print(f"  最低 loss: {min(losses):.6f}")
            print(f"  最高 loss: {max(losses):.6f}")
            
            # 计算下降比例
            loss_drop = (losses[0] - losses[-1]) / losses[0] * 100
            print(f"  Loss 下降: {loss_drop:.1f}%")
            
            # 平均下降速度
            avg_drop_per_step = (losses[0] - losses[-1]) / len(losses)
            print(f"  平均每步下降: {avg_drop_per_step:.6f}")
            
            # 有效性检查
            print("\\n✓ 有效性检查:")
            if loss_drop > 5:
                print(f"  ✓ Loss 明显下降 ({loss_drop:.1f}%)")
            elif loss_drop > 0:
                print(f"  ~ Loss 轻微下降 ({loss_drop:.1f}%)")
            else:
                print(f"  ✗ Loss 未下降或上升")
        
        # 最后 20 行
        print("\\n📝 日志尾部 (最后 20 行):")
        for line in lines[-20:]:
            if line.strip():
                print(f"  {line}")
    else:
        print("❌ 日志文件不存在")

# 元数据
meta_path = latest_run / "metadata.json"
if meta_path.exists():
    print("\\n📋 元数据:")
    with open(meta_path) as f:
        meta = json.load(f)
    
    for key, value in meta.items():
        if not isinstance(value, (dict, list)):
            print(f"  {key}: {value}")
'''
    }
    
    cells['cell_7'] = {
        'title': '7. 下载结果',
        'type': 'code',
        'content': '''# 打包并下载所有结果
from google.colab import files
from pathlib import Path
import os

print("=" * 60)
print(" 下载结果")
print("=" * 60)

# 查找运行
runs = sorted(Path("/content/runs/test_fern").glob("run_*"))
if not runs:
    print("❌ 找不到运行目录")
else:
    latest_run = runs[-1]
    print(f"\\n打包: {latest_run.name}")
    
    # 创建压缩包
    archive_name = f"{latest_run.name}_artifacts"
    print(f"创建压缩包: {archive_name}.zip")
    
    os.system(f"cd /content && zip -q -r {archive_name}.zip {latest_run.name}")
    
    archive_path = f"/content/{archive_name}.zip"
    if Path(archive_path).exists():
        size_mb = Path(archive_path).stat().st_size / 1024 / 1024
        print(f"✓ 压缩包大小: {size_mb:.1f} MB")
        
        print("\\n📥 开始下载...")
        files.download(archive_path)
        
        print("✓ 下载完成!")
        
        print("\\n📦 压缩包包含:")
        print(f"  ✓ lora_final.safetensors - LoRA 权重文件")
        print(f"  ✓ metadata.json - 训练元数据")
        print(f"  ✓ logs/ - 训练日志")
        print(f"  ✓ checkpoints/ - 中间检查点")
        print(f"  ✓ samples/ - 生成的样本图像 (如果有)")
        
        print("\\n🎯 后续步骤:")
        print(f"  1. 下载压缩包到本地电脑")
        print(f"  2. 解压文件")
        print(f"  3. 在本地 Python 环境中使用 LoRA 文件")
        print(f"     from diffusers import StableDiffusionPipeline")
        print(f"     pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')")
        print(f"     pipe.load_lora_weights('path/to/lora_final.safetensors')")
    else:
        print("❌ 压缩包创建失败")
'''
    }
    
    return cells


# ============================================================================
# 主程序
# ============================================================================

def main():
    print_header("Colab 完整训练工作流 - 执行指南")
    
    print("""
欢迎使用 LoRA Trainer Colab 自动化工作流。

本指南会逐步引导你完成从数据下载到模型验证的全过程。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 总体流程概览
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    print_step(1, "GPU 环境检查",
        ["验证 CUDA 可用性", "确认 GPU 类型和内存"])
    
    print_step(2, "克隆项目仓库",
        ["从 GitHub 获取最新代码", "验证仓库完整性"])
    
    print_step(3, "安装依赖",
        ["安装 PyTorch、Diffusers 等", "安装项目特定包 (lora-trainer)"])
    
    print_step(4, "数据处理和训练",
        ["从本地 HTTP 服务器下载 fern_new.zip",
         "解压数据到 /content/dataset",
         "验证图像和标题文件对应关系",
         "注入触发令牌 'f3rn_char'",
         "执行干运行检查配置",
         "执行真实训练 (5-15 分钟)"])
    
    print_step(5, "可视化对比",
        ["加载 Base 模型和 LoRA 模型",
         "使用触发令牌生成图像",
         "创建并排对比图展示差异"])
    
    print_step(6, "日志分析",
        ["提取训练指标 (步数、loss)",
         "计算 loss 下降比例",
         "检查有效性指标"])
    
    print_step(7, "结果下载",
        ["打包训练输出",
         "下载所有结果到本地"])
    
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔑 关键信息
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  本机 IP:              172.30.189.137
  HTTP 服务器端口:      8765
  数据文件:             fern_new.zip (~150 MB)
  Colab 工作目录:        /content/lora-trainer
  运行输出目录:         /content/runs/test_fern
  触发令牌:             f3rn_char
  
  预计总时间:           35-50 分钟
  GPU 需求:             T4 或更好 (Colab 免费)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 如何使用本指南
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 打开 Google Colab: https://colab.research.google.com

2. 创建新笔记本或打开现有笔记本

3. 按照下面的单元格代码顺序，逐个复制到 Colab 中运行

4. 每个单元格会打印进度和错误信息，方便调试

5. 完成后，所有结果会自动下载到你的电脑

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Colab 笔记本单元格代码
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    # 生成单元格
    cells = generate_colab_notebook_cells()
    
    for cell_id, cell_data in cells.items():
        cell_type = cell_data['type']
        title = cell_data['title']
        content = cell_data['content']
        
        print(f"\n{cell_id.upper()} - {title} ({'MARKDOWN' if cell_type == 'markdown' else 'CODE'})")
        print("-" * 70)
        
        if cell_type == 'markdown':
            print(f"【复制到 Markdown 单元格】\n{content}")
        else:
            print(f"【复制到 Code 单元格】\n{content}")
        
        print("\n" + "=" * 70)
    
    print("""

✅ 成功标志
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ✓ 所有 Cell 无错误执行
 ✓ 训练日志显示 loss 持续下降
 ✓ metadata.json 中 effectiveness_passed: true
 ✓ 可视化对比显示 LoRA 和 Base 图像明显不同
 ✓ lora_final.safetensors 文件已生成 (~200 MB)
 ✓ 结果压缩包已下载到本地

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ 常见问题排查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "下载失败: net::ERR_NETWORK_CHANGED"
A: 检查本机 HTTP 服务器是否在运行:
   ps aux | grep "python3 -m http.server"

Q: "GPU not found"
A: Runtime -> Change runtime type -> Hardware accelerator -> GPU

Q: "unzip: cannot find or open fern_new.zip"
A: 确认 HTTP 下载成功，检查 /content/fern_new.zip 是否存在

Q: "Loss 没有下降"
A: 检查数据集是否正确提取，确保所有图像都有 .txt 标题

Q: "Out of memory"
A: 减少 config YAML 中的 batch_size，或减少 max_train_steps

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 相关文档
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 • COLAB_DIAGNOSTIC_REPORT.md  - 详细诊断和修复指南
 • COLAB_WORKFLOW_GUIDE.md      - 完整工作流说明
 • docs/development.md           - 本地开发指南
 • docs/design/cli_design.md    - CLI 设计文档

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎉 准备好了吗？立即开始！
    """)


if __name__ == "__main__":
    main()
