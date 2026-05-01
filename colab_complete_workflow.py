#!/usr/bin/env python3
"""
Colab 训练完整工作流脚本
=======================

这个脚本包含 Colab 中完整的 LoRA 训练流程：
1. 从本地 HTTP 服务器下载数据
2. 提取和验证数据集
3. 执行干运行检查
4. 执行真实训练
5. 收集和分析结果
6. 下载最终结果

使用方法：
在 Colab 笔记本中创建新单元格，粘贴以下代码并运行。
"""

import subprocess
import os
import json
import sys
from pathlib import Path
from typing import Dict, Any
import time

# ============================================================================
# 配置
# ============================================================================

DOWNLOAD_URL = "http://172.30.189.137:8765/fern_new.zip"
TARGET_ZIP = "/content/fern_new.zip"
DATASET_DIR = "/content/dataset"
WORKDIR = "/content/lora-trainer"
CONFIG_PATH = "examples/config_fern_test.yaml"
RUN_DIR = "/content/runs/test_fern"


# ============================================================================
# 工具函数
# ============================================================================

def log_section(title: str):
    """打印格式化的分组标题"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def run_command(cmd: str, description: str = "") -> tuple[int, str, str]:
    """运行命令并返回返回码和输出"""
    if description:
        print(f"\n▶ {description}")
        print(f"  Command: {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=WORKDIR if "python -m" in cmd else None
    )
    
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"❌ Error: {result.stderr}")
    
    return result.returncode, result.stdout, result.stderr


def check_file_exists(path: str, description: str = "") -> bool:
    """检查文件是否存在"""
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    size = f" ({Path(path).stat().st_size / 1024 / 1024:.1f} MB)" if exists else ""
    print(f"{status} {description or path}{size}")
    return exists


# ============================================================================
# Step 1: 下载数据集
# ============================================================================

log_section("Step 1: 下载数据集")

if check_file_exists(TARGET_ZIP, "检查 ZIP 是否已存在"):
    print("✓ 数据文件已存在，跳过下载")
else:
    print(f"📥 从 {DOWNLOAD_URL} 下载数据...")
    returncode, stdout, stderr = run_command(
        f"wget {DOWNLOAD_URL} -O {TARGET_ZIP} --quiet --show-progress",
        "下载数据集"
    )
    
    if returncode == 0 and check_file_exists(TARGET_ZIP):
        print("✓ 下载完成")
    else:
        print("❌ 下载失败")
        sys.exit(1)


# ============================================================================
# Step 2: 提取数据集
# ============================================================================

log_section("Step 2: 提取数据集")

os.makedirs(DATASET_DIR, exist_ok=True)
print(f"▶ 提取 {TARGET_ZIP} 到 {DATASET_DIR}")

run_command(f"rm -rf {DATASET_DIR}/*", "清空旧数据")
run_command(f"unzip -q {TARGET_ZIP} -d {DATASET_DIR}", "解压数据集")

# 检查提取结果
extracted_path = Path(DATASET_DIR) / "fern_new"
if extracted_path.exists():
    print(f"✓ 提取成功: {extracted_path}")
else:
    print(f"✗ 提取失败: 找不到 {extracted_path}")
    sys.exit(1)


# ============================================================================
# Step 3: 验证数据集
# ============================================================================

log_section("Step 3: 验证数据集")

print("▶ 检查图像和标题文件对应关系...")
images = list(extracted_path.glob("*.png")) + list(extracted_path.glob("*.jpg"))
print(f"✓ 找到 {len(images)} 个图像文件")

missing_captions = []
for img in images:
    txt = img.with_suffix(".txt")
    if not txt.exists():
        missing_captions.append(str(img))

if missing_captions:
    print(f"✗ 缺少 {len(missing_captions)} 个标题文件:")
    for p in missing_captions[:5]:
        print(f"  - {p}")
    sys.exit(1)
else:
    print(f"✓ 所有图像都有对应的标题文件")


# ============================================================================
# Step 4: 注入触发令牌
# ============================================================================

log_section("Step 4: 注入触发令牌")

TRIGGER_TOKEN = "f3rn_char"
caption_files = sorted(extracted_path.glob("*.txt"))

print(f"▶ 向 {len(caption_files)} 个标题文件添加触发令牌: '{TRIGGER_TOKEN}'")

updated = 0
for txt_path in caption_files:
    text = txt_path.read_text(encoding="utf-8").strip()
    if not text.startswith(TRIGGER_TOKEN):
        txt_path.write_text(
            f"{TRIGGER_TOKEN}, {text}" if text else TRIGGER_TOKEN,
            encoding="utf-8"
        )
        updated += 1

print(f"✓ 更新了 {updated} 个文件")
print(f"📝 使用提示词示例:")
print(f"   - '{TRIGGER_TOKEN}, 1girl'")
print(f"   - 'portrait of {TRIGGER_TOKEN}, 1girl'")


# ============================================================================
# Step 5: 干运行（配置验证）
# ============================================================================

log_section("Step 5: 干运行（配置验证）")

returncode, _, _ = run_command(
    f"cd {WORKDIR} && PYTHONPATH={WORKDIR} python -m src.lora_trainer.cli "
    f"--config {CONFIG_PATH} --dataset {extracted_path} --run-dir {RUN_DIR} --dry-run",
    "执行干运行检查"
)

if returncode != 0:
    print("❌ 干运行失败，停止训练")
    sys.exit(1)

print("✓ 干运行检查通过")


# ============================================================================
# Step 6: 实际训练
# ============================================================================

log_section("Step 6: 启动训练")

print(f"⏱  开始训练... (预计 10-30 分钟，取决于 GPU)")
print(f"   配置: {CONFIG_PATH}")
print(f"   数据集: {extracted_path}")
print(f"   输出: {RUN_DIR}")

start_time = time.time()

returncode, _, _ = run_command(
    f"cd {WORKDIR} && PYTHONPATH={WORKDIR} python -m src.lora_trainer.cli "
    f"--config {CONFIG_PATH} --dataset {extracted_path} --run-dir {RUN_DIR}",
    "执行训练"
)

elapsed = time.time() - start_time
minutes = int(elapsed / 60)
seconds = int(elapsed % 60)

if returncode != 0:
    print(f"❌ 训练失败 (耗时: {minutes}m {seconds}s)")
    sys.exit(1)

print(f"✓ 训练完成 (耗时: {minutes}m {seconds}s)")


# ============================================================================
# Step 7: 收集和分析结果
# ============================================================================

log_section("Step 7: 收集和分析结果")

runs = sorted(Path(RUN_DIR).glob("run_*"))
if not runs:
    print("❌ 找不到训练输出目录")
    sys.exit(1)

latest_run = runs[-1]
print(f"✓ 找到运行目录: {latest_run.name}")

# 读取元数据
metadata_path = latest_run / "metadata.json"
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"\n📊 训练指标:")
    print(f"  总步数: {metadata.get('total_steps', 'N/A')}")
    print(f"  初始 loss: {metadata.get('first_loss', 'N/A'):.4f}" if 'first_loss' in metadata else "  初始 loss: N/A")
    print(f"  最终 loss: {metadata.get('final_loss', 'N/A'):.4f}" if 'final_loss' in metadata else "  最终 loss: N/A")
    
    if 'effectiveness' in metadata:
        eff = metadata['effectiveness']
        print(f"  有效性检查: {'✓ 通过' if eff.get('passed') else '✗ 未通过'}")
        if eff.get('reasons'):
            for reason in eff['reasons']:
                print(f"    - {reason}")
else:
    print("⚠  metadata.json 未找到")

# 检查导出的 LoRA
lora_path = latest_run / "export" / "lora_final.safetensors"
if check_file_exists(str(lora_path), "LoRA 文件"):
    print("✓ LoRA 模型已导出")

# 检查日志
log_path = latest_run / "logs" / "train.log"
if log_path.exists():
    print(f"\n📋 训练日志 (最后 20 行):")
    with open(log_path) as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(f"   {line.rstrip()}")


# ============================================================================
# Step 8: 准备下载
# ============================================================================

log_section("Step 8: 准备下载")

print("▶ 打包运行结果...")
archive_path = f"/content/{latest_run.name}_artifacts.zip"
run_command(f"cd /content && zip -r {archive_path} {latest_run.name} >/dev/null 2>&1", "创建压缩包")

if check_file_exists(archive_path, "压缩包"):
    print(f"\n✓ 压缩包已创建:")
    print(f"   路径: {archive_path}")
    print(f"\n📥 在 Colab Files 面板中可以下载此文件")
    print(f"   或使用: from google.colab import files; files.download('{archive_path}')")


# ============================================================================
# 总结
# ============================================================================

log_section("训练完成总结")

print(f"""
✓ 所有步骤完成！

📊 结果位置:
  - 运行目录: {latest_run}
  - LoRA 文件: {lora_path}
  - 日志文件: {log_path}
  - 下载包: {archive_path}

📝 后续步骤:
  1. 下载压缩包到本地
  2. 在 diffusers 中加载 LoRA 权重
  3. 使用提示词 "{TRIGGER_TOKEN}, ..." 生成图像
  4. 评估训练效果

🎨 使用示例:
  from diffusers import StableDiffusionPipeline
  pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
  pipeline.load_lora_weights("{lora_path.parent}")
  images = pipeline(prompt="{TRIGGER_TOKEN}, 1girl").images
  images[0].save("output.png")
""")
