# Colab 完整训练工作流指南

## 📋 概述

这份指南将带你完成从数据上传到模型验证的完整 Colab LoRA 训练流程。

**预计时间**: 35-45 分钟  
**GPU**: T4 (Colab 免费)  
**成本**: 0（免费）

---

## 🔑 关键信息

| 项目 | 值 |
|------|-----|
| 数据来源 | 本地 HTTP 服务器: `http://172.30.189.137:8765/fern_new.zip` |
| 本机 IP | `172.30.189.137` |
| HTTP 端口 | `8765` |
| ZIP 文件 | `fern_new.zip` (~150 MB) |
| 数据集大小 | 63 个配对的图像/标题 |
| Colab 工作目录 | `/content/lora-trainer` |
| 运行输出 | `/content/runs/test_fern` |

---

## 🚀 执行步骤

### 步骤 1: 打开 Colab 笔记本

1. 打开: https://colab.research.google.com
2. 打开现有笔记本 `examples/colab_lora_trainer.ipynb`（如果已有）
3. 或创建新笔记本

### 步骤 2: 运行 Cells 1-3（准备环境）

按顺序运行：

**Cell 1: 检查 GPU**
```python
import os
import platform

print('Python:', platform.python_version())
print('Platform:', platform.platform())

gpu_name = os.popen('nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null').read().strip()
if gpu_name:
    print('GPU:', gpu_name)
else:
    raise RuntimeError('No NVIDIA GPU found. Switch Colab runtime to GPU first.')
```

**Cell 2: 克隆仓库**
```bash
!git clone --depth 1 -b master https://github.com/neverbiasu/lora-trainer.git /content/lora-trainer
%cd /content/lora-trainer
!git rev-parse --short HEAD
```

**Cell 3: 安装依赖**
```bash
!python -m pip install -U pip setuptools wheel
!pip install -e .
!pip install -e '.[xformers]' || true
```

### 步骤 3: 运行完整训练工作流

**Cell 4: 复制并执行以下完整脚本**

```python
import subprocess
import os
import json
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
print("  LoRA 训练完整工作流")
print("=" * 70)

# 1. 下载
print("\n1️⃣ 下载数据集...")
if not Path(TARGET_ZIP).exists():
    os.system(f"wget {DOWNLOAD_URL} -O {TARGET_ZIP} --quiet --show-progress")

print(f"✓ 数据文件: {Path(TARGET_ZIP).stat().st_size / 1024 / 1024:.1f} MB")

# 2. 提取
print("\n2️⃣ 提取数据集...")
os.makedirs(DATASET_DIR, exist_ok=True)
os.system(f"rm -rf {DATASET_DIR}/*")
os.system(f"unzip -q {TARGET_ZIP} -d {DATASET_DIR}")

extracted_path = Path(DATASET_DIR) / "fern_new"
print(f"✓ 提取路径: {extracted_path}")

# 3. 验证
print("\n3️⃣ 验证数据集...")
images = list(extracted_path.glob("*.png")) + list(extracted_path.glob("*.jpg"))
print(f"✓ 找到 {len(images)} 个图像")

missing = []
for img in images:
    if not img.with_suffix(".txt").exists():
        missing.append(str(img))

print(f"✓ 缺失标题: {len(missing)}")

# 4. 注入令牌
print("\n4️⃣ 注入触发令牌...")
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

print(f"✓ 更新了 {updated} 个文件")
print(f"📝 提示词: '{TRIGGER_TOKEN}, 1girl'")

# 5. 干运行
print("\n5️⃣ 干运行检查...")
os.chdir(WORKDIR)
import subprocess
result = subprocess.run(
    f"PYTHONPATH={WORKDIR} python -m src.lora_trainer.cli "
    f"--config {CONFIG_PATH} --dataset {extracted_path} --run-dir {RUN_DIR} --dry-run",
    shell=True, capture_output=True, text=True
)
if result.returncode == 0:
    print("✓ 配置有效")
else:
    print(f"✗ 配置检查失败: {result.stderr[:200]}")

# 6. 训练
print("\n6️⃣ 启动训练...")
print("   (这会花 5-15 分钟，请耐心等待)\n")

import time
start = time.time()

result = subprocess.run(
    f"PYTHONPATH={WORKDIR} python -m src.lora_trainer.cli "
    f"--config {CONFIG_PATH} --dataset {extracted_path} --run-dir {RUN_DIR}",
    shell=True, text=True
)

elapsed = int(time.time() - start)
print(f"\n✓ 训练完成 (耗时: {elapsed//60}m {elapsed%60}s)")

# 7. 收集结果
print("\n7️⃣ 收集结果...")
runs = sorted(Path(RUN_DIR).glob("run_*"))
if runs:
    latest_run = runs[-1]
    print(f"✓ 运行目录: {latest_run.name}")
    
    # 读取元数据
    meta_path = latest_run / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n📊 训练指标:")
        print(f"   总步数: {meta.get('total_steps')}")
        print(f"   初始 loss: {meta.get('first_loss', 0):.4f}")
        print(f"   最终 loss: {meta.get('final_loss', 0):.4f}")
        
        if 'effectiveness' in meta:
            eff = meta['effectiveness']
            print(f"   有效性: {'✓ 通过' if eff.get('passed') else '✗ 未通过'}")
    
    # LoRA 文件
    lora = latest_run / "export" / "lora_final.safetensors"
    if lora.exists():
        print(f"   LoRA 文件: ✓ ({lora.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # 日志
    log = latest_run / "logs" / "train.log"
    if log.exists():
        print(f"\n📋 最后 10 行日志:")
        with open(log) as f:
            for line in f.readlines()[-10:]:
                print(f"   {line.rstrip()}")

print("\n" + "=" * 70)
print("✓ 训练工作流完成！")
print("=" * 70)
```

---

## 🎨 验证训练有效性（可视化校验）

### Cell 5: 生成对比图像（Base vs LoRA）

```python
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageOps, ImageDraw

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
TRIGGER_TOKEN = "f3rn_char"
LORA_SCALE = 0.8

# 查找 LoRA 文件
runs = sorted(Path("/content/runs/test_fern").glob("run_*/export/lora_final.safetensors"))
if not runs:
    print("❌ 找不到 LoRA 文件")
else:
    lora_path = runs[-1]
    print(f"✓ 使用 LoRA: {lora_path}")
    
    # 初始化管道
    print("\n加载模型...")
    pipe_base = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    
    pipe_lora = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe_lora.load_lora_weights(lora_path.parent.as_posix(), weight_name=lora_path.name)
    pipe_lora.fuse_lora(lora_scale=LORA_SCALE)
    pipe_lora = pipe_lora.to("cuda")
    
    # 生成图像
    PROMPTS = [
        f"{TRIGGER_TOKEN}",
        f"portrait of {TRIGGER_TOKEN}",
        f"{TRIGGER_TOKEN}, 1girl",
    ]
    SEEDS = [42, 123]
    
    print(f"\n生成图像 ({len(PROMPTS)} 个提示词 x {len(SEEDS)} 个 seed)...")
    
    for i, prompt in enumerate(PROMPTS):
        print(f"\n提示词 {i+1}: '{prompt}'")
        
        for seed in SEEDS:
            # Base 模型
            gen_base = torch.Generator("cuda").manual_seed(seed)
            img_base = pipe_base(prompt, generator=gen_base, num_inference_steps=28).images[0]
            
            # LoRA 模型
            gen_lora = torch.Generator("cuda").manual_seed(seed)
            img_lora = pipe_lora(prompt, generator=gen_lora, num_inference_steps=28).images[0]
            
            # 并排显示
            w, h = img_base.size
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(img_base, (0, 0))
            combined.paste(img_lora, (w, 0))
            
            # 添加标签
            draw = ImageDraw.Draw(combined)
            draw.text((10, 10), "Base", fill="white")
            draw.text((w + 10, 10), "LoRA", fill="white")
            
            # 显示
            display(combined)
            print(f"  seed={seed}: Generated")

print("\n✓ 可视化对比完成")
print("\n评估指标:")
print("  - 相似度: LoRA 和 Base 图像的差异程度")
print("  - 一致性: 同一 seed 下 LoRA 的可复现性")
print("  - 特征学习: LoRA 是否学到了 trigger token 的特征")
```

---

## 📊 日志分析

### Cell 6: 分析训练日志

```python
import re
from pathlib import Path
import json

# 找到最新的运行
runs = sorted(Path("/content/runs/test_fern").glob("run_*"))
if runs:
    latest_run = runs[-1]
    log_path = latest_run / "logs" / "train.log"
    
    if log_path.exists():
        print(f"📋 分析日志: {log_path.name}\n")
        
        with open(log_path) as f:
            lines = f.readlines()
        
        # 提取指标
        steps = []
        losses = []
        
        for line in lines:
            if "step=" in line:
                match = re.search(r'step=(\d+).*loss=([0-9.]+)', line)
                if match:
                    steps.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
        
        if losses:
            print(f"📊 训练统计:")
            print(f"   总步数: {len(losses)}")
            print(f"   初始 loss: {losses[0]:.4f}")
            print(f"   最终 loss: {losses[-1]:.4f}")
            print(f"   最低 loss: {min(losses):.4f}")
            print(f"   最高 loss: {max(losses):.4f}")
            
            # 计算 loss 下降比例
            loss_ratio = losses[-1] / losses[0]
            print(f"   Loss 下降: {(1 - loss_ratio) * 100:.1f}%")
            
            # 检查有效性
            if loss_ratio < 1.0:
                print(f"\n✓ Loss 在下降，训练有效")
            else:
                print(f"\n⚠ Loss 在上升，训练可能无效")
        
        # 显示最后 20 行
        print(f"\n最后 20 行日志:")
        for line in lines[-20:]:
            print(f"  {line.rstrip()}")
    else:
        print("❌ 日志文件不存在")
else:
    print("❌ 找不到运行目录")
```

---

## 📥 下载结果

### Cell 7: 下载所有结果

```python
from google.colab import files
from pathlib import Path

# 找到运行目录
runs = sorted(Path("/content/runs/test_fern").glob("run_*"))
if runs:
    latest_run = runs[-1]
    
    # 打包
    archive_path = f"/content/{latest_run.name}_artifacts.zip"
    import os
    os.system(f"cd /content && zip -r {archive_path} {latest_run.name} >/dev/null 2>&1")
    
    # 下载
    print(f"📥 下载: {archive_path}")
    files.download(archive_path)
    
    print(f"\n✓ 文件已下载到你的电脑")
    print(f"\n📦 包含内容:")
    print(f"   - lora_final.safetensors (LoRA 权重)")
    print(f"   - metadata.json (训练元数据)")
    print(f"   - train.log (训练日志)")
    print(f"   - checkpoints/ (中间检查点)")
    print(f"   - samples/ (生成的样本)")
else:
    print("❌ 找不到运行结果")
```

---

## 📋 最终总结

完成上述步骤后，你将获得：

### ✅ 完成的工作

- [x] 数据上传和验证
- [x] 模型训练 (GPU 加速)
- [x] LoRA 权重导出
- [x] 可视化对比（Base vs LoRA）
- [x] 训练指标分析
- [x] 日志检查
- [x] 结果下载

### 📊 关键指标

| 指标 | 说明 |
|------|------|
| 总步数 | 应该 ≥ 100 |
| Loss 下降 | 应该 > 10% |
| LoRA Delta | 应该 > 1e-6 |
| 文件大小 | lora_final.safetensors: 150-200 MB |

### 🎯 成功标志

✓ 训练日志中 loss 持续下降  
✓ metadata.json 中 `effectiveness_passed: true`  
✓ 可视化对比中 LoRA 和 Base 图像有明显差异  
✓ lora_final.safetensors 文件已生成  

### 🚀 后续步骤

1. 下载 `lora_final.safetensors`
2. 在本地使用 diffusers 加载
3. 生成高质量图像
4. 评估和迭代

---

## 🆘 常见问题

**Q: 下载速度很慢？**  
A: 这取决于网络连接。HTTP 服务器运行在本地，确保网络连接正常。

**Q: GPU 内存不足？**  
A: 减少 `batch_size`（在 config YAML 中），或减少 `max_train_steps`。

**Q: Loss 没有下降？**  
A: 检查数据集是否正确提取，确保所有图像都有对应的标题。

**Q: LoRA 文件生成失败？**  
A: 检查 `/content/runs/test_fern` 中的日志，查看具体错误信息。

---

## 📞 获取帮助

- 查看 [COLAB_DIAGNOSTIC_REPORT.md](./COLAB_DIAGNOSTIC_REPORT.md)
- 查看项目文档: [docs/](./docs/)
- 检查错误日志: `/content/runs/test_fern/run_*/logs/train.log`
