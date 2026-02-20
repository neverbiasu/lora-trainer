# Fern LoRA 训练数据集快速指南

> 使用《芙莉莲彼岸花开》中的 Fern 角色截图快速验证 LoRA 训练功能

## 快速开始（3 步）

> **角色特征**：Fern 是紫色长直头发的年轻女性魔法师（不是白发！那是芙莉莲）

### 第 1 步：手动收集图片（~10 分钟）

创建文件夹放下载的截图：

```bash
mkdir -p ./dataset
```

然后用任意一种方式下载 80-150 张 Fern 截图：

#### 方案 A：Google Images（推荐）
1. 打开 [Google Images](https://images.google.com)
2. 搜索：`"Fern Frieren" purple hair mage` 或 `"フェルン" anime`
3. 筛选：尺寸 → 大（Large）
4. 下载 100-150 张到 `./dataset/` 文件夹

#### 方案 B：MyAnimeList
1. 访问 [Fern - MyAnimeList](https://myanimelist.net/character/194761/Fern)
2. 点击 "Pictures" 标签页
3. 右键保存图片到 `./dataset/`

#### 方案 C：Anime 图片网站
- [Anime Reactor](https://anime.reactor.cc/) - 搜索 "Frieren Fern"
- [Know Your Meme](https://knowyourmeme.com/) - 搜索 Frieren 相关内容

**💡 提示**：图片尺寸至少 512x512 效果最好。混合不同姿态、表情的截图效果更好。

### 第 2 步：自动处理图片（~1 分钟）

运行处理脚本，自动：
- ✅ 过滤和转换格式
- ✅ 去重
- ✅ 生成训练标注

```bash
python scripts/download_fern_dataset.py --skip-download --min-size 512
```

输出：
```
60  Filtering 120 images...
   ✓ Processed 20/120...
   ✓ Processed 40/120...
   ✓ Processed 60/120...
   ✓ Processed 80/120...
   ✓ Processed 100/120...
   ✓ Processed 120/120...
✅ Filtered to 98 valid images

📝 Creating captions for 98 images...
✅ Created 98 caption files

🎉 Success! Created dataset with 98 images
   Location: ./fern_dataset/
```

### 第 3 步：开始训练（~5 分钟）

用 Fern 专用配置训练：

```bash
# 预览配置
lora-trainer --config examples/config_fern_test.yaml \
             --dataset ./fern_dataset \
             --dry-run

# 初始化运行目录
lora-trainer --config examples/config_fern_test.yaml \
             --dataset ./fern_dataset \
             --run-dir ./runs/test_fern
```

## 数据集目录结构

处理完成后，数据集目录如下：

```
fern_dataset/
├── fern_0000.png
├── fern_0000.txt      # "fern, girl, elf, white hair, ..."
├── fern_0001.png
├── fern_0001.txt
├── ...
└── fern_0097.png
```

每张图片有对应的 `.txt` 标注文件。

## 验证效果的方法

通过对比生成效果可视化 LoRA 学到了什么：

```bash
# 无 LoRA（生成普通白发女性）
lora-trainer --mode inference \
  --base-model sd15 \
  --prompt "white hair girl, detailed illustration" \
  --output ./gen_no_lora.png

# 有 LoRA（生成 Fern 风格的特征）
lora-trainer --mode inference \
  --base-model sd15 \
  --lora ./runs/test_fern/export/fern.safetensors \
  --prompt "fern, white hair girl, detailed illustration" \
  --output ./gen_with_lora.png
```

## 训练参数说明

`examples/config_fern_test.yaml` 的配置值：

## Fern 特征清单

识别 Fern 的关键特征（避免混淆）：
- ✅ **紫色长直头发**（关键！）
- ✅ 年轻女性，表情严肃
- ✅ 紫色衣装或魔法师长袍
- ✅ 时常拿着魔法书或魔法杖
- ❌ 不是白色长髮（那是 Frieren）
- ❌ 不是年长的表情（那是 Frieren）

## 训练参数说明

| 参数 | 值 | 说明 |
|------|----|----|
| base_model | sd15 | Stable Diffusion 1.5 |
| rank | 32 | LoRA 秩（default: 32） |
| learning_rate | 1e-4 | 学习率 |
| batch_size | 2 | 批次大小（8GB GPU） |
| max_train_steps | 200 | 训练步数（完整 epoch ~2min） |
| preset | balanced | 平衡配置（介于快速和质量之间） |

**调整建议**：
- 如果显存不足（OOM），降低 `batch_size` 到 1
- 如果训练太慢，改用 `preset: quick`
- 如果要更好效果，增加 `max_train_steps` 到 500

## 常见问题

### Q: 脚本下载不了图片怎么办？
A: 这是正常的。自动下载需要 Selenium 等浏览器驱动。建议用方案 A（Google Images 手动下载）。

### Q: 图片格式不支持怎么办？
A: 脚本自动支持 PNG/JPG。如果有其他格式，可用以下命令转换：
```bash
for f in *.webp; do convert "$f" "${f%.webp}.png"; done
```

### Q: 下载后怎么检查图片质量？
A: 打开 `./fern_dataset/` 文件夹，确保：
- ✅ 都是 PNG 格式
- ✅ 大部分包含 Fern（白发女性角色）
- ✅ 尺寸至少 512x512
- ✅ 没有黑屏或损坏的图片

### Q: 能否用不同角色的数据吗？
A: 可以！脚本支持任意数据集：
```bash
# 修改 script 中的 search queries
# 或直接运行：--skip-download
python scripts/download_fern_dataset.py --skip-download \
  --input ./your_character_data --output ./your_dataset
```

## 预期训练时间

| GPU | Batch | Max Steps | 预计时间 |
|-----|-------|-----------|---------|
| RTX 3090 | 4 | 200 | ~2 min |
| RTX 3080 | 2 | 200 | ~3 min |
| RTX 4000 | 1 | 200 | ~5 min |
| Apple Silicon | 2 | 200 | ~8 min |

## 下一步

✅ 验证训练流程正常运行  
→ 尝试调整参数（rank、learning_rate 等）  
→ 用不同角色数据集测试  
→ 进入 M2：完整训练管道 + 数据加载 + 导出

---

需要帮助？检查以下内容：
- 确保 `./fern_dataset/` 中有 80+ PNG 文件
- 运行 `lora-trainer --config examples/config_fern_test.yaml --dataset ./fern_dataset --dry-run` 检查配置
- 查看 README.md M1 Quick Run 部分
