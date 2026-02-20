# presets 设计

## 1. Preset 设计原则

Preset 的目标是为不同使用场景提供**开箱即用的参数组合**，避免用户手动调参。

### 1.1 三档定位

| Preset    | 目标用户            | 使用场景                          | 权衡点                    |
|-----------|--------------------|---------------------------------|--------------------------|
| **quick** | 初学者、快速验证     | 测试数据集是否可用，快速看到效果    | 速度优先，质量可能不足     |
| **balanced** | 一般用户          | 大多数场景的默认选择              | 平衡质量和时间             |
| **quality** | 追求高质量         | 最终作品、商业应用                | 质量优先，时间成本高       |

### 1.2 设计约束

- **独立完整**：每个 preset 包含所有必要参数，无需额外配置
- **可覆盖**：用户可以通过 CLI 或 config 覆盖任意参数
- **自适应**：根据数据集大小自动调整步数

---

## 2. Preset 参数定义

### 2.1 quick 预设（10-15分钟）

**目标**：快速验证概念，不追求完美质量。

```yaml
quick:
  # LoRA 配置
  rank: 16                    # 较小 rank，减少参数量
  alpha: 16
  
  # 训练配置
  learning_rate: 2e-4         # 较高 lr，快速收敛
  lr_scheduler: "constant"    # 不衰减，保持激进
  lr_warmup_steps: 0          # 无 warmup
  
  batch_size: 4
  gradient_accumulation: 1
  max_train_steps: 1000       # 较少步数
  
  # 优化
  mixed_precision: "fp16"
  enable_xformers: True
  gradient_checkpointing: False  # 不牺牲速度
  
  # 数据
  cache_latents: True         # 强制 cache（如果显存允许）
  enable_bucketing: True
  
  # 检查点
  save_every_n_steps: 500
  sample_every_n_steps: 250
  
  # 验证采样
  num_sample_images: 2        # 少量采样
  sample_steps: 20            # 快速采样（降低质量）
```

**预期效果**：
- 能够学到基本特征，但细节不足
- 适合测试数据集质量、触发词有效性
- 100 张图：~10 分钟（RTX 3060）

---

### 2.2 balanced 预设（15-25分钟）

**目标**：大多数场景的默认选择，平衡质量和时间。

```yaml
balanced:
  # LoRA 配置
  rank: 32                    # 标准 rank
  alpha: 32
  
  # 训练配置
  learning_rate: 1e-4         # 稳健 lr
  lr_scheduler: "cosine"      # 余弦衰减
  lr_warmup_steps: 100        # 短 warmup
  
  batch_size: 4
  gradient_accumulation: 1
  max_train_steps: 1500       # 中等步数
  
  # 优化
  mixed_precision: "fp16"
  enable_xformers: True
  gradient_checkpointing: False
  
  # 数据
  cache_latents: "auto"       # 自动决策
  enable_bucketing: True
  
  # 检查点
  save_every_n_steps: 500
  sample_every_n_steps: 250
  
  # 验证采样
  num_sample_images: 4
  sample_steps: 30            # 标准采样质量
```

**预期效果**：
- 能够学到主要特征和部分细节
- 适合个人项目、社区分享
- 100 张图：~15 分钟（RTX 3060）

---

### 2.3 quality 预设（30-45分钟）

**目标**：追求最佳质量，适合最终作品。

```yaml
quality:
  # LoRA 配置
  rank: 64                    # 较大 rank，更强表达能力
  alpha: 64
  
  # 训练配置
  learning_rate: 5e-5         # 保守 lr，稳定收敛
  lr_scheduler: "cosine"
  lr_warmup_steps: 200        # 更长 warmup
  
  batch_size: 2               # 小 batch，更稳定梯度
  gradient_accumulation: 2    # 模拟 batch=4
  max_train_steps: 3000       # 更多步数
  
  # 优化
  mixed_precision: "fp16"
  enable_xformers: True
  gradient_checkpointing: False
  
  # 数据
  cache_latents: "auto"
  enable_bucketing: True
  
  # 检查点
  save_every_n_steps: 500
  sample_every_n_steps: 250
  
  # 验证采样
  num_sample_images: 8        # 更多采样验证
  sample_steps: 50            # 高质量采样
```

**预期效果**：
- 学到丰富细节，高保真度
- 可能过拟合小数据集（< 50 张）
- 100 张图：~40 分钟（RTX 3060）

---

## 3. 自适应步数计算

### 3.1 基础公式（参考 kohya 经验）

```python
def calculate_max_steps(dataset_size: int, preset: str) -> int:
    """
    根据数据集大小和 preset 计算推荐步数
    
    目标：每张图训练 N 次（epoch equivalent）
    """
    
    # 每个 preset 的目标 epoch 数
    target_epochs = {
        "quick": 30,      # 快速收敛
        "balanced": 40,   # 标准
        "quality": 60,    # 充分训练
    }
    
    # 获取配置
    config = PRESETS[preset]
    effective_batch = config["batch_size"] * config["gradient_accumulation"]
    
    # 计算步数
    max_steps = (dataset_size * target_epochs[preset]) // effective_batch
    
    # 向上取整到 100 的倍数（美观）
    max_steps = ((max_steps + 99) // 100) * 100
    
    return max_steps

# Example
calculate_max_steps(dataset_size=100, preset="balanced")
# → (100 * 40) / 4 = 1000 → 向上取整到 1000
```

### 3.2 步数调整表

| 数据集大小 | quick  | balanced | quality |
|-----------|--------|----------|---------|
| 20 张     | 200    | 250      | 400     |
| 50 张     | 400    | 600      | 1000    |
| 100 张    | 800    | 1200     | 2000    |
| 200 张    | 1600   | 2400     | 4000    |
| 500 张    | 4000   | 6000     | 10000   |
| 1000 张   | 8000   | 12000    | 20000   |

### 3.3 小数据集特殊处理

```python
def adjust_for_small_dataset(config: Dict, dataset_size: int) -> Dict:
    """小数据集（< 100 张）降低 rank 防止过拟合"""
    
    if dataset_size < 50:
        config["rank"] = min(config["rank"], 16)
        config["alpha"] = config["rank"]
        config["learning_rate"] *= 0.5  # 更保守
    elif dataset_size < 100:
        config["rank"] = min(config["rank"], 32)
        config["alpha"] = config["rank"]
    
    return config
```

---

## 4. Preset 使用方式

### 4.1 CLI 使用

```bash
# 使用 quick 预设
lora-trainer train --dataset ./data --base-model sd15 --preset quick

# 使用 quality 预设并覆盖部分参数
lora-trainer train --dataset ./data --base-model sd15 --preset quality --rank 128
```

### 4.2 配置文件使用

```yaml
# config.yaml
training:
  preset: balanced

# CLI 覆盖
# lora-trainer train --config config.yaml --learning-rate 2e-4
```

### 4.3 默认 preset

- 如果用户未指定 preset，默认使用 **balanced**
- 可以通过 `--preset none` 禁用 preset，完全手动配置

---

## 5. Preset 加载优先级

```python
def load_config_with_preset(
    preset_name: str,
    config_file: Optional[str],
    cli_args: Dict,
) -> Dict:
    """
    配置加载顺序：
    1. Preset 默认值
    2. 配置文件覆盖
    3. CLI 参数覆盖
    """
    
    # 1. 加载 preset
    if preset_name and preset_name != "none":
        config = PRESETS[preset_name].copy()
    else:
        config = {}
    
    # 2. 加载配置文件
    if config_file:
        with open(config_file) as f:
            file_config = yaml.safe_load(f)
        config = deep_merge(config, file_config)
    
    # 3. CLI 参数覆盖
    config = deep_merge(config, cli_args)
    
    return config
```

---

## 6. Preset 验证与警告

### 6.1 质量预期提示

```python
def show_preset_info(preset: str, dataset_size: int):
    """显示 preset 预期效果"""
    
    info = {
        "quick": {
            "description": "Quick validation (10-15 min)",
            "quality": "Basic features, may lack details",
            "use_case": "Testing dataset, trigger words, concept validation",
        },
        "balanced": {
            "description": "Balanced quality and speed (15-25 min)",
            "quality": "Good features and partial details",
            "use_case": "Most personal projects, community sharing",
        },
        "quality": {
            "description": "Best quality (30-45 min)",
            "quality": "Rich details, high fidelity",
            "use_case": "Final works, commercial use",
        },
    }
    
    print(f"📋 Preset: {preset}")
    print(f"   {info[preset]['description']}")
    print(f"   Expected Quality: {info[preset]['quality']}")
    print(f"   Best For: {info[preset]['use_case']}")
```

### 6.2 过拟合警告

```python
def check_overfitting_risk(preset: str, dataset_size: int):
    """检查过拟合风险"""
    
    if preset == "quality" and dataset_size < 50:
        print("⚠️ Warning: 'quality' preset with small dataset (<50 images)")
        print("   Risk of overfitting. Consider using 'balanced' preset instead.")
        print("   Or add more training images.")
```

---

## 7. 自定义 Preset（Phase 2）

### 7.1 用户自定义 preset 文件

```yaml
# ~/.lora-trainer/presets/my_preset.yaml
my_preset:
  rank: 48
  alpha: 48
  learning_rate: 8e-5
  max_train_steps: 2000
  # ... 其他参数
```

### 7.2 加载自定义 preset

```bash
lora-trainer train --preset my_preset --preset-file ~/.lora-trainer/presets/my_preset.yaml
```

**MVP 不支持**，Phase 2 特性。

---

## 8. Preset 对比表

| 参数                   | quick   | balanced | quality |
|-----------------------|---------|----------|---------|
| **训练时间 (100张)**   | ~10 min | ~15 min  | ~40 min |
| **rank**              | 16      | 32       | 64      |
| **learning_rate**     | 2e-4    | 1e-4     | 5e-5    |
| **lr_scheduler**      | 常量    | cosine   | cosine  |
| **batch_size**        | 4       | 4        | 2       |
| **gradient_accum**    | 1       | 1        | 2       |
| **max_steps (100张)** | 1000    | 1500     | 3000    |
| **cache_latents**     | 开启    | auto     | auto    |
| **sample_images**     | 2       | 4        | 8       |
| **适用场景**           | 快速验证 | 常规项目  | 高质量作品 |

---

## 9. 实现要点

### 9.1 Preset 定义文件

```python
# presets.py
PRESETS = {
    "quick": {
        "lora": {
            "rank": 16,
            "alpha": 16,
        },
        "training": {
            "learning_rate": 2e-4,
            "lr_scheduler": "constant",
            "batch_size": 4,
            "gradient_accumulation": 1,
            "max_train_steps": 1000,
        },
        "data": {
            "cache_latents": True,
        },
        # ... 其他参数
    },
    "balanced": { ... },
    "quality": { ... },
}
```

### 9.2 CLI 参数解析

```python
parser.add_argument(
    "--preset",
    type=str,
    choices=["quick", "balanced", "quality", "none"],
    default="balanced",
    help="Training preset (default: balanced)",
)
```

---

## 10. 参考实现映射

| 功能模块          | 参考仓库         | 文件路径                                      |
|------------------|-----------------|---------------------------------------------|
| Preset 定义      | flymyai-lora-trainer | `config/presets/` 目录                   |
| 自适应步数        | kohya-ss        | `train_network.py` → 经验公式               |
| 配置合并         | OneTrainer      | `modules/util/config/ConfigManager.py`      |

---

## 11. MVP 简化约束

| 特性              | MVP 状态       | 备注                                    |
|------------------|---------------|----------------------------------------|
| 三档预设          | ✅ 实现        | quick / balanced / quality              |
| 自适应步数        | ✅ 实现        | 根据数据集大小调整                       |
| 过拟合警告        | ✅ 实现        | 小数据集 + quality 预设                  |
| 自定义 preset    | ❌ Phase 2     | 用户自定义 preset 文件                   |
| Preset 模板导出  | ❌ Phase 2     | 导出当前配置为 preset                    |

---

## 12. 测试要点

| 测试场景              | 输入                         | 期望输出                          |
|---------------------|-----------------------------|---------------------------------|
| 使用 quick 预设      | `--preset quick`             | rank=16, lr=2e-4, steps=1000    |
| 使用 balanced 预设   | `--preset balanced`          | rank=32, lr=1e-4, steps=1500    |
| 使用 quality 预设    | `--preset quality`           | rank=64, lr=5e-5, steps=3000    |
| Preset + CLI 覆盖   | `--preset quick --rank 32`   | rank=32（覆盖 preset 默认值）     |
| 小数据集 + quality  | 30 张图 + `--preset quality` | 警告：过拟合风险                  |
| 自适应步数（大数据集）| 500 张图 + `--preset balanced` | steps=6000（自动计算）          |
| 禁用 preset         | `--preset none --rank 32 --lr 1e-4` | 使用手动配置，无 preset 默认值 |
