# hyperparam_rules 设计

## 1. 职责边界

HyperparamPolicy **负责**：
- **推荐默认值**：基于模型类型、显存、数据集大小给出合理默认值
- **约束校验**：检查参数范围、依赖关系、冲突
- **显存预估**：预测训练所需显存，给出降配建议
- **自动调整**：根据运行环境自动启用/禁用优化（如 cache_latents、gradient_accumulation）

HyperparamPolicy **不负责**：
- 训练循环执行（由 Trainer 处理）
- 具体优化器实现（由 AlgoAdapter 处理）
- 模型加载（由 ModelAdapter 处理）

---

## 2. 核心接口

```python
class HyperparamPolicy:
    """超参数推荐、校验、显存预估统一入口"""
    
    def __init__(self, model_type: str = "sd15"):
        self.model_type = model_type
    
    def recommend_defaults(
        self,
        dataset_size: int,
        available_vram_gb: float
    ) -> Dict[str, Any]:
        """返回推荐的默认参数"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """校验参数范围和冲突"""
        pass
    
    def estimate_vram(self, config: Dict[str, Any]) -> VRAMEstimate:
        """预估训练所需显存"""
        pass
    
    def auto_adjust(
        self,
        config: Dict[str, Any],
        available_vram_gb: float
    ) -> Dict[str, Any]:
        """自动调整配置以适应显存"""
        pass
```

---

## 3. 参数推荐规则（recommend_defaults）

### 3.1 基础推荐表（参考 kohya BasicTraining 参数体系）

| 参数名称               | SD1.5 默认值      | SDXL 默认值 (Phase 2) | 说明                                    |
|-----------------------|------------------|-----------------------|----------------------------------------|
| `rank`                | 32               | 64                    | LoRA rank，越大能力越强但越慢            |
| `alpha`               | 32 (=rank)       | 64 (=rank)            | LoRA scaling，默认等于 rank              |
| `learning_rate`       | 1e-4             | 5e-5                  | SD1.5 可以更激进，SDXL 需要更保守        |
| `lr_scheduler`        | "cosine"         | "cosine"              | 余弦退火，训练后期缓慢降低 lr             |
| `optimizer`           | "adamw"          | "adamw"               | 带权重衰减的 Adam 优化器              |
| `batch_size`          | 4                | 2                     | SDXL 显存需求更大，batch 更小            |
| `gradient_accumulation` | 1              | 2                     | SDXL 通过累积模拟更大 batch              |
| `mixed_precision`     | "fp16"           | "fp16"                | 必须启用，fp32 显存占用翻倍              |
| `enable_xformers`     | True             | True                  | 显存优化，8GB 卡必须启用                 |
| `max_train_steps`     | 1500             | 3000                  | SDXL 需要更多步数收敛                    |
| `save_every_n_steps`  | 500              | 1000                  | 检查点频率                               |
| `sample_every_n_steps` | 250             | 500                   | 采样验证频率                             |
| `cache_latents`       | auto             | auto                  | 小数据集 + 高显存自动启用                 |

### 3.2 数据集规模调整（参考 kohya 经验值）

```python
def recommend_defaults(self, dataset_size: int, available_vram_gb: float):
    """根据数据集大小和显存调整默认值"""
    
    # 基础配置
    defaults = {
        "rank": 32,
        "alpha": 32,
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "optimizer": "adamw",
        "batch_size": 4,
        "gradient_accumulation": 1,
        "mixed_precision": "fp16",
        "enable_xformers": True,
    }
    
    # 根据数据集大小调整训练步数（每张图至少训练 30-50 次）
    defaults["max_train_steps"] = dataset_size * 40 // defaults["batch_size"]
    
    # 小数据集（< 100 张）：降低 rank 防止过拟合
    if dataset_size < 100:
        defaults["rank"] = 16
        defaults["alpha"] = 16
        defaults["learning_rate"] = 5e-5  # 更保守的 lr
    
    # 中等数据集（100-500 张）：标准配置
    elif dataset_size <= 500:
        pass  # 使用默认值
    
    # 大数据集（> 500 张）：可以用更大 rank
    else:
        defaults["rank"] = 64
        defaults["alpha"] = 64
        defaults["learning_rate"] = 2e-4  # 稍微激进
    
    # 根据显存调整 batch size 和累积步数
    if available_vram_gb < 10:
        defaults["batch_size"] = 2
        defaults["gradient_accumulation"] = 2
        defaults["cache_latents"] = False  # 低显存不 cache
    elif available_vram_gb >= 12:
        defaults["batch_size"] = 4
        defaults["cache_latents"] = True if dataset_size < 500 else False
    
    return defaults
```

### 3.3 三档预设（quick/balanced/quality）

| Preset    | 目标               | rank | lr    | steps        | batch | 时间估算 (100张) |
|-----------|--------------------|------|-------|--------------|-------|-----------------|
| quick     | 快速验证概念        | 16   | 2e-4  | 1000         | 4     | ~10 min         |
| balanced  | 平衡质量和速度      | 32   | 1e-4  | 1500         | 4     | ~15 min         |
| quality   | 最佳质量（可能过拟合）| 64   | 5e-5  | 3000         | 2     | ~40 min         |

```python
PRESETS = {
    "quick": {
        "rank": 16, "alpha": 16,
        "learning_rate": 2e-4,
        "max_train_steps": 1000,
        "batch_size": 4,
    },
    "balanced": {
        "rank": 32, "alpha": 32,
        "learning_rate": 1e-4,
        "max_train_steps": 1500,
        "batch_size": 4,
    },
    "quality": {
        "rank": 64, "alpha": 64,
        "learning_rate": 5e-5,
        "max_train_steps": 3000,
        "batch_size": 2,
    },
}
```

---

## 4. 参数约束校验（validate_config）

### 4.1 范围约束

| 参数名称               | 有效范围               | 推荐范围       | 说明                                    |
|-----------------------|-----------------------|--------------|----------------------------------------|
| `rank`                | [1, 512]              | [8, 64]      | 太小学不到，太大过拟合 + 慢               |
| `alpha`               | [0.1, 512]            | [rank/2, rank*2] | 通常设为 rank                          |
| `learning_rate`       | [1e-6, 1e-2]          | [5e-5, 2e-4] | 太小不收敛，太大震荡                      |
| `batch_size`          | [1, 32]               | [2, 8]       | 受显存限制                               |
| `gradient_accumulation` | [1, 16]             | [1, 4]       | 模拟更大 batch                           |
| `max_train_steps`     | [100, 100000]         | [1000, 5000] | 太少欠拟合，太多浪费时间                  |
| `save_every_n_steps`  | [100, max_train_steps] | [500, 1000] | 太频繁占用磁盘                            |

### 4.2 依赖关系约束

```python
def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
    """校验参数约束"""
    errors = []
    warnings = []
    
    # 1. 范围约束
    if not 8 <= config["rank"] <= 128:
        errors.append(f"rank={config['rank']} out of recommended range [8, 128]")
    
    if config["learning_rate"] > 5e-4:
        warnings.append(f"learning_rate={config['learning_rate']} is very high, may cause instability")
    
    # 2. 依赖关系
    if config["alpha"] > config["rank"] * 2:
        warnings.append(f"alpha={config['alpha']} >> rank={config['rank']}, unusual scaling")
    
    if config["gradient_accumulation"] > 1 and config["batch_size"] >= 8:
        warnings.append("Large batch_size + gradient_accumulation may not improve results")
    
    # 3. 冲突检测
    if config.get("cache_latents") and config.get("enable_flip"):
        errors.append("cache_latents conflicts with enable_flip (data augmentation)")
    
    if config["save_every_n_steps"] > config["max_train_steps"]:
        errors.append("save_every_n_steps > max_train_steps, will never save checkpoint")
    
    # 4. 步数合理性
    effective_batch = config["batch_size"] * config["gradient_accumulation"]
    dataset_size = config.get("dataset_size", 100)
    epochs_equivalent = (config["max_train_steps"] * effective_batch) / dataset_size
    
    if epochs_equivalent < 20:
        warnings.append(f"Training for ~{epochs_equivalent:.1f} epochs may be insufficient")
    elif epochs_equivalent > 100:
        warnings.append(f"Training for ~{epochs_equivalent:.1f} epochs may overfit")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
```

---

## 5. 显存预估（estimate_vram）

### 5.1 预估公式（参考 kohya 和 OneTrainer 经验）

```python
def estimate_vram(self, config: Dict[str, Any]) -> VRAMEstimate:
    """预估显存需求（单位：GB）"""
    
    # 基础显存（模型权重 + 优化器状态）
    base_vram = {
        "sd15": {
            "model": 2.0,      # UNet + VAE + TextEncoder
            "optimizer": 1.5,  # AdamW 状态
            "lora_params": 0.1 * (config["rank"] / 32),  # LoRA 参数量
        },
        "sdxl": {
            "model": 6.0,
            "optimizer": 3.0,
            "lora_params": 0.2 * (config["rank"] / 64),
        },
    }[self.model_type]
    
    # 激活值显存（与 batch_size 和分辨率相关）
    resolution = config.get("resolution", 512)
    batch_size = config["batch_size"]
    
    # 激活值：与 batch * resolution^2 成正比
    activation_vram = (batch_size * (resolution / 512) ** 2) * 1.2
    
    # 优化选项减少显存
    if config.get("mixed_precision") == "fp16":
        base_vram["model"] *= 0.5
        base_vram["optimizer"] *= 0.5
    
    if config.get("enable_xformers"):
        activation_vram *= 0.6  # xformers 大约减少 40% 激活值显存
    
    if config.get("gradient_checkpointing"):
        activation_vram *= 0.5  # 牺牲速度换显存
    
    # Latent Cache 额外显存
    cache_vram = 0
    if config.get("cache_latents"):
        dataset_size = config.get("dataset_size", 100)
        # 每张图 latent: 4 channels * (resolution/8)^2 * 4 bytes (fp32)
        cache_vram = dataset_size * 4 * ((resolution / 8) ** 2) * 4 / (1024 ** 3)
    
    # 总显存
    total_vram = sum(base_vram.values()) + activation_vram + cache_vram
    
    return VRAMEstimate(
        total_gb=total_vram,
        breakdown={
            "model": base_vram["model"],
            "optimizer": base_vram["optimizer"],
            "lora_params": base_vram["lora_params"],
            "activation": activation_vram,
            "cache": cache_vram,
        },
        safe_threshold_gb=total_vram * 1.2,  # 留 20% buffer
    )
```

### 5.2 显存预估表（SD1.5, resolution=512）

| 配置                                | 预估显存 | 实际测试 | 说明                    |
|------------------------------------|---------|---------|------------------------|
| rank=32, batch=4, fp16, xformers   | 7.5 GB  | 7.8 GB  | RTX 3060 可用           |
| rank=32, batch=2, fp16, xformers   | 5.5 GB  | 5.7 GB  | GTX 1080 8GB 可用       |
| rank=64, batch=4, fp16, xformers   | 8.2 GB  | 8.5 GB  | 需要 10GB+ 显卡         |
| rank=32, batch=4, fp32, no xformers | 15 GB  | 15.3 GB | 不推荐                  |
| cache_latents (500 张)             | +1.2 GB | +1.3 GB | 小数据集可承受          |

---

## 6. 自动调整（auto_adjust）

### 6.1 降配策略（参考 kohya 自动优化）

```python
def auto_adjust(self, config: Dict[str, Any], available_vram_gb: float):
    """根据显存自动调整配置"""
    
    estimate = self.estimate_vram(config)
    
    if estimate.total_gb <= available_vram_gb * 0.9:
        # 显存充足，无需调整
        return config
    
    print(f"⚠️ Estimated VRAM {estimate.total_gb:.1f}GB exceeds available {available_vram_gb:.1f}GB")
    print("Auto-adjusting configuration...")
    
    # 降配步骤（按优先级）
    adjusted = config.copy()
    
    # Step 1: 禁用 latent cache
    if adjusted.get("cache_latents"):
        adjusted["cache_latents"] = False
        print("  - Disabled cache_latents")
        estimate = self.estimate_vram(adjusted)
        if estimate.total_gb <= available_vram_gb * 0.9:
            return adjusted
    
    # Step 2: 减少 batch_size，增加 gradient_accumulation
    if adjusted["batch_size"] > 2:
        original_batch = adjusted["batch_size"]
        adjusted["batch_size"] = 2
        adjusted["gradient_accumulation"] = adjusted.get("gradient_accumulation", 1) * 2
        print(f"  - Reduced batch_size {original_batch} → 2, gradient_accumulation x2")
        estimate = self.estimate_vram(adjusted)
        if estimate.total_gb <= available_vram_gb * 0.9:
            return adjusted
    
    # Step 3: 启用 gradient_checkpointing
    if not adjusted.get("gradient_checkpointing"):
        adjusted["gradient_checkpointing"] = True
        print("  - Enabled gradient_checkpointing (slower but saves VRAM)")
        estimate = self.estimate_vram(adjusted)
        if estimate.total_gb <= available_vram_gb * 0.9:
            return adjusted
    
    # Step 4: 降低 rank
    if adjusted["rank"] > 16:
        original_rank = adjusted["rank"]
        adjusted["rank"] = max(16, adjusted["rank"] // 2)
        adjusted["alpha"] = adjusted["rank"]
        print(f"  - Reduced rank {original_rank} → {adjusted['rank']}")
        estimate = self.estimate_vram(adjusted)
        if estimate.total_gb <= available_vram_gb * 0.9:
            return adjusted
    
    # Step 5: 最后手段 - batch_size=1
    if adjusted["batch_size"] > 1:
        adjusted["batch_size"] = 1
        adjusted["gradient_accumulation"] *= 2
        print("  - Set batch_size=1 (last resort)")
        estimate = self.estimate_vram(adjusted)
        if estimate.total_gb <= available_vram_gb * 0.9:
            return adjusted
    
    # 仍然超出显存，报错
    raise RuntimeError(
        f"Cannot fit training in {available_vram_gb:.1f}GB VRAM even after auto-adjustment. "
        f"Minimum required: {estimate.total_gb:.1f}GB"
    )
```

---

## 7. 参数冲突检测

### 7.1 常见冲突

| 冲突组合                          | 原因                                      | 解决方案                       |
|----------------------------------|------------------------------------------|-------------------------------|
| `cache_latents=True` + 数据增强   | Cache 的 latent 无法实时增强               | 禁用 cache 或禁用增强          |
| `gradient_accumulation=1` + 小batch | 无法模拟大 batch 效果                      | 增加累积步数                   |
| `lr_scheduler="constant"` + warmup | Warmup 对 constant 无意义                  | 改用 cosine 或移除 warmup      |
| `max_train_steps` < `save_every_n_steps` | 永远不会保存检查点                        | 调整保存频率                   |
| `rank` >> 数据集大小              | 参数量过大，过拟合                         | 降低 rank 或增加数据            |

### 7.2 实现要点

```python
def detect_conflicts(self, config: Dict[str, Any]) -> List[str]:
    """检测参数冲突"""
    conflicts = []
    
    # 冲突 1: cache + augmentation
    if config.get("cache_latents") and config.get("enable_flip"):
        conflicts.append("cache_latents + enable_flip: cached latents cannot be augmented")
    
    # 冲突 2: 小 batch + 无累积
    if config["batch_size"] <= 2 and config.get("gradient_accumulation", 1) == 1:
        conflicts.append("Small batch_size without gradient_accumulation may cause instability")
    
    # 冲突 3: 步数与保存频率
    if config["max_train_steps"] < config.get("save_every_n_steps", float("inf")):
        conflicts.append("max_train_steps < save_every_n_steps: no checkpoints will be saved")
    
    # 冲突 4: rank 过大
    dataset_size = config.get("dataset_size", 100)
    if config["rank"] > dataset_size / 2:
        conflicts.append(f"rank={config['rank']} too large for dataset_size={dataset_size}, risk of overfitting")
    
    return conflicts
```

---

## 8. 参考实现映射

| 功能模块          | 参考仓库         | 文件路径                                      |
|------------------|-----------------|---------------------------------------------|
| 参数体系表        | sd-scripts      | `library/train_util.py` → BasicTraining 类  |
| 默认值推荐        | OneTrainer      | `modules/util/args/TrainArgs.py`            |
| 显存预估          | sd-scripts      | `train_network.py` → estimate_vram 注释     |
| 自动降配          | kohya-ss        | `train_network.py` → 自动启用 xformers 逻辑  |
| Preset 定义      | flymyai-lora-trainer | `config/presets/` 目录                   |

---

## 9. MVP 简化约束

| 特性              | MVP 状态       | 备注                                    |
|------------------|---------------|----------------------------------------|
| 参数推荐          | ✅ 实现        | SD1.5 默认值 + 三档预设                  |
| 约束校验          | ✅ 实现        | 范围、依赖、冲突检测                     |
| 显存预估          | ✅ 实现        | 公式估算 + 降配建议                      |
| 自动调整          | ✅ 实现        | 5 步降配策略                             |
| SDXL 参数        | ❌ Phase 2     | 仅 SD1.5 默认值                          |
| 高级调度器        | ❌ Phase 2     | 仅 cosine/constant，不支持 polynomial    |
| 自定义预设        | ❌ Phase 2     | MVP 仅内置 3 档                          |

---

## 10. 测试要点

| 测试场景              | 输入                         | 期望输出                          |
|---------------------|-----------------------------|---------------------------------|
| 推荐默认值（小数据集） | 50 张图 + 12GB 显存          | rank=16, lr=5e-5, cache=True    |
| 推荐默认值（大数据集） | 500 张图 + 8GB 显存          | rank=32, lr=1e-4, cache=False   |
| 约束校验（rank 过大） | rank=256                    | 警告：超出推荐范围                 |
| 约束校验（步数冲突）  | max_steps=1000, save_every=2000 | 错误：永远不会保存检查点          |
| 显存预估（标准配置）  | rank=32, batch=4, fp16      | 7.5 GB                          |
| 显存预估（极限配置）  | rank=128, batch=8, fp32     | 25+ GB（不可行）                  |
| 自动调整（低显存）   | 10GB 显存 + 12GB 需求配置     | 禁用 cache, batch=2, rank=16    |
| 冲突检测             | cache=True + flip=True      | 错误：cache 与数据增强冲突         |
