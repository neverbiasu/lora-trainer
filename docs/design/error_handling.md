# error_handling 设计

## 1. 错误处理原则

### 1.1 用户友好性
- **避免技术术语**：用户可能不懂 CUDA、OOM、RuntimeError
- **提供具体建议**：不只说"失败了"，要说"怎么修复"
- **分级提示**：ERROR（致命）、WARNING（非致命）、INFO（提示）

### 1.2 可调试性
- **保留堆栈跟踪**：开发模式保留完整错误信息
- **记录上下文**：捕获错误时记录配置、数据集信息、系统状态
- **错误代码**：为常见错误分配唯一代码（如 `E001: Dataset not found`）

---

## 2. 错误分类体系

### 2.1 按阶段分类

| 阶段              | 典型错误                          | 处理策略                    |
|------------------|----------------------------------|-----------------------------|
| **启动阶段**      | 配置文件缺失、参数冲突             | 启动前校验，快速失败          |
| **数据集加载**    | 数据集不存在、缺失 caption         | validate 命令提前发现         |
| **模型加载**      | 模型路径错误、版本不兼容           | 提供下载链接或建议            |
| **训练初始化**    | 显存不足、CUDA 不可用              | 自动降配或建议配置            |
| **训练循环**      | OOM、NaN loss、梯度爆炸            | 降配、early stop、保存检查点  |
| **导出阶段**      | 权重格式错误、元数据缺失           | 校验权重完整性                |

---

## 3. 常见错误处理

### 3.1 数据集相关错误

#### E001: 数据集不存在
```python
try:
    dataset_path = Path(config["data"]["dataset_path"])
    if not dataset_path.exists():
        raise DatasetNotFoundError(dataset_path)
except DatasetNotFoundError as e:
    print(f"❌ [E001] Dataset not found: {e.path}")
    print(f"   Please check the path and try again.")
    print(f"   Hint: Use absolute path or ensure relative path is correct.")
    sys.exit(1)
```

#### E002: 缺失 caption 文件
```python
validation_result = validate_dataset(dataset_path)
if not validation_result.valid:
    print(f"❌ [E002] Dataset validation failed:")
    for err in validation_result.errors[:5]:  # 只显示前 5 个
        if err.error_type == "missing_caption":
            print(f"  - {err.file_path}: Missing .txt file")
    
    print("\n💡 Suggestions:")
    print("  1. Run 'lora-trainer validate --dataset ./data' for full report")
    print("  2. Ensure every .png/.jpg has a corresponding .txt file")
    print("  3. Use caption generation tools if needed")
    sys.exit(1)
```

#### E003: 数据集过小
```python
if dataset_size < 10:
    print(f"❌ [E003] Dataset too small: {dataset_size} images")
    print(f"   Minimum required: 10 images")
    print(f"   Recommended: 20+ images for quality results")
    sys.exit(1)
elif dataset_size < 20:
    print(f"⚠️ [W001] Small dataset: {dataset_size} images")
    print(f"   Results may be suboptimal. Consider adding more images.")
```

---

### 3.2 显存相关错误

#### E010: CUDA 不可用
```python
if not torch.cuda.is_available():
    print("❌ [E010] CUDA not available")
    print("   LoRA training requires a NVIDIA GPU with CUDA support.")
    print("\n💡 Troubleshooting:")
    print("  1. Check if GPU is detected: nvidia-smi")
    print("  2. Ensure PyTorch with CUDA is installed:")
    print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)
```

#### E011: 显存不足（OOM）
```python
try:
    loss = train_step(batch, model, optimizer)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"❌ [E011] Out of memory (OOM)")
        print(f"   Current config:")
        print(f"     - Rank: {config['lora']['rank']}")
        print(f"     - Batch Size: {config['training']['batch_size']}")
        print(f"     - Mixed Precision: {config['training']['mixed_precision']}")
        print("\n💡 Try these fixes:")
        print("  1. Reduce batch size: --batch-size 2")
        print("  2. Reduce rank: --rank 16")
        print("  3. Enable gradient checkpointing: --gradient-checkpointing")
        print("  4. Use auto-adjust: --auto-adjust-vram")
        
        # 自动降配（如果启用）
        if config.get("auto_adjust_vram"):
            print("\n🔧 Auto-adjusting configuration...")
            adjusted_config = hyperparam_policy.auto_adjust(config, available_vram_gb)
            print("✓ Configuration adjusted. Retrying training...")
            # 重试训练
        else:
            sys.exit(1)
    else:
        raise
```

#### E012: 显存不足（预估阶段）
```python
vram_estimate = hyperparam_policy.estimate_vram(config)
available_vram = get_available_vram_gb()

if vram_estimate.total_gb > available_vram * 0.9:
    print(f"❌ [E012] Insufficient VRAM")
    print(f"   Estimated: {vram_estimate.total_gb:.1f} GB")
    print(f"   Available: {available_vram:.1f} GB")
    print(f"\n   Breakdown:")
    for k, v in vram_estimate.breakdown.items():
        print(f"     - {k}: {v:.1f} GB")
    
    print("\n💡 Try running with auto-adjustment:")
    print("  lora-trainer train --config config.yaml --auto-adjust-vram")
    sys.exit(1)
```

---

### 3.3 模型相关错误

#### E020: 模型路径错误
```python
try:
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
except OSError as e:
    if "does not appear to be a file" in str(e):
        print(f"❌ [E020] Model not found: {model_path}")
        print("\n💡 Available options:")
        print("  1. Use built-in model: --base-model sd15")
        print("  2. Download from HuggingFace:")
        print("     https://huggingface.co/runwayml/stable-diffusion-v1-5")
        print("  3. Specify local path: --model-path /path/to/model")
        sys.exit(1)
```

#### E021: 模型版本不兼容
```python
try:
    # 加载模型
    unet = load_unet(model_path)
except Exception as e:
    if "version" in str(e).lower() or "incompatible" in str(e).lower():
        print(f"❌ [E021] Model version incompatible")
        print(f"   This trainer supports SD1.5 format only (MVP).")
        print("\n💡 Suggestions:")
        print("  1. Use SD1.5 models: runwayml/stable-diffusion-v1-5")
        print("  2. Convert SDXL models (Phase 2 feature)")
        sys.exit(1)
```

---

### 3.4 训练相关错误

#### E030: NaN loss
```python
def check_loss_valid(loss: float, step: int):
    """检查 loss 是否有效"""
    
    if math.isnan(loss):
        print(f"❌ [E030] NaN loss detected at step {step}")
        print("\n💡 Possible causes:")
        print("  1. Learning rate too high → Try --learning-rate 5e-5")
        print("  2. Mixed precision instability → Try --mixed-precision fp32")
        print("  3. Corrupted image in dataset → Run validate command")
        print("\n   Saving emergency checkpoint...")
        save_checkpoint(step, prefix="nan_loss_emergency")
        sys.exit(1)
    
    if math.isinf(loss):
        print(f"❌ [E031] Infinite loss at step {step}")
        print("   This usually indicates gradient explosion.")
        print("   Try reducing learning rate or enabling gradient clipping.")
        sys.exit(1)
```

#### E032: 梯度爆炸
```python
def clip_gradients(model, max_grad_norm=1.0):
    """梯度裁剪"""
    
    total_norm = torch.nn.utils.clip_grad_norm_(
        get_trainable_params(model),
        max_grad_norm,
    )
    
    if total_norm > max_grad_norm * 10:
        print(f"⚠️ [W010] Large gradient norm: {total_norm:.2f}")
        print(f"   Gradient clipped to {max_grad_norm}")
        print("   Consider reducing learning rate if this persists.")
```

---

### 3.5 配置相关错误

#### E040: 配置文件格式错误
```python
try:
    with open(config_path) as f:
        config = yaml.safe_load(f)
except yaml.YAMLError as e:
    print(f"❌ [E040] Invalid YAML syntax in {config_path}")
    print(f"   Error at line {e.problem_mark.line + 1}:")
    print(f"   {e.problem}")
    print("\n💡 Check YAML syntax:")
    print("  - Correct indentation (2 spaces)")
    print("  - No tabs")
    print("  - Quoted strings with special characters")
    sys.exit(1)
```

#### E041: 缺失必填参数
```python
def validate_required_fields(config: Dict):
    """校验必填字段"""
    
    required_fields = [
        ("model", "base_model"),
        ("data", "dataset_path"),
    ]
    
    missing = []
    for section, field in required_fields:
        if section not in config or field not in config[section]:
            missing.append(f"{section}.{field}")
    
    if missing:
        print(f"❌ [E041] Missing required fields in config:")
        for field in missing:
            print(f"  - {field}")
        print("\n💡 Example config:")
        print("  model:")
        print("    base_model: sd15")
        print("  data:")
        print("    dataset_path: ./my_data")
        sys.exit(1)
```

#### E042: 参数冲突
```python
conflicts = hyperparam_policy.detect_conflicts(config)
if conflicts:
    print(f"❌ [E042] Configuration conflicts detected:")
    for conflict in conflicts:
        print(f"  - {conflict}")
    print("\n💡 Fix conflicts and retry.")
    sys.exit(1)
```

---

## 4. 警告处理

### 4.1 非致命警告

```python
def show_warnings(warnings: List[str]):
    """显示警告但不退出"""
    
    if warnings:
        print("⚠️ Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nThese are non-critical. Training can proceed.")
        
        # 可选：询问用户是否继续
        if input("\nContinue? [Y/n]: ").lower() == "n":
            sys.exit(0)
```

### 4.2 常见警告

| 警告代码 | 触发条件                          | 建议                                    |
|---------|----------------------------------|-----------------------------------------|
| W001    | 数据集 < 20 张                    | 增加图片数量                             |
| W002    | 图片分辨率过大（> 2048）           | 调整分辨率或启用 VAE tiling              |
| W003    | rank 过大（> 数据集大小 / 2）      | 降低 rank 防止过拟合                     |
| W004    | learning_rate > 5e-4              | 可能不稳定，考虑降低                      |
| W005    | 训练步数 > 数据集 * 100            | 可能过拟合                               |
| W010    | 梯度范数过大                       | 梯度爆炸风险                             |

---

## 5. 错误恢复机制

### 5.1 自动保存检查点

```python
def train_with_error_recovery(trainer, dataloader):
    """带错误恢复的训练循环"""
    
    try:
        for step, batch in enumerate(dataloader):
            loss = trainer.train_step(batch)
            
            # 定期保存检查点
            if step % 500 == 0:
                trainer.save_checkpoint(step)
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ OOM at step {step}")
            print("Saving emergency checkpoint...")
            trainer.save_checkpoint(step, prefix="oom_emergency")
            
            # 尝试自动降配
            if trainer.auto_adjust_enabled:
                print("Retrying with reduced batch size...")
                trainer.config["batch_size"] = max(1, trainer.config["batch_size"] // 2)
                torch.cuda.empty_cache()
                # 递归重试（限制次数）
        else:
            raise
    
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(step, prefix="interrupted")
        sys.exit(0)
```

### 5.2 断点续训

```python
def resume_from_checkpoint(run_dir: Path, checkpoint_name: str):
    """从检查点恢复"""
    
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    
    if not checkpoint_path.exists():
        print(f"❌ [E050] Checkpoint not found: {checkpoint_path}")
        print("\n💡 Available checkpoints:")
        checkpoints = list((run_dir / "checkpoints").glob("*.safetensors"))
        for cp in checkpoints:
            print(f"  - {cp.name}")
        sys.exit(1)
    
    try:
        state_dict = load_file(checkpoint_path)
        print(f"✓ Loaded checkpoint: {checkpoint_name}")
        return state_dict
    except Exception as e:
        print(f"❌ [E051] Failed to load checkpoint: {e}")
        print("   The checkpoint file may be corrupted.")
        sys.exit(1)
```

---

## 6. 错误日志

### 6.1 日志级别

```python
import logging

def setup_logging(run_dir: Path, verbose: bool = False):
    """配置日志系统"""
    
    log_file = run_dir / "logs" / "train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 文件日志（详细）
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台日志（简洁）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 根日志
    logger = logging.getLogger("lora_trainer")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### 6.2 错误日志记录

```python
logger = logging.getLogger("lora_trainer")

try:
    train_step(batch)
except Exception as e:
    logger.error(f"Training failed at step {step}", exc_info=True)
    logger.debug(f"Config: {config}")
    logger.debug(f"Batch info: {batch.keys()}")
    raise
```

---

## 7. 用户反馈示例

### 7.1 成功提示

```
✓ Dataset validation passed: 150 images
✓ Models loaded: SD1.5
✓ LoRA injected: 16 modules, 786K trainable params
✓ Estimated VRAM: 7.5 GB (safe for 8GB GPU)

🚀 Starting training...
```

### 7.2 错误提示

```
❌ [E002] Dataset validation failed: 5 missing caption files
  - data/image_001.png: Missing .txt file
  - data/image_005.png: Missing .txt file
  ... (3 more)

💡 Suggestions:
  1. Run 'lora-trainer validate --dataset ./data --verbose' for full report
  2. Create .txt files with the same name as each image
  3. Use BLIP or other caption tools to generate captions

For help: https://docs.lora-trainer.dev/errors/E002
```

---

## 8. 参考实现映射

| 功能模块          | 参考仓库         | 文件路径                                      |
|------------------|-----------------|---------------------------------------------|
| 错误提示          | flymyai-lora-trainer | 友好的错误信息设计                          |
| 自动降配          | kohya-ss        | `train_network.py` → OOM 处理               |
| 断点续训          | OneTrainer      | `modules/trainer/GenericTrainer.py`         |
| 日志系统          | sd-scripts      | `library/train_util.py` → setup_logging     |

---

## 9. MVP 简化约束

| 特性              | MVP 状态       | 备注                                    |
|------------------|---------------|----------------------------------------|
| 常见错误处理      | ✅ 实现        | 数据集、显存、配置错误                    |
| 友好提示          | ✅ 实现        | 用户友好的错误信息和建议                  |
| 自动降配          | ✅ 实现        | OOM 时自动调整配置                       |
| 断点续训          | ✅ 实现        | 从检查点恢复                             |
| 详细日志          | ✅ 实现        | 文件 + 控制台日志                        |
| 错误码文档        | ❌ Phase 2     | 在线错误码查询系统                        |
| 自动问题诊断      | ❌ Phase 2     | AI 诊断工具                              |

---

## 10. 测试要点

| 测试场景              | 触发条件                         | 期望输出                          |
|---------------------|-----------------------------|---------------------------------|
| 数据集不存在          | `--dataset /nonexist`        | E001 错误 + 路径检查建议          |
| 缺失 caption         | 数据集中 5 张图无 .txt           | E002 错误 + validate 命令建议     |
| CUDA 不可用          | 无 GPU 环境                     | E010 错误 + 安装 CUDA 指引        |
| 显存不足（预估）      | 配置需求 > 可用显存              | E012 错误 + 降配建议              |
| OOM（训练中）        | 训练时显存溢出                   | E011 错误 + 自动降配重试          |
| NaN loss            | loss 计算结果为 NaN              | E030 错误 + 可能原因和修复建议     |
| 配置文件错误          | YAML 语法错误                   | E040 错误 + 语法检查提示          |
| 参数冲突              | cache + flip 同时启用            | E042 错误 + 冲突说明              |
| 小数据集警告          | 15 张图训练                     | W001 警告 + 继续/取消选择         |
| 用户中断训练          | Ctrl+C                          | 保存检查点 + 优雅退出              |
