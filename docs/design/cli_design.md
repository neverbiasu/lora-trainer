# CLI 设计

## 1. 设计原则

### 1.0 本周 V0.2 目标

在保持“单入口、无子命令”不变的前提下，补齐：

- `--resume`
- `--export-only`
- 相关错误码与分发路径

### 1.1 单入口，无子命令
- 工具名为 `lora-trainer`，名称本身已表达训练用途
- 所有行为通过扁平参数控制，不引入子命令层级
- 参考 sd-scripts（`train_network.py`）、naifu、Diffusers 的训练入口模式
- 配置优先：复杂参数放在 config.yaml，CLI 用于快速覆盖

### 1.2 设计参考

| 参考来源 | 模式 | 采纳要点 |
| --- | --- | --- |
| sd-scripts `train_network.py` | 单脚本，参数通过 argparse 管理 | 扁平参数结构，`--output_dir`、`--learning_rate`、`--network_dim`（rank） |
| Diffusers `train_text_to_image_lora.py` | 单脚本，通过 `accelerate launch` 启动 | `--pretrained_model_name_or_path`、`--resolution`、`--mixed_precision`、`--rank` |
| naifu | 单入口 `train.py` | 配置驱动，CLI 面较小 |
| OneTrainer | 配置优先 + CLI 覆盖 | YAML 配置 + 可选 CLI 覆盖 |

本地参考代码：

- `docs/references/sd-scripts/train_network.py`
- `docs/references/sd-scripts/library/train_util.py`
- `docs/references/diffusers/examples/dreambooth/`

### 1.3 为什么不使用子命令

多数成熟训练工具采用单入口：

```bash
# sd-scripts
accelerate launch train_network.py --pretrained_model_name_or_path=... --output_dir=...

# Diffusers
accelerate launch train_text_to_image_lora.py --pretrained_model_name_or_path=... --output_dir=...

# naifu
python train.py --config config.yaml
```

子命令（`train`、`validate`、`export`）会增加不必要的间接层。
改用 `--validate-only`、`--export-only`、`--dry-run` 等模式开关更直接。

---

## 2. 用法

### 2.1 核心工作流

```bash
# Train with config file (primary usage)
lora-trainer --config config.yaml

# Train with inline args (quick usage)
lora-trainer --dataset ./data --base-model sd15

# Config + CLI overrides
lora-trainer --config config.yaml --rank 64 --learning-rate 2e-4

# Validate dataset only (no training)
lora-trainer --config config.yaml --validate-only

# Preview resolved config (no training)
lora-trainer --config config.yaml --dry-run

# Resume from checkpoint
lora-trainer --resume ./output/run_001

# Resume and override max steps
lora-trainer --resume ./output/run_001 --max-steps 3000

# Export from existing run (no training)
lora-trainer --resume ./output/run_001 --export-only
```

### 2.2 参数分组

#### 数据
```
--dataset PATH              Dataset directory (required unless --config or --resume)
--resolution INT            Training resolution (default: 512)
--cache-latents             Enable latent caching (default: auto)
--no-bucketing              Disable aspect-ratio bucketing
```

#### 模型
```
--base-model NAME           Base model: sd15 (required unless --config or --resume)
--model-path PATH           Custom model path (overrides --base-model)
--rank INT                  LoRA rank (default: 32)
--alpha FLOAT               LoRA alpha (default: 32)
```

#### 训练
```
--learning-rate FLOAT       Learning rate (default: 1e-4)
--lr-scheduler NAME         Scheduler: cosine / constant (default: cosine)
--batch-size INT            Batch size (default: 4)
--gradient-accumulation INT Gradient accumulation steps (default: 1)
--max-steps INT             Max training steps (default: auto)
--preset NAME               Preset: quick / balanced / quality (default: balanced)
--seed INT                  Random seed (default: 42)
```

#### 优化
```
--mixed-precision TYPE      fp16 / bf16 / fp32 (default: fp16)
--enable-xformers           Enable xformers memory optimization
--gradient-checkpointing    Enable gradient checkpointing (saves VRAM)
```

#### 输出
```
--output-dir PATH           Output directory (default: ./output)
--save-every-n-steps INT    Checkpoint frequency (default: 500)
--sample-every-n-steps INT  Sample frequency (default: 250)
--sample-prompts PATH       Sample prompt file
```

#### 模式开关
```
--config PATH               YAML configuration file
--resume PATH               Resume from run directory or checkpoint
--validate-only             Validate dataset and exit
--export-only               Export model from run and exit (requires --resume)
--dry-run                   Preview resolved config and exit
--verbose                   Verbose logging
```

---

## 3. 配置优先级

解析顺序（后者覆盖前者）：

1. **内置默认值**：在 `DEFAULTS` 中定义
2. **预设**：`--preset balanced` 应用预设参数
3. **配置文件**：`--config config.yaml`
4. **CLI 参数**：`--rank 64`

Example:
```bash
lora-trainer --config config.yaml --rank 64
# config.yaml has rank: 32 -> final rank = 64 (CLI wins)
```

---

## 4. 模式逻辑

```
if --validate-only:
    load config -> validate dataset -> print report -> exit

elif --export-only:
    requires --resume
    load run -> export model -> exit

elif --dry-run:
    load config -> resolve all defaults -> print preview -> exit

elif --resume:
    load run config snapshot -> load checkpoint -> continue training

else:
    load config -> validate dataset -> train -> export
```

### 4.1 分发契约（V0.2）

- `--resume` + 非 `--export-only`
  - 进入训练恢复路径：`Trainer.start(resume=...) -> train -> end`
- `--resume` + `--export-only`
  - 进入导出路径：`Trainer.start(resume=...) -> end`
- `--export-only` 且无 `--resume`
  - 直接报错 `E041`

默认流程为：**validate -> train -> export**（单命令完成全流程）。

---

## 5. 错误处理

| 场景 | 提示信息 | 建议 |
| --- | --- | --- |
| 数据集不存在 | `[E001] Dataset not found: ./my_data` | 检查路径 |
| 缺少 captions | `[E002] 20 images missing caption files` | 使用 `--validate-only` 查看详情 |
| 预计 OOM | `[E012] Estimated VRAM 12GB > available 8GB` | 使用 `--batch-size 2` 或 `--rank 16` |
| 配置冲突 | `[E042] cache-latents conflicts with augmentation` | 禁用其中一个选项 |
| 缺少 --dataset 或 --config | `[E041] Must provide --dataset or --config` | 至少提供其中一个 |
| --export-only 未配合 --resume | `[E041] --export-only requires --resume` | 添加 `--resume ./output/run_001` |
| resume 路径不存在 | `[E061] resume path not found` | 检查 run/checkpoint 路径 |
| resume 权重加载失败 | `[E063] failed to load LoRA weights` | 检查 checkpoint 完整性与版本 |

---

## 6. 帮助输出

```
$ lora-trainer --help

usage: lora-trainer [OPTIONS]

LoRA Trainer - Minimalist but not simplistic.

Data:
  --dataset PATH              Dataset directory
  --resolution INT            Training resolution (default: 512)
  --cache-latents             Enable latent caching
  --no-bucketing              Disable aspect-ratio bucketing

Model:
  --base-model NAME           Base model: sd15
  --model-path PATH           Custom model path
  --rank INT                  LoRA rank (default: 32)
  --alpha FLOAT               LoRA alpha (default: 32)

Training:
  --learning-rate FLOAT       Learning rate (default: 1e-4)
  --batch-size INT            Batch size (default: 4)
  --max-steps INT             Max training steps (default: auto)
  --preset NAME               Preset: quick / balanced / quality
  --seed INT                  Random seed (default: 42)

Output:
  --output-dir PATH           Output directory (default: ./output)
  --save-every-n-steps INT    Checkpoint frequency (default: 500)
  --sample-every-n-steps INT  Sample frequency (default: 250)

Mode:
  --config PATH               YAML configuration file
  --resume PATH               Resume from run directory
  --validate-only             Validate dataset and exit
  --export-only               Export model and exit (requires --resume)
  --dry-run                   Preview config and exit
  --verbose                   Verbose logging

Examples:
  lora-trainer --config config.yaml
  lora-trainer --dataset ./data --base-model sd15
  lora-trainer --config config.yaml --validate-only
  lora-trainer --resume ./output/run_001
```

---

## 7. Dry-run 输出

```
$ lora-trainer --dataset ./data --base-model sd15 --dry-run

Dry-run: resolved configuration

Dataset:
  Path: ./data
  Images: 150
  Validation: passed

Model:
  Base: sd15
  Rank: 32, Alpha: 32

Training:
  LR: 1e-4 (cosine)
  Batch: 4
  Steps: 1500 (auto)
  Preset: balanced

VRAM: ~7.5 GB (safe for 8GB+)

Output:
  Dir: ./output/run_002
  Checkpoints: every 500 steps
  Samples: every 250 steps

Config valid. Remove --dry-run to start training.
```

---

## 8. 实现说明

### 8.1 参数解析（argparse，无子解析器）

完整实现见 `src/lora_trainer/cli.py`。

关键设计点：
- 单一 `ArgumentParser`，不使用 subparsers
- 参数通过 `add_argument_group()` 进行分组
- 模式开关（`--validate-only`、`--export-only`、`--dry-run`）使用 `store_true`
- `_validate_args()` 在分发前检查参数组合

### 8.2 配置合并

```python
def resolve_config(args):
    config = DEFAULTS.copy()
    if args.preset:
        config = deep_merge(config, get_preset(args.preset))
    if args.config:
        config = deep_merge(config, load_yaml(args.config))
    cli_overrides = extract_cli_overrides(args)
    config = deep_merge(config, cli_overrides)
    return config
```

---

## 9. MVP 范围

| 功能 | 状态 | 备注 |
| --- | --- | --- |
| 扁平参数（无子命令） | MVP | 单入口 |
| --config YAML | MVP | 主配置方式 |
| --validate-only | MVP | 数据集校验 |
| --export-only | MVP | 从 run 导出 |
| --resume | MVP | 仅权重恢复（不含优化器状态） |
| --dry-run | MVP | 配置预览 |
| 交互式配置向导 | Phase 2 | MVP 仅 CLI + YAML |
| GUI | Phase 2 | CLI 优先 |
| Multi-GPU | Phase 2 | MVP 先单卡 |

---

## 10. 测试用例

| 场景 | 输入 | 期望 |
| --- | --- | --- |
| 最小内联参数 | `--dataset ./data --base-model sd15` | 使用默认值训练 |
| 配置文件 | `--config config.yaml` | 加载 YAML |
| CLI 覆盖 | `--config config.yaml --rank 64` | rank=64 生效 |
| Dry-run | `--dry-run --config config.yaml` | 输出预览，不训练 |
| 仅校验 | `--validate-only --dataset ./data` | 输出报告并退出 |
| 恢复训练 | `--resume ./output/run_001` | 从 checkpoint 继续 |
| 仅导出 | `--resume ./output/run_001 --export-only` | 导出后退出 |
| 缺少参数 | (无 --dataset、无 --config、无 --resume) | 报错 E041 并给出建议 |
| 无恢复路径却导出 | `--export-only` | 报错：需要 --resume |
