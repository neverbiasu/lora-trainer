# config_manager 设计（mvp v0.1）

## 1. 职责

| 项 | 说明 |
| --- | --- |
| 配置加载 | 解析 yaml 配置文件 |
| 版本管理 | 处理 config_version 字段，支持迁移 |
| 默认值填充 | 合并用户配置与默认配置 |
| 校验 | 检查必需字段、类型、范围 |

## 2. 输入/输出

| 类型 | 内容 |
| --- | --- |
| 输入 | config.yaml 路径 / cli 覆盖参数 |
| 输出 | resolved_config（完整配置对象） / 校验错误列表 |

## 3. 配置结构（yaml schema）

### mvp 最小配置

```yaml
config_version: "0.1"

model:
  base_model: "runwayml/stable-diffusion-v1-5"
  precision: "fp16"
  device: "cuda"

data:
  dataset_path: "path/to/dataset"
  batch_size: 1
  cache_latents: auto  # auto / true / false

training:
  max_steps: 1000
  learning_rate: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_steps: 100
  gradient_accumulation_steps: 4

lora:
  rank: 32
  alpha: 32
  target_modules:
    - "to_q"
    - "to_k"
    - "to_v"
    - "to_out"

sampling:
  sample_every_steps: 100
  sample_prompt: "a photo of sks person"
  sample_seed: 42

export:
  format: "comfyui"
  output_dir: "outputs"
```

## 4. 默认值策略（参考 kohya）

| 参数 | 默认值 | 来源 | 决策理由 |
| --- | --- | --- | --- |
| lora.rank | 32 | 社区共识 | 平衡效果与显存 |
| lora.alpha | =rank | kohya | alpha=rank 最稳定 |
| training.lr | 1e-4 | kohya | sd1_5 标准学习率 |
| training.optimizer | adamw | kohya | 标准选择 |
| training.scheduler | cosine | kohya | 效果优于 linear |
| data.batch_size | 1 | 低显存友好 | 配合梯度累积 |
| training.gradient_accumulation_steps | 4 | 经验值 | 等效 bs=4 |

## 5. 配置加载流程

1. 加载 yaml 文件。
2. 检查 config_version（必需）。
3. 合并默认值（defaults.yaml）。
4. 应用 cli 覆盖参数。
5. 类型与范围校验。
6. 输出 resolved_config。

## 6. 版本迁移（向后兼容）

```python
# 示例：v0.1 → v0.2 迁移
def migrate_config(config, from_version, to_version):
    if from_version == "0.1" and to_version == "0.2":
        # 参数重命名
        if "lora.network_dim" in config:
            config["lora.rank"] = config.pop("lora.network_dim")
        
        # 新增字段默认值
        config.setdefault("training.gradient_checkpointing", False)
    
    return config
```

## 7. 校验规则

| 字段 | 规则 | 错误提示 |
| --- | --- | --- |
| model.base_model | 必需、字符串 | "model.base_model is required" |
| lora.rank | 整数、8-64 | "lora.rank must be in [8, 64]" |
| lora.alpha | 可选、正数 | "lora.alpha must be positive" |
| training.max_steps | 必需、正整数 | "training.max_steps is required" |
| data.dataset_path | 必需、存在的路径 | "dataset_path not found: {path}" |

## 8. cli 覆盖策略

```bash
# 覆盖配置参数
train -c config.yaml \
  --lora.rank=16 \
  --training.learning_rate=0.0002 \
  --data.batch_size=2
```

优先级：cli > yaml > defaults

## 9. 实现参考

- kohya: `sd-scripts/library/config_util.py`（配置加载与校验）
- onetrainer: `modules/config/` （配置类与版本管理）
