# run_manager 设计（mvp v0.1）

## 1. 职责

| 项 | 说明 |
| --- | --- |
| 目录管理 | 创建 run 目录结构 |
| 快照保存 | 保存 config snapshot、版本信息 |
| 日志管理 | 统一日志流（stdout + file） |

## 1.1 V0.2 新增职责

| 项 | 说明 |
| --- | --- |
| checkpoint manifest | 维护可恢复训练所需最小状态（step、weights_path、updated_at） |
| resume 元信息 | 提供 resume 时可直接读取的最近 checkpoint 指针 |

## 2. run 目录结构

```
outputs/
└── run_20260205_143022_sd15_lora/      # run_id
    ├── config_snapshot.yaml            # 完整配置快照
    ├── metadata.json                   # 运行元信息（7要素）
    ├── logs/
    │   └── train.log                   # 训练日志
    ├── checkpoints/
    │   ├── step_0500.safetensors       # 中间 checkpoint
    │   └── step_1000.safetensors       # 最终 checkpoint
    ├── samples/
    │   ├── step_0100.png               # 采样图
    │   ├── step_0200.png
    │   └── ...
    └── export/
        ├── lora_weights.safetensors    # 导出的 LoRA
        └── metadata.json               # 导出元信息
```

## 3. run_id 生成规则

### 算法说明 — generate_run_id

- 目的：生成用于 run 目录名的可读且确定性的 run_id。
- 输入：已解析的 `config`（应包含 `model.base_model` 与 `lora.algorithm`）。
- 输出：格式为 `run_{YYYYMMDD_HHMMSS}_{model_name}_{algo}` 的字符串 `run_id`。
- 步骤：
  1. 获取当前时间戳，格式化为 `YYYYMMDD_HHMMSS`。
  2. 从 `config["model"]["base_model"]` 提取 `model_name`（取最后一段路径）。
  3. 从 `config["lora"]["algorithm"]` 读取 `algo`（默认 `lora`）。
  4. 拼接为 `run_{timestamp}_{model_name}_{algo}` 并返回。

- 接口说明：`generate_run_id(config: dict) -> str`

## 4. metadata.json（7 要素快照）

```json
{
  "run_id": "run_20260205_143022_sd15_lora",
  "created_at": "2026-02-05T14:30:22Z",
  "reproducibility": {
    "base_model_hash": "sha256:abc123...",
    "config_hash": "sha256:def456...",
    "seed": 42,
    "data_manifest_hash": "sha256:789abc...",
    "code_version": "v0.1.0",
    "environment": {
      "torch": "2.0.0",
      "diffusers": "0.21.0",
      "transformers": "4.30.0"
    }
  },
  "training_metrics": {
    "total_steps": 1000,
    "final_loss": 0.0234,
    "duration_seconds": 1234
  }
}
```

## 4.1 checkpoint manifest（新增）

建议新增文件：`checkpoints/manifest.json`

```json
{
  "latest_step": 500,
  "latest_checkpoint": "checkpoints/step_0500.safetensors",
  "history": [
    {"step": 100, "path": "checkpoints/step_0100.safetensors", "created_at": "..."},
    {"step": 500, "path": "checkpoints/step_0500.safetensors", "created_at": "..."}
  ]
}
```

用途：

- `--resume run_dir` 时无需扫描目录，直接读 manifest。
- `--export-only --resume run_dir` 可直接定位最新 checkpoint。

## 5. config_snapshot.yaml（完整配置）

保存 `resolved_config`（包含所有默认值）：

```yaml
# 自动添加元信息
_snapshot_meta:
  created_at: "2026-02-05T14:30:22Z"
  original_config_path: "/path/to/config.yaml"
  config_version: "0.1"

# 完整配置（含默认值）
model:
  base_model: "runwayml/stable-diffusion-v1-5"
  precision: "fp16"
  device: "cuda"
# ... 其余配置
```

## 6. 日志管理

### 日志格式

```
[2026-02-05 14:30:22] [INFO] Starting training run: run_20260205_143022_sd15_lora
[2026-02-05 14:30:23] [INFO] Loaded config: config.yaml
[2026-02-05 14:30:25] [INFO] Model loaded: runwayml/stable-diffusion-v1-5
[2026-02-05 14:30:30] [INFO] LoRA injected to 48 modules
[2026-02-05 14:30:35] [INFO] Training started
[2026-02-05 14:31:00] [TRAIN] Step 100/1000 | Loss: 0.1234 | LR: 1.0e-04
[2026-02-05 14:31:25] [SAMPLE] Generated sample at step 100
[2026-02-05 14:32:00] [TRAIN] Step 200/1000 | Loss: 0.0987 | LR: 9.8e-05
...
```

### 日志流配置

日志配置（行为规范）

- 目标：将 stdout 与文件日志统一为同一时间戳流，便于可观测性与可复现性。
- 处理器：写入 `logs/train.log` 并输出到 stdout。
- 级别与格式：默认 INFO 级别；格式应包含时间戳与级别，例如 `[YYYY-MM-DD HH:MM:SS] [LEVEL] message`。
- 日志轮转：可选（非 MVP）；生产环境使用外部轮转策略。
- 示例日志行（供人工/CI 阅读）：
  - `[2026-02-05 14:30:22] [INFO] 开始训练 run：run_20260205_143022_sd15_lora`
  - `[2026-02-05 14:31:00] [TRAIN] Step 100/1000 | Loss: 0.1234 | LR: 1.0e-04`

## 7. 生命周期钩子

接口与运行时行为

- 方法签名：
  - `start(config: dict) -> Path` — 生成 `run_id`、创建目录、保存 `config_snapshot`、初始化 `metadata`、配置日志，并返回 `run_dir`。
  - `save_checkpoint(step: int, weights: Mapping[str, torch.Tensor]) -> Path` — 将模型/适配器权重保存到 `checkpoints/step_XXXX.safetensors`。
  - `save_sample(step: int, image: PIL.Image.Image) -> Path` — 将生成的图片写入 `samples/step_XXXX.png`。
  - `end(metrics: dict) -> None` — 完成 `metadata.training_metrics` 并写入 `metadata.json`。

- `start()` 流程（详细）：
  1. 计算 `run_id = generate_run_id(config)`。
  2. 解析 `export_dir = config["export"]["output_dir"]`（缺省 `./output`）。
  3. 创建 `run_dir = export_dir / run_id` 及子目录：`logs/`、`checkpoints/`、`samples/`、`export/`。
  4. 将已解析的完整配置持久化为 `config_snapshot.yaml`（包含 `_snapshot_meta`）。
  5. 初始化 `metadata`（7 要素骨架），并持久化为 `metadata.json`。
  6. 配置日志处理器，写入 `logs/train.log` 并输出至 stdout。

- 检查点与样本命名规范：
  - 检查点：`checkpoints/step_{step:04d}.safetensors`
  - 样本：`samples/step_{step:04d}.png`

- 错误处理与不变量：
  - 对相同 `run_id`，`start()` 应幂等（非显式允许时不覆盖现有快照）。
  - 写入产物的方法必须断言 `run_dir` 已创建；若未调用 `start()`，应抛出 `RuntimeError`。
  - `metadata.json` 在 `end()` 后须包含 `created_at`、`reproducibility` 骨架与最终 `training_metrics`。

- 可观测性：
  - 在 `start()` 成功后输出 INFO 日志，包含 `run_id` 与解析后的 `run_dir`。
  - 每次保存检查点与样本时记录日志（包含路径与 step）。

> 说明：本节定义实现的“行为契约（what）”，实际实现应放在 `src/`，并在实现代码中遵循本契约。

### 7.1 V0.2 补充接口

- `resolve_resume_checkpoint(resume_path: Path) -> Path`
  - 输入可以是 run_dir 或 checkpoint 文件。
  - run_dir 场景优先读取 `checkpoints/manifest.json`。
- `update_checkpoint_manifest(step: int, checkpoint_path: Path) -> None`
  - 每次 `save_checkpoint` 后更新。

## 8. 实现参考

- kohya: `sd-scripts/library/train_util.py`（checkpoint 保存逻辑）
- onetrainer: `modules/run/` （run 目录管理）
- diffusers: `docs/references/diffusers/examples/dreambooth/`（训练产物组织）
