# Trainer 设计（函数级可落地版）

## 1. 目标

把 `src/lora_trainer/trainer.py` 变成可执行训练编排层，明确每个函数的职责边界。

本设计关注：

- 函数职责边界
- 函数调用顺序
- 与 `config_manager / run_manager / model_adapter / lora` 的协作契约

---

## 2. 生命周期调用链

```text
CLI
  -> ConfigManager.resolve + validate
  -> Trainer(config).train()

Trainer.train()
  -> start()
  -> loop:
       loss = train_step(batch)
       if step % validate_every_n_steps == 0: validate(step)
       if step % save_every_n_steps == 0:     save_checkpoint(step)
  -> end()
```

### 2.1 V0.2 补充状态流

```text
CLI --resume path
  -> Trainer.start(resume=path)
       load base models
       inject lora modules
       lora.load_weights(path)
       restore global_step (from checkpoint meta or filename)
  -> Trainer.train()

CLI --export-only --resume path
  -> Trainer.start(resume=path)
  -> Trainer.end()   # 跳过训练循环
```

---

## 3. 函数逐一设计

### 3.1 `Trainer.__init__(config)`

**职责**：保存配置，初始化字段占位，不执行重操作。

字段：

```python
self.config           # 已 resolve + validate 的配置字典
self.device           # torch.device
self.global_step = 0
self.last_loss = 0.0  # 供 end() 写 summary 使用

self.model_adapter    = None
self.lora_adapter     = None
self.unet             = None
self.vae              = None
self.text_encoder     = None

self.optimizer        = None
self.scheduler        = None
self.noise_scheduler  = None
self.data_loader      = None
self.run_manager      = None
```

---

### 3.2 `Trainer.start(resume: str | None = None)`

**职责**：一次性建立所有运行态依赖。

执行顺序：

1. 创建 run 目录（`RunManager.start(config)`）。
2. 加载基础模型（`SD15ModelAdapter.load_models()`）。
3. 创建数据加载器。
4. 注入 LoRA（`LoRAAdapter.apply_to(unet=..., text_encoder=...)`）。
5. 创建 optimizer 与 lr scheduler。
6. 若 `resume` 不为 None：
   - `lora_adapter.load_weights(resume)`
   - 从 checkpoint 元数据或文件名解析 `global_step`。
7. 记录启动日志（seed、rank/alpha、max_steps、device）。

**失败策略**：任一步骤抛错则直接终止，不进入 `train()`。

---

### 3.3 `Trainer.train()`

**职责**：完整训练入口，管理 start → loop → end 生命周期。

```python
# pseudocode
def train(self) -> None:
    self.start()
    try:
        for epoch in range(num_epochs):
            for batch in self.data_loader:
                if self.global_step >= max_steps:
                    break
                loss = self.train_step(batch)
                self.last_loss = loss

                if validate_every > 0 and self.global_step % validate_every == 0:
                    self.validate(self.global_step)
                if self.global_step % save_every == 0:
                    self.save_checkpoint(self.global_step)

                self.global_step += 1
    finally:
        self.end()
```

**触发条件配置键**：

| 键 | 所属节 | 类型 | 默认值 | 说明 |
|---|---|---|---|---|
| `max_train_steps` | `training` | int | — | 总步数上限 |
| `save_every_n_steps` | `training` | int | 500 | checkpoint 保存间隔 |
| `every_n_steps` | `validation` | int | 250 | 采样验证触发间隔；0 = 不采样 |

---

### 3.4 `Trainer.train_step(batch)`

**职责**：执行单步训练，返回标量 loss。

执行顺序：

1. 从 batch 取图像与 captions，`.to(device)`。
2. `encode_image` → latents（no_grad）。
3. 随机加噪（`noise_scheduler.add_noise`）。
4. `encode_prompt` → text_embeddings（no_grad）。
5. `unet(noisy_latents, timesteps, text_embeddings)` → `model_pred`。
6. `mse_loss(model_pred, noise) / grad_accum_steps`，`backward()`。
7. 若到达累积边界：`optimizer.step → scheduler.step → zero_grad`。

**梯度累积**：由 `training.gradient_accumulation`（默认 1）控制。

---

### 3.5 `Trainer.validate(step)`

**职责**：固定 prompt + 固定 seed 采样，保存图像供肉眼判断训练进度。

**触发条件**：由 `train()` 在 `global_step % validation.every_n_steps == 0` 时调用；`every_n_steps=0` 或 `validation` 节缺失时永不调用。

**采样参数**（独立的 `validation` 配置节）：

```yaml
validation:                       # 整节可选；缺失时 validate() 直接 return
  every_n_steps: 250              # 触发间隔；0 = 不采样
  prompt: "a photo of a sks subject"
  negative_prompt: ""
  seed: 1234                      # 与 training.seed 独立，固定保证图像可横向对比
  num_inference_steps: 20
  guidance_scale: 7.5
  width: 512
  height: 512
```

**输出**：`run/samples/step_{step:06d}.png`。

```python
# pseudocode
def validate(self, step: int) -> None:
    val_cfg = self.config.get("validation")
    if not val_cfg:
        return
    generator = torch.Generator(device=self.device).manual_seed(val_cfg["seed"])
    # 走完整去噪流程，保存图像到 run/samples/step_{step:06d}.png
    logger.info("step=%d sample saved", step)
```

---

### 3.6 `Trainer.save_checkpoint(step)`

**职责**：周期性保存训练状态。

**MVP 最小保存内容**：

```python
# pseudocode
weights_path  = run/checkpoints/step_{step:06d}.safetensors
metadata_path = run/checkpoints/step_{step:06d}.json
# weights_path: lora_adapter.state_dict()
# metadata_path: { "global_step": step, "rank": ..., "alpha": ... }
```

可后续扩展：optimizer / scheduler / scaler 状态。

---

### 3.7 `Trainer.end()`

**职责**：训练结束清理与最终导出。

执行顺序：

1. 导出最终 LoRA：`run/export/lora_final.safetensors`。
2. 写 run summary（`total_steps=self.global_step`、`final_loss=self.last_loss`、耗时）。
3. `RunManager.end(metrics)`。

---

## 4. 配置节汇总

```yaml
training:
  seed: 42
  learning_rate: 1e-4
  lr_scheduler: cosine          # cosine | constant
  batch_size: 4
  gradient_accumulation: 1
  max_train_steps: 1500
  save_every_n_steps: 500

validation:                     # 可选；缺失时跳过 validate()
  every_n_steps: 250            # 0 = 不采样
  prompt: "a photo of a sks subject"
  negative_prompt: ""
  seed: 1234
  num_inference_steps: 20
  guidance_scale: 7.5
  width: 512
  height: 512
```

---

## 5. 与其它模块的接口契约

| 模块 | Trainer 调用方式 |
|---|---|
| `ConfigManager` | 只接收已 resolve + validate 的配置，不重复做字段合法性检查 |
| `RunManager` | `start()` 获取 run 目录并写快照；`end()` 写 summary |
| `SD15ModelAdapter` | `load_models()` / `encode_image()` / `encode_prompt()` |
| `LoRAAdapter` | `apply_to()` / `get_trainable_params()` / `load_weights()` / `export_weights()` |

---

## 6. 异常处理

| 场景 | 处理策略 |
|---|---|
| OOM | 捕获 `torch.cuda.OutOfMemoryError`，提示降低 `batch_size` / `resolution` / 开 `gradient_checkpointing` |
| batch shape 非法 | 快速失败，标记 run 为失败态 |
| resume 路径不存在 | `E061`：抛 `FileNotFoundError` |
| checkpoint 格式不支持 | `E062`：抛 `ValueError` |
| LoRA 权重加载失败 | `E063`：missing/unexpected keys 超阈值时抛错 |

---

## 7. 日志契约

INFO 级别使用结构化 key=value：

```
step=100 loss=0.0842 lr=9.8e-5 grad_norm=0.31 time_per_step=420ms
```

---

## 8. 验收标准（V0.2）

- `--max-train-steps 10` 端到端可完成，产出 checkpoint + 采样图。
- `loss` 趋势不发散，日志 step 连续。
- `--resume` 路径可正确还原 `global_step`，续训 loss 无跳变。
- `--export-only --resume` 仅导出，不进入训练循环。

---

## 9. 参考实现

- `docs/references/sd-scripts/train_network.py`（训练主流程与恢复入口）
- `docs/references/sd-scripts/library/train_util.py`（checkpoint 生命周期）
- `docs/references/diffusers/examples/dreambooth/`（训练脚本组织方式）
- `docs/references/OneTrainer/modules/trainer/`（训练编排职责划分）
