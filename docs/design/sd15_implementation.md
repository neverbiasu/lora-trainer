# SD1.5 实现设计（M3 基线）

## 1. 目标

给出清晰的 SD1.5 LoRA 训练基线规范，确保实现保持精简、可复现，并与 M3 验收标准一致。

## 2. 架构基线

| 组件 | 作用 | 典型形状 |
| --- | --- | --- |
| VAE | 图像与 latent 双向转换 | `3x512x512 <-> 4x64x64` |
| UNet | 噪声预测核心网络 | latent + timestep + text cond |
| Text Encoder | 文本编码 | token 序列 -> hidden states |
| Scheduler | 扩散步策略 | 训练与采样的时间步操作 |

## 3. 运行假设

- 默认训练分辨率为 512。
- latent 缩放与归一化遵循 SD1.5 兼容约定。
- M3 中 LoRA 注入 UNet 的注意力投影层。
- 文本编码器默认冻结。

## 4. 适配器边界

仅 `ModelAdapter` 可以负责以下能力：

- 加载 SD1.5 组件
- 暴露 LoRA 注入目标模块
- 提供文本/图像转换辅助接口

该边界保证 `Trainer` 不感知具体模型族实现细节。

## 5. 数据路径契约

### 文本路径

```text
prompt -> tokenizer -> input ids -> text encoder -> text embeddings
```

### 图像路径

```text
image tensor -> normalize -> vae encode -> latent
latent -> vae decode -> image tensor
```

## 6. 显存与稳定性策略

M3 策略顺序：

1. 先冻结基础模块。
2. 仅按配置启用 gradient checkpointing。
3. 将优化策略明确记录进配置快照。

## 7. 目标模块策略

- 基线优先使用注意力投影层，兼顾效果与参数效率。
- 目标列表集中管理并可版本化。
- 配置目标缺失时拒绝静默降级。

## 8. 训练兼容规则

- batch tensor 必须满足 latent 与 embedding 维度约束。
- 在 step loop 启动前完成 dtype/device 组合校验。
- scheduler 与 loss 路径需使用一致的 timestep dtype。

## 9. 可观测性

run 启动时记录以下字段：

- `model_family=sd15`
- `base_model_identifier`
- `resolution`
- `dtype`
- `device`
- `target_module_count`

## 10. 验收标准（M3）

- SD1.5 基线可完整加载并返回可用 handles。
- LoRA 目标发现与注入具备确定性。
- 单次短训练（`max_steps=10`）可产出 checkpoint 与 sample。
- 导出阶段可消费训练产出的 adapter 制品且无 schema mismatch。

## 11. 非目标

- M3 不覆盖 SDXL 双编码器细节。
- 不覆盖高级采样器变体与高分辨率多阶段流程。
- 不覆盖分布式训练配置。

## 12. 后续扩展（M4+）

- 增加 SDXL 专用适配配置。
- 增加模型族统一能力报告。
- 增加面向不同生态的导出兼容矩阵。
