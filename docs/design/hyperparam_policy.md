# hyperparam_policy 层设计（mvp v0.1）

## 1. 职责

| 类型 | 说明 |
| --- | --- |
| recommendation | 基于模型族/数据规模/显存档给出建议 |
| constraint | 限制 rank/alpha/lr 合理范围 |
| validation | 配置冲突与硬件可行性检查 |

## 2. 输入/输出

| 类型 | 内容 |
| --- | --- |
| 输入 | resolved_config / 硬件信息（显存/设备类型）/ 数据规模统计（图像数、分辨率分布） |
| 输出 | 推荐参数集（用于 preset）/ 校验结果（通过/阻断 + 错误原因） |

## 3. mvp 规则（示意）

1. rank：8–64（按显存档自动推荐）。
2. alpha：默认等于 rank。
3. learning_rate：sd1_5 推荐 1e-4。
4. batch_size：默认 1，配合 gradient_accumulation。
5. cache_latents：低显存自动启用。

## 4. 报错策略

1. 违反硬约束：阻断训练并给出可操作建议。
2. 违反软约束：允许运行但给出警告。

## 5. 未来扩展

- sdxl 的双 text_encoder 策略。
- 多算法的策略差异化配置。
