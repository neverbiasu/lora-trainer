# LoRA Adapter 实现设计（M3）

## 1. 目标

定义确定性、最小化的 SD1.5 UNet LoRA 注入方案，保证易验证、易训练、易导出。

## 2. 设计原则

- 基础模型始终冻结，仅训练低秩适配器参数。
- 目标模块匹配必须显式、可复现。
- merge/export 行为必须可逆、可观测。
- 实现保持可插拔，便于后续接入 LyCORIS 变体。

## 3. 数学契约

对基础权重 `W`，LoRA 增量定义为：

- `ΔW = B @ A`
- `W' = W + s * ΔW`
- `s = alpha / rank`

约束：

- `rank > 0`
- `alpha > 0`
- `rank << min(d_in, d_out)`（保证参数效率）

## 4. 组件模型

| 组件 | 职责 |
| --- | --- |
| `LoRAModule` | 持有低秩参数与缩放策略 |
| `LoRALinear` | 组合冻结基座投影与 LoRA 增量 |
| `LoRAInjector` | 查找目标模块并替换/包裹 |
| `LoRARegistry` | 追踪注入模块，支持保存/加载/合并 |

## 5. 对外接口（API 签名）

```text
class LoRAInjector:
    inject(model, target_patterns, rank, alpha, dropout) -> LoRARegistry

class LoRARegistry:
    trainable_parameters() -> Iterator[Parameter]
    state_dict() -> dict[str, Tensor]
    load_state_dict(weights) -> None
    merge_into_base() -> None
```

## 6. 目标匹配策略（SD1.5）

### 默认目标族

- `attn2.to_q`, `attn2.to_k`, `attn2.to_v`, `attn2.to_out.0`
- `attn1.to_q`, `attn1.to_k`, `attn1.to_v`, `attn1.to_out.0`

### 规则

- 以规范化模块路径匹配，不依赖模糊子串。
- M3 仅允许线性投影类模块。
- 被跳过模块必须记录原因（`type_mismatch`、`not_found`、`duplicate`）。

## 7. 初始化策略

- 下投影 `A` 采用方差稳定初始化。
- 上投影 `B` 初始为 0。
- 保证训练初始 `ΔW` 接近 0，避免破坏基座输出。

## 8. 训练期语义

- 前向输出 = 基座输出 + LoRA 增量。
- 基座参数不参与更新。
- 优化器仅接收 LoRA 参数。

### 伪代码

```text
inject(model):
  for each target module path:
    validate module type
    wrap with LoRA-aware projection
    register adapter handle
  return registry
```

## 9. 保存 / 加载 / 合并契约

### 保存

- 仅保存 adapter 权重，不保存完整基座模型。
- 同步保存元信息：`base_model`、`rank`、`alpha`、target 策略。

### 加载

- 先重建 adapter 结构，再加载权重。
- 校验元信息与当前目标结构兼容性。

### 合并

- 在 no-grad 语境下执行权重合并。
- 在 registry 标记 merge 状态，默认防止重复合并。

## 10. 校验与指标

必须暴露以下指标：

- 注入模块数量
- 可训练参数量与占比
- 缺失目标模块列表
- 加载/合并兼容性检查结果

## 11. 验收标准（M3）

- 在 SD1.5 UNet 默认目标上注入成功。
- 默认 rank 配置下，可训练参数占比 < 1%。
- adapter 权重可在全新进程中保存并重载。
- merge 路径在冒烟推理下数值稳定。

## 12. 后续扩展（M4+）

- 扩展 registry schema 以支持 LoCon/LoHa（LyCORIS）。
- 增加逐层 rank/alpha 覆盖策略。
- 增加多 adapter 叠加组合策略。
