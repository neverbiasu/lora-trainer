# LoRA 模块设计（mvp v0.1）

## 1. 职责

| 项 | 说明 |
| --- | --- |
| LoRA 注入 | 将 LoRA 层注入到 UNet 注意力层 |
| 可训练参数管理 | 提供 LoRA 权重供优化器更新 |
| 权重导出 | 导出训练好的 LoRA 权重 |

## 2. 核心类

| 类 | 说明 |
| --- | --- |
| `LoRAModule` | 单个 LoRA 层实现（lora_down/up + forward） |
| `LoRAAdapter` | 管理多个 LoRAModule，处理注入与导出 |

## 3. mvp 实现范围

1. 仅标准 LoRA（UNet 注意力层）
2. 可选训练 text_encoder（SD1.5）
3. 不支持 LoHA/LoKr/其他变体（YAGNI）

## 4. 接口设计

**LoRAAdapter 方法**：
1. `prepare(model)` - 注入 LoRA 到目标层
2. `get_trainable_params()` - 返回可训练参数列表
3. `export_weights(save_path, metadata)` - 导出权重到 safetensors

**设计决策**：
- 不使用抽象基类（只做 LoRA，不需要通用接口）
- 通过 forward hook 自动工作（不需要显式 forward 方法）
- 训练循环在 Trainer 中管理（LoRAAdapter 只负责参数管理）
