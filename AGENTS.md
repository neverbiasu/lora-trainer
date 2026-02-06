# AGENTS 指南（顶层设计约束）

## U 型设计原则（Top-down + Bottom-up）

- **Top-down（核心骨架先行）**
  - 先定义统一生命周期与稳定接口边界（如 `TrainerCore`、`ComponentFactory/Registry`）。
  - 先定配置与运行态的最小闭环（`ConfigManager`、`RunManager`）。
  - 目标：确保“可复现 + 可扩展”的最小可用闭环。

- **Bottom-up（工具能力反哺）**
  - 从真实训练痛点提炼工具与子能力（如 `validate_dataset`、采样、导出与转换）。
  - 以接口/协议接入核心骨架，避免脚本堆叠与重复入口。
  - 目标：让工具能力推动抽象和配置规范演进。

- **U 型闭环**
  - 核心骨架提供“插槽”，工具能力反哺抽象与配置设计。
  - 优先保障端到端闭环：`validate → train → export`。
  - 避免“多脚本入口 + 参数爆炸”的演化路径。

## 执行约束（适用于后续实现）

- 保持**单入口**（CLI）与**单配置源**（YAML）为 MVP 默认路径。
- 扩展新模型/算法时，优先通过 Adapter 接口实现，而不是新增脚本入口。
- 每个 run 必须可复现：记录 config snapshot、版本信息、seed 与数据摘要。
