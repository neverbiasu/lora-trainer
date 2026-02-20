# recorder_manager 层设计（mvp v0.1）

## 1. 职责

| 项 | 说明 |
| --- | --- |
| 训练快照 | 记录可复现要素 |
| 训练指标 | 记录训练指标与采样结果 |
| 运行报告 | 输出可机器解析报告 |

## 2. 输入/输出

| 类型 | 内容 |
| --- | --- |
| 输入 | `trainer` 生命周期事件（start/step/end）/ 配置与环境信息 |
| 输出 | run_metadata（json）/ config_snapshot（yaml）/ 训练曲线（loss/step） |

## 3. 必须记录的 7 项

1. base_model 标识（hash 或版本）。
2. 全量训练配置。
3. seed。
4. 数据集清单 hash。
5. 代码版本。
6. 运行环境版本。
7. 时间戳与训练曲线。

## 4. 报告格式

- 人类可读摘要（stdout）。
- 机器可读 json（用于 ci/回归）。
