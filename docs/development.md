# 开发文档（mvp v0.1）

> 目标：定义 mvp 的开发边界、交付物与验收标准。不写具体代码，只产出设计与规范文档。

## 1. 开发范围

| 范围 | 内容 |
| --- | --- |
| 做 | 1) cli 单入口：`train -c config.yaml` 2) 单配置源：yaml（含 `config_version`） 3) 训练闭环：`validate → train → export` 4) 可复现：运行快照（配置、版本、数据摘要、seed、环境信息） 5) mvp 模型族：sd1_5（sdxl 作为 phase_2） |
| 不做 | 1) 数据采集/打标/预处理 ui 2) 推理 ui/服务化托管 3) 多脚本入口 |

## 2. mvp 约束与承诺

| 维度 | 约束/承诺 |
| --- | --- |
| 硬件 | sd1_5 lora：最低 8gb（best_effort），推荐 12gb；sdxl lora（phase_2）：最低 12gb（紧张），推荐 20gb+ |
| 数据规模 | 典型 lora 数据集：20–1000 张；更大数据集视为 fine_tuning 管线（后续阶段处理） |
| 训练时长 | sd1_5：10–40 分钟可见结果；sdxl：30–120 分钟可见结果 |
| 质量定义 | 固定 prompt + 固定 seed 的对比采样（训练前/中/后）；过拟合信号：生成构图/细节过度重复 |

## 3. 交付物清单

1. 设计文档（见 docs/design）。
2. 运行快照规范（`recorder_manager`）。
3. `hyperparam_policy` 规则集（推荐/约束/校验）。
4. mvp 预设参数（quick/balanced/quality）。

## 4. 运行快照（可复现要素）

1. base model 标识（文件 hash 或模型版本）。
2. 全量训练配置（含 bucketing/cache/steps/lr/rank/alpha）。
3. seed（全局 + 采样）。
4. 数据集清单（文件列表 hash + caption hash）。
5. 代码版本（git commit / tag）。
6. 运行环境（torch/diffusers/transformers 版本）。
7. 时间戳与训练曲线（loss/step）。

## 5. 里程碑（文档阶段）

1. m1：设计冻结（本次）。
2. m2：参考实现提取（kohya / lycoris / onetrainer）。
3. m3：实现阶段规划（本次不做）。

## 6. 验收标准（文档）

- 每个层级都有独立设计文档（docs/design）。
- 训练闭环流程在设计文档中可追溯。
- 超参数策略与约束有明确规则描述。
