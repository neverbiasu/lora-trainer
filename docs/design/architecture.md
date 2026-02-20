# 架构总览（mvp v0.1）

## 1. 核心层级

| 层级 | 职责 |
| --- | --- |
| trainer | 编排与生命周期 |
| algo_adapter | 算法与训练循环 |
| hyperparam_policy | 推荐/约束/校验 |
| model_adapter | 模型加载与条件构造 |
| recorder_manager | 可复现快照与日志 |

## 2. 关系与依赖

| 来源 | 依赖 | 说明 |
| --- | --- | --- |
| trainer | algo_adapter / hyperparam_policy / model_adapter / recorder_manager | 统一调度与生命周期管理 |
| algo_adapter | model_adapter | 获取模型与目标模块信息 |
| hyperparam_policy | config + hardware | 只依赖配置与硬件信息 |
| recorder_manager | runtime events | 只观察运行态，不参与训练决策 |

## 3. 训练生命周期（抽象流程）

1. `config_manager` 解析配置（含默认值与版本）。
2. `hyperparam_policy` 计算推荐/约束/校验。
3. `model_adapter` 加载 base_model + text_encoder + conditioning。
4. `algo_adapter` 构建 lora 模块并注入目标层。
5. `trainer` 执行 `train_step`（含采样/保存/日志）。
6. `recorder_manager` 保存快照与指标。
7. `exporter` 生成目标格式产物（mvp：comfyui）。

## 4. 关键边界

1. 超参数策略与算法实现分离（避免策略逻辑散落在算法内）。
2. 训练循环与数据管线分离（避免数据逻辑侵入算法）。
3. 记录与业务逻辑分离（避免记录行为影响训练）。

## 5. 扩展点

- 新模型族：扩展 `model_adapter`（sdxl / sd2_1）。
- 新算法：扩展 `algo_adapter`（lokr / locon / dora）。
- 新策略：扩展 `hyperparam_policy`（硬件/数据条件）。
- 新导出格式：扩展 `exporter`（a1111 / 其它）。
