# 设计文档索引（mvp v0.1）

## 当前阶段

- MVP（v0.1）核心闭环已完成：`validate → train → export`
- 当前进入 V0.2 工程化阶段：`load_weights / resume / export-only / metadata schema`

## 本周目标（V0.2）

- 已补充设计文档：
	- `TRAINER.md`：resume/export-only 状态流与错误码
	- `lora.md`：`load_weights` 与导出 metadata schema
	- `run_manager.md`：checkpoint manifest 设计
	- `cli_design.md`：参数分发契约与错误码

- 参考来源（本地）：
	- `docs/references/sd-scripts/train_network.py`
	- `docs/references/sd-scripts/library/train_util.py`
	- `docs/references/diffusers/examples/dreambooth/`
	- `docs/references/OneTrainer/modules/trainer/`

## 核心设计文档

| 主题 | 文档 | 状态 |
| --- | --- | --- |
| 架构总览 | [architecture.md](architecture.md) | ✅ |
| 文件结构与参考 | [file_structure_and_references.md](file_structure_and_references.md) | ✅ |

## 层级设计文档

| 层级 | 文档 | 状态 |
| --- | --- | --- |
| trainer 层 | [trainer.md](trainer.md) | ✅ |
| lora 模块 | [lora.md](lora.md) | ✅ |
| hyperparam_policy 层 | [hyperparam_policy.md](hyperparam_policy.md) | ✅ |
| model_adapter 层 | [model_adapter.md](model_adapter.md) | ✅ |
| recorder_manager 层 | [recorder_manager.md](recorder_manager.md) | ✅ |
| 数据集与数据流 | [data_pipeline.md](data_pipeline.md) | ✅ |
| exporter | [exporter.md](exporter.md) | ✅ |

## 新增设计文档

| 主题 | 文档 | 状态 |
| --- | --- | --- |
| 配置管理器 | [config_manager.md](config_manager.md) | ✅ |
| 运行管理器 | [run_manager.md](run_manager.md) | ✅ |

## 参考代码提取文档

| 来源 | 文档 | 状态 |
| --- | --- | --- |
| kohya 训练循环 | [extraction_kohya_training_loop.md](extraction_kohya_training_loop.md) | ✅ |
| LyCORIS 算法适配器 | [extraction_lycoris_algo_adapter.md](extraction_lycoris_algo_adapter.md) | ✅ |

## 数据与工具设计文档

| 主题 | 文档 | 状态 |
| --- | --- | --- |
| 数据加载器 | [data_loader.md](data_loader.md) | ✅ |
| 超参数规则 | [hyperparam_rules.md](hyperparam_rules.md) | ✅ |
| LoRA实现细节 | [lora_adapter_impl.md](lora_adapter_impl.md) | ✅ |
| CLI设计 | [cli_design.md](cli_design.md) | ✅ |
| 预设配置 | [presets.md](presets.md) | ✅ |
| SD1.5实现 | [sd15_implementation.md](sd15_implementation.md) | ✅ |
| 错误处理 | [error_handling.md](error_handling.md) | ✅ |

## V0.2 设计任务映射

| 目标 | 设计文档 | 状态 |
| --- | --- | --- |
| Resume 状态机与生命周期 | [trainer.md](trainer.md) | ⏳ 待补充 |
| LoRA 回载与导出元信息 schema | [lora.md](lora.md) | ⏳ 待补充 |
| Checkpoint manifest 与保留策略 | [run_manager.md](run_manager.md) | ⏳ 待补充 |
| CLI 参数流（resume/export-only） | [cli_design.md](cli_design.md) | ⏳ 待补充 |

---

- 文档名使用 snake_case。
- ✅ = 已完成，⏳ = 待补充。
