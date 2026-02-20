# 模块与参考代码安排（mvp v0.1）

## 1. 核心模块文件安排表

| 模块 | 文件路径 | 职责 | 设计文档 | 参考仓库 | 克隆必要 | mvp阶段 |
| --- | --- | --- | --- | --- | --- | --- |
| **架构基础** | | | | | | |
| config_manager | `src/core/config_manager.py` | 加载/校验 yaml 配置、填充默认值、版本管理 | 需要（补充） | kohya、onetrainer | 中 | mvp |
| run_manager | `src/core/run_manager.py` | 创建 run 目录、保存快照、日志管理 | 需要（补充） | onetrainer | 中 | mvp |
| recorder_manager | `src/core/recorder_manager.py` | 记录 7 要素快照、训练指标、报告生成 | ✅ | kohya | 中 | mvp |
| **训练编排层** | | | | | | |
| trainer | `src/core/trainer.py` | 生命周期管理（start/train_step/end）、调度其他模块 | ✅ | onetrainer | 高 | mvp |
| data_loader | `src/data/data_loader.py` | 数据集加载、bucketing、缓存 lora | 需要（补充） | kohya、flymyai | 高 | mvp |
| **算法层** | | | | | | |
| lora_module | `src/lora_trainer/lora.py` | LoRA 实现：权重注入、参数管理、导出 | ✅ (lora.md) | lycoris、kohya | 高 | mvp |
| **模型加载层** | | | | | | |
| model_adapter_base | `src/model/adapter_base.py` | 基类：加载模型、获取目标模块、返回 conditioning | ✅ (model_adapter.md) | onetrainer | 高 | mvp |
| sd15_adapter | `src/model/sd15_adapter.py` | sd1.5 特化：unet + vae + te 加载与条件构造 | 需要（补充） | kohya、lycoris | 高 | mvp |
| sdxl_adapter | `src/model/sdxl_adapter.py` | sdxl 特化：双 te + size_embedding 处理 | 需要（补充） | kohya | 低 | phase_2 |
| **超参数策略层** | | | | | | |
| hyperparam_policy | `src/config/hyperparam_policy.py` | 推荐/约束/校验规则、显存预估、冲突检测 | ✅ (hyperparam_policy.md) | kohya | 高 | mvp |
| presets | `src/config/presets.py` | quick/balanced/quality 三档预设 | 需要（补充） | kohya、onetrainer | 中 | mvp |
| **数据处理** | | | | | | |
| dataset_validator | `src/data/validator.py` | 验证 image+caption 配对、空检查、路径合法性 | ✅ (data_pipeline.md) | flymyai | 高 | mvp |
| bucket_sampler | `src/data/bucket_sampler.py` | aspect-ratio bucketing、batch 采样 | 需要（补充） | kohya | 高 | mvp |
| latent_cache | `src/data/latent_cache.py` | 预先 vae encode、缓存管理 | 需要（补充） | kohya | 中 | mvp |
| **导出层** | | | | | | |
| exporter_base | `src/export/exporter_base.py` | 基类：导出接口 | ✅ (exporter.md) | onetrainer | 中 | mvp |
| comfyui_exporter | `src/export/comfyui_exporter.py` | comfyui 格式导出、元信息写入 | 需要（补充） | flymyai | 中 | mvp |
| a1111_exporter | `src/export/a1111_exporter.py` | a1111 格式导出（可选） | 需要（补充） | kohya | 低 | phase_2 |
| **工具与工程** | | | | | | |
| cli_main | `src/cli/main.py` | cli 单入口、参数解析 | 需要（补充） | kohya | 高 | mvp |
| utils | `src/utils/` | 通用工具（seed/hash/logging/device 等） | 需要（补充） | kohya | 中 | mvp |

## 2. 设计文档补充清单

已有设计文档（✅）：
- recorder_manager.md
- trainer.md
- lora.md
- model_adapter.md
- hyperparam_policy.md
- data_pipeline.md
- exporter.md
- architecture.md

**需要补充的设计文档**：

| 文档名 | 目的 | 优先级 |
| --- | --- | --- |
| config_manager.md | 配置加载、版本管理、默认值策略 | 高 |
| run_manager.md | run 目录规范、快照结构、日志管理 | 高 |
| data_loader.md | 数据加载流程、prefetch、缓存策略 | 高 |
| lora_adapter_impl.md | 标准 lora 的数学原理与实现细节 | 高 |
| sd15_implementation.md | sd1.5 加载与特殊处理 | 中 |
| cli_design.md | 命令行接口设计、参数映射 | 高 |
| presets_design.md | 三档预设的参数推荐逻辑 | 中 |
| error_handling.md | 常见错误捕获、用户友好的提示 | 中 |

## 3. 参考仓库克隆清单

| 仓库 | url | mvp关键参考 | 克隆优先级 |
| --- | --- | --- | --- |
| **kohya-ss/sd-scripts** | https://github.com/kohya-ss/sd-scripts | train_util.py（训练循环）、bucket_sampler（分组采样）、config 管理、latent cache、cli 参数化 | 🔴 高 |
| **LyCORIS** | https://github.com/KohakuBlueleaf/LyCORIS | lora.py（lora 基类）、forward/backward 接口、多算法统一实现 | 🔴 高 |
| **Nerogar/OneTrainer** | https://github.com/Nerogar/OneTrainer | trainer_base.py（生命周期）、checkpoint 管理、组件工厂模式、config 系统 | 🟡 中 |
| **FlyMyAI/flymyai-lora-trainer** | https://github.com/FlyMyAI/flymyai-lora-trainer | validate_dataset（数据校验）、低显存配置、comfyui 导出 | 🟡 中 |
| **edenartlab/sd-lora-trainer** | https://github.com/edenartlab/sd-lora-trainer | pipeline 管理、单内核多入口思想（可选参考） | 🟢 低 |
| **cloneofsimo/lora** | https://github.com/cloneofsimo/lora | lora 基础实现（参考用，不必深入） | 🟢 低 |

## 4. 克隆策略与分工

### 第一优先级（本周完成）
- [ ] 克隆 `kohya-ss/sd-scripts` → 提取 `train_util.py`、`library/utils.py`、bucket 管理逻辑
- [ ] 克隆 `LyCORIS` → 提取 `lora.py` 基类、forward/backward 实现

### 第二优先级（下周）
- [ ] 克隆 `OneTrainer` → 参考生命周期框架
- [ ] 克隆 `FlyMyAI` → 参考数据校验与导出

### 可选参考（后续）
- [ ] edenartlab、cloneofsimo（深度参考收益递减）

## 5. 文件结构建议（项目本体）

```
src/
├── cli/
│   ├── __init__.py
│   ├── main.py                    # 需要设计文档
│   └── parser.py
├── core/
│   ├── __init__.py
│   ├── trainer.py                 # 需要设计文档
│   ├── config_manager.py          # 需要设计文档
│   └── run_manager.py             # 需要设计文档
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── data_loader.py             # 需要设计文档
│   ├── validator.py               # ✅ 见 data_pipeline.md
│   ├── bucket_sampler.py          # 需要设计文档
│   └── latent_cache.py            # 需要设计文档
├── model/
│   ├── __init__.py
│   ├── adapter_base.py            # ✅ 见 model_adapter.md
│   ├── sd15_adapter.py            # 需要设计文档
│   └── sdxl_adapter.py            # phase_2
├── lora.py                        # ✅ 见 lora.md
├── config/
│   ├── __init__.py
│   ├── hyperparam_policy.py       # ✅ 见 hyperparam_policy.md
│   └── presets.py                 # 需要设计文档
├── export/
│   ├── __init__.py
│   ├── exporter_base.py           # ✅ 见 exporter.md
│   ├── comfyui_exporter.py        # 需要设计文档
│   └── a1111_exporter.py          # phase_2
└── utils/
    ├── __init__.py
    ├── device.py
    ├── seed.py
    ├── hash.py
    └── logging.py
```

## 6. 建议行动

1. **本周内**：创建补充设计文档清单（8 个）
2. **克隆参考仓库**：按优先级克隆，存到 `reference/` 目录
3. **代码提取**：逐个仓库提取关键代码片段到设计文档中（保持学习闭环）
4. **实现优先级**：trainer + config_manager + data_loader + lora_adapter（MVP 核心路径）
