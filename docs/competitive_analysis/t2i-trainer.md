## 表 A：产品与架构总览

| 字段 | 内容 |
| --- | --- |
| 项目 | T2ITrainer（多模型族的 LoRA 训练框架/脚本集合，基于 diffusers） [[github.com]](https://github.com/NewBieAI-Lab/NewbieLoraTrainer/blob/main/train_newbie_lora.py), [[github.com]](https://github.com/lrzjason/T2ITrainer) |
| Star / License | 556；AGPL-3.0（LICENSE 文件为 AGPLv3） [[github.com]](https://github.com/lrzjason/T2ITrainer), [[deepwiki.com]](https://deepwiki.com/lrzjason/T2ITrainer/1-t2itrainer-overview) |
| 定位 | DeepWiki 概述为“综合 LoRA 训练框架”，支持 Kolors、Flux、SD3.5 等多架构，并包含修复/编辑类任务（inpainting、object removal、style transfer 等） [[github.com]](https://github.com/NewBieAI-Lab/NewbieLoraTrainer/blob/main/train_newbie_lora.py), [[github.com]](https://github.com/lrzjason/T2ITrainer/blob/main/LICENSE) |
| 产品形态 | 训练脚本族 + 多 UI 入口（含前端目录与多 `ui_*.py`）+ 自动化 setup 脚本（Windows/Linux） [[github.com]](https://github.com/lrzjason/T2ITrainer), [[deepwiki.com]](https://deepwiki.com/lrzjason/T2ITrainer/2-getting-started), [[github.com]](https://github.com/lrzjason/T2ITrainer/blob/main/LICENSE) |

## 表 B：技术架构要点（单表）

| 子系统/机制 | 关键设计 | 关键模块位置（路径） | 对外表现 |
| --- | --- | --- | --- |
| 多模型族统一框架 | DeepWiki 指出支持多架构（Kolors/Flux/SD3.5/HunyuanDiT 等），并用多个训练控制器与 UI 脚本承载 | DeepWiki 架构概述与脚本映射 [[github.com]](https://github.com/NewBieAI-Lab/NewbieLoraTrainer/blob/main/train_newbie_lora.py), [[github.com]](https://github.com/lrzjason/T2ITrainer/blob/main/LICENSE) | 不同模型族按 UI/脚本分入口，但共享“训练框架与数据处理”理念 [[github.com]](https://github.com/NewBieAI-Lab/NewbieLoraTrainer/blob/main/train_newbie_lora.py), [[github.com]](https://github.com/lrzjason/T2ITrainer) |
| 数据处理与缓存 | DeepWiki 提到 create_metadata_cache、CachedImageDataset、BucketBatchSampler 等，用缓存文件（npz/latent/metadata）组织训练输入 | DeepWiki 训练数据流描述 [[github.com]](https://github.com/NewBieAI-Lab/NewbieLoraTrainer/blob/main/train_newbie_lora.py), [[github.com]](https://github.com/lrzjason/T2ITrainer/blob/main/LICENSE) | 适合大规模/多分辨率训练；对高显存模型（Flux/SD3.5）更重要 [[github.com]](https://github.com/NewBieAI-Lab/NewbieLoraTrainer/blob/main/train_newbie_lora.py), [[github.com]](https://github.com/lrzjason/T2ITrainer/blob/main/LICENSE) |
| 依赖与安装体系 | 提供 setup.bat/setup.sh 自动化安装流程（venv、依赖、模型下载），并要求 CUDA 12.1 等环境 | DeepWiki Getting Started + README 安装部分 [[github.com]](https://github.com/lrzjason/T2ITrainer/blob/main/LICENSE), [[deepwiki.com]](https://deepwiki.com/lrzjason/T2ITrainer/2-getting-started) | 更产品化的“开箱即用”训练体验，但环境依赖较重 [[github.com]](https://github.com/lrzjason/T2ITrainer/blob/main/LICENSE), [[deepwiki.com]](https://deepwiki.com/lrzjason/T2ITrainer/2-getting-started) |
| 许可证约束 | LICENSE 文件为 AGPL-3.0 | `LICENSE` 页面内容 [[deepwiki.com]](https://deepwiki.com/lrzjason/T2ITrainer/1-t2itrainer-overview) | 作为服务/闭源产品集成会有强 copyleft 影响，需要在你模板对比中显式标注 [[deepwiki.com]](https://deepwiki.com/lrzjason/T2ITrainer/1-t2itrainer-overview), [[github.com]](https://github.com/lrzjason/T2ITrainer) |

## 表 C：结论

| 分类 | 内容 |
| --- | --- |
| 架构优点 | 多模型族覆盖广（Kolors/Flux/SD3.5 等）且有明确的数据缓存与 bucket 体系；并提供自动化安装脚本与 UI 入口，工程化程度高 [[github.com]](https://github.com/NewBieAI-Lab/NewbieLoraTrainer/blob/main/train_newbie_lora.py), [[github.com]](https://github.com/lrzjason/T2ITrainer/blob/main/LICENSE), [[deepwiki.com]](https://deepwiki.com/lrzjason/T2ITrainer/2-getting-started) |
| 架构不足 | 多入口脚本与 UI 容易形成“入口增殖”；同时 AGPL 许可对生态复用不友好（尤其企业/闭源） [[github.com]](https://github.com/lrzjason/T2ITrainer), [[deepwiki.com]](https://deepwiki.com/lrzjason/T2ITrainer/1-t2itrainer-overview) |
| 可借鉴点 | 1）把“数据缓存 + bucket + 多分辨率”做成核心系统；2）把安装与模型下载流程脚本化，降低上手成本 [[github.com]](https://github.com/lrzjason/T2ITrainer/blob/main/LICENSE), [[deepwiki.com]](https://deepwiki.com/lrzjason/T2ITrainer/2-getting-started) |
| 反模式提醒 | 若你要抽象通用模板，建议把“不同模型族的差异”收敛到 Adapter/Strategy 层，而不是持续增加 `ui_xxx.py`/`train_xxx.py` 脚本数量 [[github.com]](https://github.com/NewBieAI-Lab/NewbieLoraTrainer/blob/main/train_newbie_lora.py), [[github.com]](https://github.com/lrzjason/T2ITrainer) |
