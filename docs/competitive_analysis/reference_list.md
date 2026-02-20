# 竞品参考清单（含评估置信度）

> 置信度：对“可直接借鉴其设计思路”的信心程度，分数越高越值得优先参考。

| 排名 | 竞品 | 可参考方向 | 关键可借鉴点 | 评估置信度（0-100） |
| --- | --- | --- | --- | --- |
| 1 | Nerogar/OneTrainer | 顶层编排/组件工厂 | 统一生命周期 + 组件工厂装配 + 强配置体系 | 92 |
| 2 | Mikubill/naifu | 单入口 + 配置驱动 | 单入口训练器 + 配置即产品 | 90 |
| 3 | edenartlab/sd-lora-trainer | 单内核多入口 | 单 pipeline 内核 + 多入口薄封装 | 88 |
| 4 | kohya-ss/sd-scripts | 共享训练内核 | 共享核心库（train_util）+ 训练脚本族 | 85 |
| 5 | KohakuBlueleaf/LyCORIS | 算法层可插拔 | 统一算法接口 + 多算法实现 | 84 |
| 6 | FlyMyAI/flymyai-lora-trainer | 数据校验/工具链 | validate_dataset + 低显存路径 | 80 |
| 7 | cloneofsimo/lora | 资产工具链 | merge/convert/提取等工具思路 | 78 |
| 8 | harubaru/waifu-diffusion | 数据/训练分层 | 数据工具与训练解耦 | 76 |
| 9 | replicate/lora-training | 服务接口规范 | 训练服务接口规范化思路（仅参考接口层） | 72 |
| 10 | LarryJane491/Lora-Training-in-Comfy | 生态对接 | ComfyUI 训练节点封装思路 | 68 |
| 11 | lrzjason/T2ITrainer | 数据管线 | cache/bucket 数据管线思路 | 65 |
| 12 | laksjdjf/sd-trainer | 工程化分层 | 目录模块化工程分层 | 60 |
| 13 | NewBieAI-Lab/NewbieLoraTrainer | 配置模板 | TOML 配置模板化思路 | 58 |
