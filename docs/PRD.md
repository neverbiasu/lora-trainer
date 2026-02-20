# PRD 初稿（v0.1）

**项目代号**：LoRA Trainer CLI（暂定名）
**版本**：v0.1（MVP 设计稿）
**定位**：CLI‑first、极简接口、面向训练人员/入门但有基础用户，同时支持训练工程师/研究者的可复现与扩展。
**License**：AGPL‑3.0（初期），支持 Sponsor；未来如需更宽松许可，可考虑双许可证/商业许可。 [gitcode.com]

---

## 1. 背景与问题

### 1.1 背景

当前 LoRA 训练生态呈现两类极端：

- **脚本集合型**：能力全但入口多、参数巨大、学习曲线陡（典型：sd-scripts 的“数百参数”与多脚本入口）。 [kblueleaf.net], [github.com]
- **产品化型**：体验好但系统较重、许可证可能限制集成（典型：OneTrainer 的强配置系统与训练编排器，AGPL）。 [awesome.ecosyste.ms], [runcomfy.com], [gitcode.com]

本项目希望在两者之间做一个“极简但不简陋”的 CLI‑first 训练器：

- 让有基础的用户“看得懂、跑得通”。
- 同时给工程师/研究者“可复现、可扩展、可研究 LoRA 的数理与工程实现”。

### 1.2 目标（North Star）

在不做服务、不做数据采集、不做预处理、不做推理 UI 的前提下：

- 用户只需要准备 image+caption 配对数据集，提供一个配置文件，即可完成 train → 训练中采样 → 导出 LoRA 的闭环。
- CLI 启动成本做到“一个命令 + 一个配置文件”。
- 结构上必须支持后续扩展模型族/算法族，而不是靠复制脚本。

---

## 2. 目标用户与使用场景

### 2.1 目标用户

**模型训练人员 / 初学者（有一定基础）**

- 目标：快速训练 LoRA，能理解每个参数的含义与影响范围。
- 痛点：参数太多、步骤太多、失败成本高。

**训练工程师 / 研究者**

- 目标：可复现、可控、可扩展（加新算法/新模型/新损失）。
- 痛点：脚本堆叠导致可维护性差，实验记录不一致。

### 2.2 典型场景

- 单概念/单风格 LoRA 训练（MVP）。
- 训练过程可观测（loss 曲线、固定 prompt 采样）。
- 导出到生态工具（不做推理 UI，但训练产物能用；参考 eden 强调导出兼容外部工具的思路）。 [replicate.com], [microsoft.github.io]

---

## 3. 产品原则与非目标

### 3.1 产品原则

- **极简启动**：默认只需 `train -c config`；额外 CLI 参数只做覆盖，不做“全参数 CLI 化”。（避免 sd-scripts 的参数爆炸） [kblueleaf.net], [github.com]
- **可复现优先**：每个 run 必须自动落盘 config snapshot、版本信息、seed、数据集摘要。
- **扩展点先行**：先定义接口/抽象，再实现 MVP；避免 Bottom‑up 增殖。参考 OneTrainer 的“编排器+工厂组件化”思想。 [awesome.ecosyste.ms], [deepwiki.com]
- **不做预处理但做校验**：不 resize/打标/抠图，但必须提供 dataset validation（FlyMyAI 的 validate_dataset 是很好的产品化点）。 [deepwiki.com], [civitai.com]

### 3.2 非目标（明确不做）

- 不做数据采集。
- 不做预处理/打标/抠图（可由另一个产品承担）。
- 不做推理 UI（但支持导出与最小可用的 sample 生成用于训练中验证）。
- 不做训练服务（可提供镜像/部署方案，但不提供托管服务；不走 replicate 那种 service 封装路线）。 [docs.flymy.ai], [deepwiki.com]

---

## 4. 端到端用户流程（MVP）

目标：流程“清晰、短、可解释”。

### 4.1 准备数据集（MVP 约定）

- `dataset/xxx.png` + `dataset/xxx.txt` 一一对应。
- 允许子目录（可选，MVP 可先不支持）。

### 4.2 validate（训练前校验）

- 检查图片/文本配对、扩展名、空 caption、重复文件名、路径合法性。
- 输出：可读报告 + 可机器解析 JSON（方便 CI/脚本）。

### 4.3 train -c config.yaml

- 自动创建 workspace（run 目录）。
- 训练中：定期保存 checkpoint（可选）、定期生成 sample（固定 prompt+seed）。
- 训练结束：导出 LoRA（+ 元信息）。

### 4.4 export（可选独立步骤，MVP 可以和 train 绑定）

- 导出目标先选 1 个，并在 PRD 中声明优先级。
- 产物携带训练信息（base model、rank、alpha、trigger 等）。

---

## 5. 功能需求（MVP / V1）

### 5.1 MVP（必须有）

**A. CLI 命令集（最小集合）**

- `trainer validate -c config.yaml`
- `trainer train -c config.yaml`
- `trainer export -c config.yaml`（可选，或作为 train 的最后一步）

**B. 配置系统（先极简）**

- 支持 YAML（先定 YAML，后续可加 TOML）。
- Schema：类型、默认值、范围、说明（至少在文档与运行时校验里体现）。
- 版本号：`config_version`（为未来迁移留口）。
- 参考 OneTrainer 的 BaseConfig 思路，但 MVP 保持轻量。 [runcomfy.com], [awesome.ecosyste.ms]

**C. Workspace（Run 管理）**

目录结构（建议）：

- `runs/<run_id>/config.yaml`（snapshot）
- `runs/<run_id>/logs/`（日志/指标）
- `runs/<run_id>/samples/`（训练中采样）
- `runs/<run_id>/checkpoints/`（可选）
- `runs/<run_id>/export/`（最终 LoRA）

**D. 训练中采样（不等于推理 UI）**

- 固定 prompts（来自 config）。
- 固定 seeds。
- 采样频率可配。

**E. 扩展点接口（MVP 也要定义）**

- ModelAdapter（MVP 实现 1 个：例如 SD1.5）。
- LoRAAdapter（MVP 实现：标准 LoRA）。
- Exporter（MVP 实现 1 个导出目标）。
- DataSpec（MVP：image+caption）。

### 5.2 V1（验证扩展性）

- 接入其他 LoRA 变体（例如 DoRA 或 LoKr；可选对接 LyCORIS 的算法层——LyCORIS 的定位就是"算法层可插拔库"）。 [huggingface.co], [deepwiki.com]
- 接入第二个 ModelAdapter（例如 FLUX 或 Qwen，二选一）。
- 增强数据层（比如支持 mask/control 的目录约定，但仍不做预处理）。

---

## 6. 技术架构（U 型：Top-down core + Bottom-up tools）

### 6.0 顶层设计（U 型）

- **Top-down（核心骨架）**：先定义稳定的生命周期与接口边界，保证最小闭环可复现。
	- 统一入口与生命周期（`TrainerCore`）
	- 组件装配与注册（`ComponentFactory/Registry`）
	- 配置与运行态（`ConfigManager`、`RunManager`）
- **Bottom-up（工具与能力）**：从实际训练痛点提炼工具，并以接口方式接入核心骨架。
	- 数据校验与诊断（`validate_dataset`、错误提示）
	- 采样、导出、转换等工具链
- **U 型闭环**：核心骨架提供可扩展“插槽”，工具能力反哺设计抽象与配置规范，避免脚本堆叠。

### 6.1 核心（Top-down，先做）

- TrainerCore：统一生命周期（start/train/eval/export/end）。
- ComponentFactory/Registry：根据 config 选择 ModelAdapter/LoRAAdapter/Exporter。
- ConfigManager：加载/校验/默认值/版本号。
- RunManager：workspace 创建、snapshot、日志、产物落盘。
- 这条路线对应 OneTrainer 的“编排器+工厂装配”，但实现更轻量。 [awesome.ecosyste.ms], [deepwiki.com]

### 6.2 外围（Bottom-up，可选工具）

- dataset validator（强烈建议 MVP 就做）。 [civitai.com], [deepwiki.com]
- merge/extract/convert 工具（后续再加；LyCORIS 与 cloneofsimo/lora 都把工具链当重要能力）。 [huggingface.co], [deepwiki.com]
- 镜像/部署脚本（Dockerfile/conda/uv/poetry 多方案）。

---

## 7. 非功能需求（NFR）

- **可复现性**：run 必须记录 config、seed、版本信息、数据摘要。
- **可观测性**：最低限度本地日志 + 可选 TensorBoard（不是 UI，是日志输出）。
- **可维护性**：Adapter 接口稳定；新增模型/算法不复制脚本。
- **可移植性**：至少 Linux；Windows 可选。
- **安全**：不上传数据；本地路径处理与日志脱敏（如果用户愿意分享 run）。

---

## 8. 里程碑（建议）

- M0：骨架跑通（Config → validate → train loop stub → run manager）。
- M1：MVP 训练 SD1.5 标准 LoRA + 训练中采样 + 导出一种格式。
- M2：扩展点验证（第二算法或第二模型）。
- M3：工具链与更强的复现/诊断。
