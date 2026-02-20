# Model Adapter 设计（函数职责 + 逐函数参考来源）

## 1. 目标

这份文档用于直接指导 `src/lora_trainer/model_adapter.py` 编码，回答三件事：

- 每个函数做什么
- 每个函数依赖什么输入/状态
- 每个函数参考哪份外部实现

## 2. 类职责边界

### `ModelAdapter`（抽象基类）

- 只定义统一接口，不放 SD1.5 具体实现。
- 保证调用方（Trainer）只依赖抽象接口。

### `SD15ModelAdapter`（SD1.5 实现）

- 实现 `load_models / get_target_modules / encode_prompt / encode_image / decode_latent`。
- 管理 `vae/unet/text_encoder/tokenizer` 生命周期。

## 3. 函数关系图

```text
Trainer
  -> SD15ModelAdapter.load_models()
      -> _is_checkpoint(path)
      -> load_checkpoint_with_text_encoder_conversion()   # checkpoint 分支
      -> (or) StableDiffusionPipeline.from_pretrained()  # diffusers 分支

Trainer
  -> SD15ModelAdapter.get_target_modules()
  -> SD15ModelAdapter.encode_prompt()
  -> SD15ModelAdapter.encode_image()
  -> SD15ModelAdapter.decode_latent()
```

## 4. 逐函数设计 + 参考来源

> 说明：下表“主参考”是首选实现依据；“补充参考”用于工程兼容和边界情况。

| 函数 | 职责 | 输入/输出 | 主参考来源 | 补充参考来源 |
| --- | --- | --- | --- | --- |
| `load_checkpoint_with_text_encoder_conversion` | 读取 checkpoint 并修正 text encoder key 命名 | `ckpt_path, device -> (checkpoint, state_dict)` | `docs/references/sd-scripts/library/model_util.py`（ckpt/safetensors 读取与 key 兼容思路） | `docs/references/OneTrainer/modules/modelLoader/stableDiffusion/StableDiffusionModelLoader.py` |
| `ModelAdapter.__init__` | 初始化模型标识与 device | `model_name_or_path -> None` | `docs/references/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`（device/dtype 迁移思路） | - |
| `ModelAdapter.load_models` | 抽象方法，仅定义契约 | `-> tuple[nn.Module, ...]` | 本项目接口设计约束 | - |
| `ModelAdapter.get_target_modules` | 抽象方法，定义 LoRA 目标层接口 | `-> list[str]` | `docs/references/sd-scripts/networks/lora.py`（目标层命名惯例） | OneTrainer LoRA 目标策略 |
| `ModelAdapter.encode_prompt` | 抽象方法，定义文本编码接口 | `list[str] -> Tensor` | `pipeline_stable_diffusion.py` 的 `encode_prompt` 语义 | - |
| `ModelAdapter.encode_image` | 抽象方法，定义图像编码接口 | `Tensor -> Tensor` | `AutoencoderKL` 编码语义 | - |
| `ModelAdapter.decode_latent` | 抽象方法，定义 latent 解码接口 | `Tensor -> Tensor` | `AutoencoderKL` 解码语义 | - |
| `ModelAdapter._is_checkpoint` | 判断输入是否为 checkpoint 文件 | `Path -> bool` | 本项目格式支持约束 | - |
| `SD15ModelAdapter.load_models` | SD1.5 组件加载主流程（checkpoint 与 diffusers 两分支） | `-> (vae, unet, text_encoder)` | `docs/references/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`、`docs/references/diffusers/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py` | OneTrainer loader 组织方式 |
| `SD15ModelAdapter.get_target_modules` | 返回 LoRA 注入目标（`to_q/to_k/to_v/to_out.0`） | `-> list[str]` | `sd-scripts` LoRA 目标命名 | OneTrainer 目标层覆盖策略 |
| `SD15ModelAdapter.encode_prompt` | tokenizer + text_encoder 得到 prompt embedding | `list[str] -> Tensor` | `pipeline_stable_diffusion.py` `encode_prompt` | - |
| `SD15ModelAdapter.encode_image` | VAE 编码并应用 `scaling_factor` | `images -> latents` | `docs/references/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py` | - |
| `SD15ModelAdapter.decode_latent` | latent 反缩放后 VAE 解码 | `latents -> images` | `autoencoder_kl.py` + SD pipeline 解码语义 | - |
| `SD15ModelAdapter._ensure_loaded` | 延迟加载保护，确保句柄可用 | `-> None` | 本项目工程约束（lazy-load） | OneTrainer 延迟初始化思路 |

## 5. `SD15ModelAdapter.load_models` 分支细化

### 5.1 checkpoint 分支

1. 读取并转换 state_dict
2. `create_unet_diffusers_config(..., image_size=512)` + `convert_ldm_unet_checkpoint`
3. `create_vae_diffusers_config(..., image_size=512)` + `convert_ldm_vae_checkpoint`
4. `convert_ldm_clip_checkpoint`
5. 构造并加载 `unet/vae/text_encoder`
6. 初始化 tokenizer（MVP 采用 `openai/clip-vit-large-patch14`）

### 5.2 diffusers 分支

1. `StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=...)`
2. `pipe.to(device)`
3. 提取 `pipe.vae/pipe.unet/pipe.text_encoder/pipe.tokenizer`

## 6. 实现顺序（实际编码优先级）

1. 先把 `ModelAdapter` 恢复为纯抽象接口。
2. 完成 `SD15ModelAdapter.load_models` 两个分支。
3. 完成 `get_target_modules`。
4. 完成 `encode_prompt`。
5. 完成 `encode_image/decode_latent`。
6. 加 `_ensure_loaded` 做运行时保护。

## 7. 验收标准

- `ModelAdapter` 不包含 SD1.5 实现细节。
- `SD15ModelAdapter.load_models` 两分支都能返回完整三元组。
- `encode_prompt/encode_image/decode_latent` 在模型已加载后可直接调用。
- 文档中每个关键函数都有明确参考来源。
# Model Adapter 设计（函数级可落地版）

## 1. 目标与边界

本设计只回答一件事：`src/lora_trainer/model_adapter.py` 里的每个函数**要做什么、依赖谁、返回什么、先写哪一个**。

MVP 边界：

- 支持 SD1.5 的基础加载路径。
- 支持两类模型来源：checkpoint（`.safetensors/.pt/.pth`）与 diffusers 目录/Hub。
- 为 LoRA 训练提供最小能力：模型加载、目标模块列表、prompt/image 编解码接口。

非 MVP：

- SDXL、双编码器、分布式、量化训练。

## 2. 与代码文件的映射

实现文件：`src/lora_trainer/model_adapter.py`

本文件里当前存在两层抽象：

1. 顶层工具函数（checkpoint 读入与 key 兼容）
2. `ModelAdapter` 基类 + `SD15ModelAdapter` 具体实现

## 3. 函数关系总览（调用链）

```text
Trainer/调用方
  -> ModelAdapter.load_models()
      -> _is_checkpoint(path)
      -> [if checkpoint]
           load_checkpoint_with_text_encoder_conversion()
           create_*_config / convert_*_checkpoint
           build vae/unet/text_encoder
         [else]
           StableDiffusionPipeline.from_pretrained(...)
      -> return (vae, unet, text_encoder)

Trainer/调用方
  -> get_target_modules()      # LoRA 注入位点
  -> encode_prompt(prompts)    # 文本编码
  -> encode_image(images)      # 图像->latent
  -> decode_latent(latents)    # latent->图像
```

## 4. 函数逐一设计（可直接照着写）

### 4.1 `load_checkpoint_with_text_encoder_conversion`

职责：

- 读取 checkpoint/safetensors。
- 将旧 key 结构转换成 `text_model.*` 结构，保证可被 CLIPTextModel 正常加载。

输入：

- `ckpt_path: str`
- `device: torch.device`

输出：

- `tuple[checkpoint_obj | None, state_dict]`

前置条件：

- 路径存在且可读。

失败策略：

- 文件不存在、格式异常、torch 反序列化失败时抛异常，不静默处理。

### 4.2 `ModelAdapter.__init__`

职责：

- 保存模型路径。
- 选择运行设备（`cuda` 优先，否则 `cpu`）。

输入：

- `model_name_or_path: str`

输出：

- 无返回值，初始化 `self.model_name_or_path` 与 `self.device`。

### 4.3 `ModelAdapter._is_checkpoint`

职责：

- 判断路径是否为 checkpoint 文件后缀。

输入/输出：

- 输入 `Path`，输出 `bool`。

规则：

- `suffix in {".safetensors", ".pt", ".pth"}`。

### 4.4 `ModelAdapter.load_models`

职责：

- 统一加载并返回 `(vae, unet, text_encoder)`。
- 屏蔽 checkpoint 与 diffusers 两条来源差异。

输入：

- 无显式参数，使用 `self.model_name_or_path` 与 `self.device`。

输出：

- `tuple[nn.Module, ...]`，固定顺序：`vae, unet, text_encoder`。

关键约束：

- 所有分支都必须 `return` 或 `raise`，不能隐式返回 `None`。
- 模块实例化与 `.to(device=...)` 建议分两句写，避免类型检查器误判。
- checkpoint 分支的配置构造必须传入必需参数（如 `image_size`）。

分支 A（checkpoint）：

1. 读 `state_dict`
2. 构造 `unet_config` 并转换权重
3. 构造 `vae_config` 并转换权重
4. 构造 `clip_config` 并转换权重
5. 加载权重 + 日志
6. 返回三元组

分支 B（diffusers）：

1. `StableDiffusionPipeline.from_pretrained(...)`
2. `pipe.to(device=...)`
3. 返回 `pipe.vae, pipe.unet, pipe.text_encoder`

### 4.5 `ModelAdapter.get_target_modules`

职责：

- 返回 LoRA 注入目标层名列表。

MVP 规则：

- 先覆盖 UNet cross-attn：`to_q/to_k/to_v/to_out.0`。
- 可选再扩 self-attn。

输出：

- `list[str]`，元素是模块名匹配片段。

### 4.6 `ModelAdapter.encode_prompt`

职责：

- 将 `prompts` 编码为文本 embedding。

输入：

- `list[str]`

输出：

- `torch.Tensor`（batch, seq, hidden）

依赖：

- `tokenizer` + `text_encoder`。

### 4.7 `ModelAdapter.encode_image`

职责：

- 图像张量编码到 latent 空间。

输入：

- `images: torch.Tensor`，归一化区间由调用方约定。

输出：

- `latents: torch.Tensor`

### 4.8 `ModelAdapter.decode_latent`

职责：

- latent 解码回图像张量。

输入/输出：

- 输入 latent，输出图像 tensor。

### 4.9 `SD15ModelAdapter.*`

职责：

- 对 `ModelAdapter` 抽象进行 SD1.5 的具体化。
- 所有 `NotImplementedError` 对应函数都应优先复用基类能力，不重复造加载逻辑。

建议：

- `SD15ModelAdapter.load_models()` 先调用 `super().load_models()`，再缓存到 `self.vae/self.unet/self.text_encoder`。

## 5. 最小实现顺序（按这个顺序写，最稳）

1. 先修正 `ModelAdapter.load_models()` 的类型与分支返回一致性。
2. 实现 `SD15ModelAdapter.load_models()`（包装基类并缓存句柄）。
3. 实现 `get_target_modules()`（先返回最小可用列表）。
4. 实现 `encode_prompt()`（tokenize + text_encoder）。
5. 实现 `encode_image()/decode_latent()`（VAE 编解码）。

这样可以最快打通：`load -> inject LoRA -> train_step`。

## 6. 与其它模块的协作契约

调用方（未来 `Trainer`）只依赖以下最小接口：

- `load_models()`：拿到三大组件
- `get_target_modules()`：注入 LoRA
- `encode_prompt()/encode_image()/decode_latent()`：训练与采样

不要把优化器、调度器、训练步逻辑放进 adapter。

## 7. 参考实现来源（写代码时优先看）

主参考：

- `docs/references/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`
- `docs/references/diffusers/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py`

工程分支参考：

- `docs/references/OneTrainer/modules/modelLoader/stableDiffusion/StableDiffusionModelLoader.py`
- `docs/references/sd-scripts/library/model_util.py`

原则：

- M3 先以 diffusers 主路径跑通。
- 兼容分支（复杂 checkpoint）再借鉴 OneTrainer/sd-scripts。

## 8. 完成判定（Definition of Done）

满足以下条件即可判定 adapter 设计可落地：

- 每个公开函数都能回答“输入/输出/副作用/失败策略”。
- 调用方不需要了解 checkpoint 与 diffusers 的内部差异。
- 函数命名与 `src/lora_trainer/model_adapter.py` 保持一致。
- 默认冻结基础模型参数。
- 提供 UNet 注入 LoRA 所需的目标模块列表。
- 提供文本与图像的编码/解码接口。
- 提供可选显存优化开关（如 gradient checkpointing）。

### M3 不包含

- SDXL 双文本编码器支持。
- 多骨干混训。
- 量化加载与分布式加载。

## 3. 职责

| 领域 | 职责 |
| --- | --- |
| 模型加载 | 统一构建并持有 `unet / vae / text_encoder / tokenizer / scheduler` |
| 冻结策略 | 约束可训练边界（默认仅 LoRA 可训练） |
| 能力暴露 | 返回目标模块列表与运行能力信息 |
| 运行转换 | 处理 prompt -> embedding、image <-> latent 转换 |

## 4. 对外接口（API 签名）

```text
class ModelAdapter:
    init(config: ModelConfig)
    load() -> ModelHandles
    get_lora_targets() -> list[str]
    encode_prompt(prompts: list[str]) -> Tensor
    encode_image(images: Tensor) -> Tensor
    decode_latent(latents: Tensor) -> Tensor
    apply_memory_optimizations() -> None
```

## 5. 数据契约

### 输入

- `base_model_identifier`：HF 仓库名或本地路径
- `device`：`cpu | cuda | mps`
- `dtype`：`fp32 | fp16 | bf16`
- `train_text_encoder`：是否训练文本编码器
- `gradient_checkpointing`：是否启用梯度检查点

### 输出

- `ModelHandles`：
  - `unet`
  - `vae`
  - `text_encoder`
  - `tokenizer`
  - `noise_scheduler`

## 6. 运行流程

```text
解析配置
  -> 加载组件
  -> 冻结基础参数
  -> 按需启用显存优化
  -> 暴露 handles 与 target 列表
```

### 伪代码

```text
load():
  load vae/unet/text_encoder/tokenizer/scheduler
  set eval/train modes by policy
  freeze all base params
  if train_text_encoder: unfreeze text_encoder
  return handles
```

## 7. 模型加载逻辑（实现摘要）

本节定义 M3 阶段应落地的“模型如何加载”标准流程，作为 `src/lora_trainer/model_adapter.py` 的实现依据。

### 7.1 加载入口与来源判定

- 统一入口：`load_models()`。
- 输入 `model_name_or_path` 支持两种来源：
  - 本地目录（存在路径）；
  - Hugging Face Hub 模型标识。
- 来源判定失败时直接报错，禁止静默回退。

### 7.2 分组件加载顺序

推荐按以下顺序加载并注册句柄：

1. `tokenizer`
2. `text_encoder`
3. `vae`
4. `unet`
5. `scheduler`

说明：

- 加载后立即统一迁移到目标 `device` 与 `dtype`。
- 句柄必须全部可用后才返回，禁止返回部分成功状态。

### 7.3 配置修正与兼容策略

参考 diffusers `StableDiffusionPipeline` 的兼容处理，M3 建议在加载时执行以下校验：

- `scheduler.config.steps_offset != 1` 时修正为 `1` 并记录 warning。
- `scheduler.config.clip_sample == True` 时修正为 `False` 并记录 warning。
- 旧版 UNet 且 `sample_size < 64` 时修正为 `64` 并记录 warning。

注意：

- 所有“修正”都必须记录到 run snapshot，保证可追溯。
- 若启用 strict 模式，可将上述 warning 升级为 error。

### 7.4 模式与冻结策略

- `vae`：`eval()` + `requires_grad_(False)`。
- `unet`：`train()`（仅为 LoRA 训练准备）+ 基础权重 `requires_grad_(False)`。
- `text_encoder`：默认 `eval()` + `requires_grad_(False)`；仅在 `train_text_encoder=True` 时解冻。

### 7.5 运行前一致性校验

返回前必须完成以下检查：

- 核心组件不为 `None`。
- `unet.config.in_channels` 与训练配置一致。
- `vae_scale_factor` 可正确推导（通常由 `vae.config.block_out_channels` 推导）。
- `scheduler` 已拥有有效 timesteps 设置能力。

### 7.6 文本与图像处理契约

- 文本编码：
  - 由 `tokenizer(max_length, truncation, padding)` 产出 `input_ids`；
  - 由 `text_encoder` 产出 prompt embedding；
  - 长文本截断时必须写 warning。
- 图像编码：
  - 输入范围统一到 `[-1, 1]`；
  - 经 VAE 编码得到 latent；
  - 应用 `scaling_factor`（优先读取配置，不写死常量）。

### 7.7 伪代码（加载主流程）

```text
load_models():
  source = detect_local_or_hub(model_name_or_path)
  tokenizer = load_tokenizer(source)
  text_encoder = load_text_encoder(source, dtype, device)
  vae = load_vae(source, dtype, device)
  unet = load_unet(source, dtype, device)
  scheduler = load_scheduler(source)

  normalize_scheduler_config_if_needed(scheduler)
  normalize_unet_config_if_needed(unet)

  set_modes_and_freeze(vae, unet, text_encoder, train_text_encoder)
  validate_handles(vae, unet, text_encoder, tokenizer, scheduler)
  return handles
```

### 7.8 借鉴来源优先级（按组件）

本项目 M3 推荐采用“diffusers 做主干 + OneTrainer/sd-scripts 补工程能力”的策略。

| 组件/能力 | 主借鉴（优先） | 次借鉴（补充） | 结论 |
| --- | --- | --- | --- |
| UNet 加载 | `docs/references/diffusers/src/diffusers/models/unets/unet_2d_condition.py` + `.../pipelines/stable_diffusion/pipeline_stable_diffusion.py` | `docs/references/OneTrainer/modules/modelLoader/stableDiffusion/StableDiffusionModelLoader.py` | M3 先用 diffusers `UNet2DConditionModel.from_pretrained(..., subfolder="unet")`，OneTrainer 仅借鉴 dtype/量化与多来源分支 |
| VAE 加载 | `docs/references/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py` + `.../pipeline_stable_diffusion.py` | `docs/references/sd-scripts/library/model_util.py`（`load_vae` 思路） | M3 主路径用 diffusers `AutoencoderKL.from_pretrained(..., subfolder="vae")`；额外 VAE 覆盖能力可参考 sd-scripts |
| Text Encoder/Tokenizer | `.../pipeline_stable_diffusion.py`（`encode_prompt`、截断 warning、attention_mask） | `docs/references/OneTrainer/modules/model/StableDiffusionModel.py`（clip_skip/层输出策略） | M3 先实现标准 CLIP 文本编码；进阶再加 OneTrainer 的 layer-skip 策略 |
| Scheduler 加载与修正 | `.../pipeline_stable_diffusion.py`（`steps_offset`、`clip_sample` 修正） | `docs/references/OneTrainer/modules/modelLoader/stableDiffusion/StableDiffusionModelLoader.py`（二次封装） | M3 必须保留 diffusers 的配置修正逻辑，避免历史配置导致行为偏差 |
| ckpt/safetensors 兼容 | `docs/references/sd-scripts/library/model_util.py`（格式转换与兼容） | `docs/references/OneTrainer/modules/modelLoader/stableDiffusion/StableDiffusionModelLoader.py`（`download_from_original_stable_diffusion_ckpt`） | 非 MVP 必需，但若要支持老权重，优先复用现成转换路径，不自写转换器 |

### 7.9 你当前最该“抄”的文件（M3 直用）

按实现收益排序：

1. `docs/references/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`
   - 抄：组件组装顺序、兼容修正、`encode_prompt` 行为契约。
2. `docs/references/diffusers/src/diffusers/models/unets/unet_2d_condition.py`
   - 抄：UNet 类型与配置边界（不抄内部网络实现）。
3. `docs/references/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py`
   - 抄：VAE 编解码入口、scaling factor 契约。
4. `docs/references/OneTrainer/modules/modelLoader/stableDiffusion/StableDiffusionModelLoader.py`
   - 抄：工程化加载分支（目录模型、ckpt、safetensors）、可选量化入口。
5. `docs/references/sd-scripts/library/model_util.py`
   - 抄：旧格式权重兼容与 VAE 覆盖加载策略。

### 7.10 M3 组件落地决策（明确回答）

- UNet 借鉴哪里：
  - 首选 `diffusers` 的 `UNet2DConditionModel` 加载与配置；
  - 工程分支（ckpt/safetensors）借鉴 OneTrainer。
- VAE 借鉴哪里：
  - 首选 `diffusers` 的 `AutoencoderKL` 与 pipeline 编解码路径；
  - 需要“额外 VAE 覆盖加载”时借鉴 sd-scripts 的 `load_vae` 方案。
- 为什么不是全抄 OneTrainer/sd-scripts：
  - 两者工程能力强，但路径更重、抽象层更深；
  - M3 目标是最小可复现闭环，diffusers 主干更稳、更短、更易维护。

### 7.11 模型格式转换参考代码（可直接借鉴）

本节给出“checkpoint / safetensors <-> diffusers”转换的现成实现位置，避免重复造轮子。

#### A. Diffusers 官方转换路径（首选）

- `docs/references/diffusers/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py`
  - 关键入口：`download_from_original_stable_diffusion_ckpt`
  - 适用：原始 `.ckpt/.safetensors` 转成可直接训练/推理的 pipeline 组件。
- `docs/references/diffusers/src/diffusers/loaders/single_file.py`
  - 关键入口：`from_single_file`
  - 适用：单文件模型加载为 diffusers 组件，工程接入最短。

#### B. OneTrainer 工程化落地（强参考）

- `docs/references/OneTrainer/modules/modelLoader/stableDiffusion/StableDiffusionModelLoader.py`
  - 关键方法：`__load_ckpt`、`__load_safetensors`
  - 转换调用：`download_from_original_stable_diffusion_ckpt(..., from_safetensors=True/False)`
  - 适用：多来源加载分支如何组织（diffusers 目录 / ckpt / safetensors）。

#### C. sd-scripts 双向转换（补充）

- `docs/references/sd-scripts/library/model_util.py`
  - 关键函数：
    - `convert_ldm_unet_checkpoint`
    - `convert_ldm_vae_checkpoint`
    - `save_stable_diffusion_checkpoint`
    - `save_diffusers_checkpoint`
  - 适用：需要做“回写 checkpoint”或对转换细节做深度可控时。

#### M3 推荐实现顺序

1. 输入 checkpoint/safetensors：优先走 diffusers 官方转换入口（A）。
2. 内部训练格式：统一为 diffusers 组件对象（`unet/vae/text_encoder/tokenizer/scheduler`）。
3. 导出阶段：默认导出 LoRA safetensors；如需全模型 checkpoint，再引入 C 的回写能力。

#### 约束

- M3 不建议自研权重映射转换器。
- 若官方转换失败，再走 sd-scripts 深度转换函数兜底，并记录告警与转换来源。

## 8. LoRA 目标策略（SD1.5）

### 默认目标族

- `attn2.to_q / to_k / to_v / to_out.0`（优先跨注意力）
- `attn1.to_q / to_k / to_v / to_out.0`（可选自注意力）

### 规则

- M3 默认包含 `attn2.*` 与 `attn1.*`。
- Feed-forward 目标默认关闭，按需启用。
- 目标匹配必须可复现、可记录。

## 9. 失败处理

- 模型加载失败：抛出可恢复配置/运行时错误，并给出路径或仓库提示。
- dtype 与 device 组合不支持：走降级策略或在严格模式下直接失败。
- 组件缺失：提前终止，不进入部分可运行状态。

## 10. 可观测性

在 `INFO` 级别记录以下字段：

- `base_model_identifier`
- `device`
- `dtype`
- `train_text_encoder`
- `gradient_checkpointing`
- `target_module_count`

同时建议记录：

- `source_type`（local/hub）
- `scheduler_config_patched`（是否修正过配置）
- `unet_config_patched`（是否修正过配置）
- `vae_scale_factor`

## 10.5 `SD15ModelAdapter.generate`（推理采样）

### 职责

使用已加载的训练组件临时组装 `StableDiffusionPipeline`，完成推理采样，供 `Trainer.validate()` 调用。

### 为什么不手写去噪循环

sd-scripts / OneTrainer 均采用 pipeline 封装方式，原因：

- `scheduler.init_noise_sigma`、CFG 拼接顺序等细节在不同调度器间行为不同，pipeline 已内部处理。
- 混精度 dtype 对齐、VAE decode 后处理均已封装。
- 换调度器只需换 `scheduler=` 参数，不改循环代码。

### 输入 / 输出

```python
def generate(
    self,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
) -> torch.Tensor:
    ...
```

- 输出：`torch.Tensor`，形状 `[3, H, W]`，值域 `[0, 1]`，CPU。

### 实现要点

1. `SDPipeline(vae, unet, text_encoder, tokenizer, scheduler=DDIMScheduler(...), safety_checker=None)`
2. `pipe.to(device)` → 调用 → `del pipe`（调用后立即释放，避免显存驻留）
3. 固定 seed：`torch.Generator(device=device).manual_seed(seed)`
4. `TF.to_tensor(result.images[0])` 将 PIL 转为 `[3,H,W]` float tensor

### 参考来源

- `docs/references/sd-scripts/library/train_util.py`（`sample_image_inference` L6489-L6560）

---

## 11. 验收标准（M3）

- 可从配置来源成功加载 SD1.5 并返回完整 `ModelHandles`。
- 除显式允许模块外，基础参数全部被冻结。
- 目标模块列表非空且跨运行稳定。
- 文本/图像转换在 512 管线下无 shape mismatch。

补充验收：

- 当给定旧配置时，兼容修正逻辑可触发且可追踪。
- 本地路径与 Hub 两种来源都可成功加载。

## 12. 后续扩展（M4+）

- 增加 SDXL 与自定义骨干的适配协议。
- 增加 xformers/flash-attn 等策略开关。
- 增加更严格的导出兼容能力报告。
