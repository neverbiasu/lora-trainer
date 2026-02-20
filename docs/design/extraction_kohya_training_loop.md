# kohya 训练循环提取（mvp v0.1）

## 参考文件
- `sd-scripts/library/train_util.py` - 核心训练工具
- `sd-scripts/train_network.py` - 网络训练入口
- `sd-scripts/library/config_util.py` - 配置管理

## 1. 训练循环核心结构

### 关键类与方法（train_util.py）

```python
# 参考位置：sd-scripts/library/train_util.py

class NetworkTrainer:
    def train_step(self, batch, **kwargs):
        """训练单步核心流程"""
        # 1. 数据准备
        latents = self.vae.encode(batch["images"])
        text_embeddings = self.text_encoder(batch["input_ids"])
        
        # 2. 噪声采样
        timesteps = torch.randint(...)
        noise = torch.randn_like(latents)
        noisy_latents = add_noise(latents, noise, timesteps)
        
        # 3. UNet 前向（注入 LoRA）
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings)
        
        # 4. Loss 计算
        loss = F.mse_loss(noise_pred, noise)
        
        # 5. 反向传播
        loss.backward()
        
        # 6. Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
```

## 2. 参数体系（BasicTraining）

| 参数类别 | 参数名 | 默认值 | 说明 | mvp 支持 |
| --- | --- | --- | --- | --- |
| **优化器** | | | | |
| optimizer_type | AdamW8bit | adamw | 优化器类型 | ✅ adamw |
| learning_rate | 1e-4 | - | 学习率 | ✅ |
| lr_scheduler | cosine | constant | 学习率调度器 | ✅ cosine |
| lr_warmup_steps | 0 | - | 预热步数 | ✅ |
| **训练控制** | | | | |
| max_train_epochs | None | - | 最大 epoch 数 | ✅ |
| max_train_steps | None | - | 最大步数（优先） | ✅ |
| gradient_accumulation_steps | 1 | - | 梯度累积 | ✅ |
| **显存优化** | | | | |
| mixed_precision | fp16 | no | 混合精度 | ✅ fp16 |
| gradient_checkpointing | False | - | 梯度检查点 | ✅ |
| xformers_memory_efficient_attention | False | - | xformers 优化 | ✅ |
| **LoRA 参数** | | | | |
| network_dim | 4 | - | LoRA 秩（rank） | ✅ (默认32) |
| network_alpha | None | - | LoRA 缩放系数（None=rank） | ✅ |
| network_module | networks.lora | - | 网络模块类型 | ✅ |
| **数据处理** | | | | |
| train_batch_size | 1 | - | 批大小 | ✅ |
| cache_latents | False | - | 缓存 latent | ✅ 自动策略 |
| cache_latents_to_disk | False | - | 磁盘缓存 | ❌ phase_2 |

## 3. 优化器与调度器配置

### Optimizer 工厂模式（参考）

```python
# sd-scripts/library/train_util.py

def get_optimizer(args, trainable_params):
    optimizer_type = args.optimizer_type.lower()
    
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon
        )
    elif optimizer_type == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(...)
    # ...
```

### Scheduler 配置

```python
# cosine / linear / polynomial / constant
scheduler = get_scheduler(
    name=args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps,
    num_training_steps=args.max_train_steps
)
```

## 4. Mixed Precision 处理

```python
# 使用 accelerate 或 torch.cuda.amp
scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision=='fp16')

with torch.cuda.amp.autocast(enabled=args.mixed_precision=='fp16'):
    loss = train_step(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 5. Latent Cache 策略

```python
# sd-scripts/library/train_util.py

class LatentsCachingDataset:
    def __init__(self, dataset, vae, cache_to_disk=False):
        self.cache = {}
        self.cache_to_disk = cache_to_disk
        
        # 预先编码所有图片
        for idx, (image, caption) in enumerate(dataset):
            latent = vae.encode(image)
            if cache_to_disk:
                torch.save(latent, f"cache/{idx}.pt")
            else:
                self.cache[idx] = latent
    
    def __getitem__(self, idx):
        if self.cache_to_disk:
            latent = torch.load(f"cache/{idx}.pt")
        else:
            latent = self.cache[idx]
        return latent, self.dataset[idx][1]  # caption
```

## 6. Bucket 采样器（纵横比）

```python
# sd-scripts/library/train_util.py

class BucketBatchSampler:
    """按纵横比分组采样"""
    
    def __init__(self, dataset, batch_size, bucket_reso_steps=64):
        # 计算每张图的分辨率 bucket
        self.buckets = defaultdict(list)
        
        for idx, (w, h) in enumerate(dataset.resolutions):
            # 取最接近的 bucket
            bucket_w = round(w / bucket_reso_steps) * bucket_reso_steps
            bucket_h = round(h / bucket_reso_steps) * bucket_reso_steps
            self.buckets[(bucket_w, bucket_h)].append(idx)
    
    def __iter__(self):
        # 每个 bucket 内 shuffle，然后分 batch
        for bucket_indices in self.buckets.values():
            random.shuffle(bucket_indices)
            for i in range(0, len(bucket_indices), self.batch_size):
                yield bucket_indices[i:i+self.batch_size]
```

## 7. mvp 实现要点

### 必须实现
1. ✅ 训练循环核心（forward/loss/backward/step）
2. ✅ AdamW 优化器 + cosine 调度器
3. ✅ fp16 混合精度（torch.cuda.amp）
4. ✅ Gradient accumulation
5. ✅ Latent cache（内存缓存优先）
6. ✅ Bucket sampler（按纵横比分组）

### 可选优化（phase_2）
- AdamW8bit（需要 bitsandbytes）
- Gradient checkpointing（低显存模式）
- xformers（需要额外依赖）
- Disk cache（大数据集）

## 8. 关键决策映射

| kohya 参数 | mvp 对应 | 决策 |
| --- | --- | --- |
| optimizer_type | optimizer | 固定 adamw |
| lr_scheduler | scheduler | cosine（可选 linear） |
| network_dim | lora.rank | 默认 32 |
| network_alpha | lora.alpha | 默认等于 rank |
| cache_latents | data.cache_latents | 自动策略（低显存启用） |
| 混合精度 | training.precision | fp16 |
