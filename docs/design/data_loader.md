# data_loader 设计

## 1. 职责边界

DataLoader 负责：
- **数据集校验**：检查图片-文本对、分辨率、格式
- **Bucket 管理**：按宽高比分组，减少 padding 浪费
- **Latent Cache**：预计算 VAE latent，加速训练
- **Batch 构造**：从 bucket 采样，确保 batch 内宽高一致
- **Prefetch 优化**：异步加载下一个 batch

DataLoader **不负责**：
- 数据增强（MVP 不做 flip/crop）
- Caption 生成（依赖用户提供 .txt）
- 模型加载（由 ModelAdapter 处理）

---

## 2. 核心接口

```python
class DataLoaderManager:
    """数据加载统一入口"""
    
    def __init__(
        self,
        dataset_path: str,
        resolution: int,           # 目标分辨率（如 512）
        batch_size: int,
        enable_bucketing: bool = True,  # MVP 默认 True
        cache_latents: bool = False,    # 由 HyperparamPolicy 控制
        num_workers: int = 4
    ):
        pass
    
    def validate_dataset(self) -> ValidationResult:
        """校验数据集，返回错误列表"""
        pass
    
    def prepare_buckets(self) -> List[Bucket]:
        """构建 bucket 索引"""
        pass
    
    def cache_latents_if_needed(self, vae) -> None:
        """如果启用，预计算所有 latent"""
        pass
    
    def get_dataloader(self) -> DataLoader:
        """返回 PyTorch DataLoader"""
        pass
```

---

## 3. 数据集校验（validate_dataset）

### 3.1 校验规则（参考 flymyai-lora-trainer）

| 检查项               | 规则                                      | 错误级别 |
|---------------------|------------------------------------------|---------|
| 图片-文本配对        | 每个 .png/.jpg 必须有对应 .txt            | ERROR   |
| 图片可读性           | 能被 PIL.Image.open 打开                  | ERROR   |
| 最小分辨率           | 宽高均 >= 256                             | ERROR   |
| 最大分辨率           | 宽高均 <= 2048（防止 OOM）                 | WARNING |
| Caption 非空         | .txt 文件不能为空                          | ERROR   |
| 数据集大小           | 至少 10 张图片（< 20 提示警告）            | WARNING |
| 重复文件名           | 检测文件名冲突                             | ERROR   |

### 3.2 返回格式

```python
@dataclass
class ValidationResult:
    valid: bool
    total_images: int
    errors: List[ValidationError]  # 致命错误
    warnings: List[str]            # 非致命警告
    
@dataclass
class ValidationError:
    file_path: str
    error_type: str  # "missing_caption" | "invalid_image" | ...
    message: str
```

### 3.3 实现要点（参考 kohya）

- 使用 `Pillow` 打开图片，捕获 `UnidentifiedImageError`
- 检查 `image.size` 是否符合分辨率范围
- 使用 `Path.glob("*.txt")` 快速匹配 caption 文件
- 并行校验（`ThreadPoolExecutor`）加速大数据集

---

## 4. Bucket 管理

### 4.1 Bucketing 原理（参考 kohya BucketBatchSampler）

**目标**：将宽高比相近的图片分到同一个 bucket，batch 内图片可以用同一分辨率，避免过度 padding。

**步骤**：
1. 对每张图片计算宽高比 `aspect_ratio = width / height`
2. 根据允许的分辨率（如 512x512, 512x768, 768x512）生成 bucket 列表
3. 将图片分配到最接近的 bucket
4. 每个 epoch 从各 bucket 随机采样，确保 batch 内宽高一致

### 4.2 Bucket 定义

```python
@dataclass
class Bucket:
    resolution: Tuple[int, int]  # (width, height)
    images: List[str]            # 文件路径列表
    aspect_ratio: float
    
    def __len__(self):
        return len(self.images)
```

### 4.3 Bucket 生成策略（参考 kohya）

```python
def generate_buckets(
    base_resolution: int,  # 如 512
    min_bucket_reso: int = 256,
    max_bucket_reso: int = 1024,
    bucket_reso_step: int = 64,
    max_aspect_ratio: float = 2.0
) -> List[Bucket]:
    """
    生成所有允许的 bucket 分辨率
    
    Example for base_resolution=512:
    - (512, 512) → 1:1
    - (512, 768) → 2:3
    - (768, 512) → 3:2
    - (640, 640) → 1:1
    """
    buckets = []
    for w in range(min_bucket_reso, max_bucket_reso + 1, bucket_reso_step):
        for h in range(min_bucket_reso, max_bucket_reso + 1, bucket_reso_step):
            aspect = w / h
            if 1 / max_aspect_ratio <= aspect <= max_aspect_ratio:
                # 确保总像素数接近 base_resolution^2
                if abs(w * h - base_resolution ** 2) < base_resolution ** 2 * 0.2:
                    buckets.append(Bucket((w, h), [], aspect))
    return buckets
```

### 4.4 图片分配算法

```python
def assign_images_to_buckets(
    image_paths: List[str],
    buckets: List[Bucket]
) -> List[Bucket]:
    """将图片分配到最接近的 bucket"""
    for img_path in image_paths:
        img = Image.open(img_path)
        img_aspect = img.width / img.height
        
        # 找到最接近的 bucket
        closest_bucket = min(
            buckets,
            key=lambda b: abs(b.aspect_ratio - img_aspect)
        )
        closest_bucket.images.append(img_path)
    
    # 过滤空 bucket
    return [b for b in buckets if len(b) > 0]
```

---

## 5. Latent 缓存（cache_latents）

### 5.1 Cache 策略（参考 kohya LatentsCachingDataset）

**触发条件**（由 HyperparamPolicy 决定）：
- 数据集较小（< 500 张）
- 显存充足（> 10GB 或用户显式启用）
- 不使用数据增强（MVP 无 flip/crop）

**实现流程**：
1. 遍历所有图片，用 VAE encoder 计算 latent
2. 保存 latent 到内存或磁盘（.npz 格式）
3. 训练时直接加载 latent，跳过 VAE encode

### 5.2 Cache 存储格式

```python
# 内存 cache（小数据集）
latent_cache = {
    "image_001.png": torch.Tensor([...]),  # shape: (4, h/8, w/8)
    "image_002.png": torch.Tensor([...]),
}

# 磁盘 cache（大数据集）
# 保存为 dataset_path/latent_cache/image_001.npz
np.savez_compressed(
    cache_path,
    latent=latent.cpu().numpy(),
    original_size=(width, height)
)
```

### 5.3 Cache 实现要点

```python
def cache_latents(
    vae,
    image_paths: List[str],
    cache_dir: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    """预计算 latent"""
    cache = {}
    vae.eval()
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Caching latents"):
            img = Image.open(img_path).convert("RGB")
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(vae.device)
            
            # VAE encode
            latent = vae.encode(img_tensor).latent_dist.sample()
            latent = latent * 0.18215  # SD scaling factor
            
            if cache_dir:
                # 保存到磁盘
                cache_path = cache_dir / f"{Path(img_path).stem}.npz"
                np.savez_compressed(cache_path, latent=latent.cpu().numpy())
            else:
                # 保存到内存
                cache[img_path] = latent.cpu()
    
    return cache
```

---

## 6. Batch 构造（BucketBatchSampler）

### 6.1 Sampler 接口（参考 kohya）

```python
class BucketBatchSampler:
    """从 bucket 中采样 batch，确保 batch 内分辨率一致"""
    
    def __init__(
        self,
        buckets: List[Bucket],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True
    ):
        self.buckets = buckets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 预计算总 batch 数
        self.total_batches = sum(
            len(b) // batch_size for b in buckets
        )
    
    def __iter__(self):
        """每次返回一个 batch 的图片索引"""
        for bucket in self.buckets:
            indices = list(range(len(bucket.images)))
            if self.shuffle:
                random.shuffle(indices)
            
            # 分批次返回
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    yield [(bucket, idx) for idx in batch_indices]
    
    def __len__(self):
        return self.total_batches
```

### 6.2 Dataset 接口

```python
class LoRADataset(Dataset):
    """配合 BucketBatchSampler 使用的数据集"""
    
    def __init__(
        self,
        buckets: List[Bucket],
        latent_cache: Optional[Dict] = None
    ):
        self.buckets = buckets
        self.latent_cache = latent_cache
    
    def __getitem__(self, idx: Tuple[Bucket, int]):
        bucket, img_idx = idx
        img_path = bucket.images[img_idx]
        caption_path = img_path.replace(".png", ".txt").replace(".jpg", ".txt")
        
        # 加载 caption
        with open(caption_path, "r") as f:
            caption = f.read().strip()
        
        # 加载图片或 latent
        if self.latent_cache and img_path in self.latent_cache:
            latent = self.latent_cache[img_path]
            return {
                "latent": latent,
                "caption": caption,
                "resolution": bucket.resolution
            }
        else:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(bucket.resolution, Image.LANCZOS)
            img_tensor = transforms.ToTensor()(img)
            return {
                "image": img_tensor,
                "caption": caption,
                "resolution": bucket.resolution
            }
```

---

## 7. Prefetch 优化

### 7.1 异步加载（参考 PyTorch DataLoader）

```python
dataloader = DataLoader(
    dataset,
    batch_sampler=bucket_sampler,
    num_workers=4,           # 多进程加载
    pin_memory=True,         # 加速 GPU 传输
    persistent_workers=True  # 保持 worker 进程
)
```

### 7.2 Prefetch 策略

- **num_workers=4**：主流配置，适配 8GB+ VRAM
- **pin_memory=True**：CUDA 设备必开
- **persistent_workers=True**：避免每个 epoch 重启 worker

---

## 8. 参考实现映射

| 功能模块          | 参考仓库         | 文件路径                                      |
|------------------|-----------------|---------------------------------------------|
| 数据集校验        | flymyai-lora-trainer | `flymyai_lora_trainer/validate_dataset.py` |
| Bucket 生成      | sd-scripts      | `library/train_util.py` → `get_bucket_manager()` |
| Bucket 批采样器 | sd-scripts    | `library/train_util.py` → `BucketBatchSampler` |
| Latent 缓存       | sd-scripts      | `library/train_util.py` → `LatentsCachingDataset` |
| Dataset 基类     | sd-scripts      | `library/train_util.py` → `BaseDataset`    |

---

## 9. MVP 简化约束

| 特性              | MVP 状态       | 备注                                    |
|------------------|---------------|----------------------------------------|
| Bucketing        | ✅ 默认启用    | 必须实现，提升效率                        |
| Latent Cache     | ✅ 可选启用    | 由 HyperparamPolicy 自动决策              |
| 数据增强         | ❌ 不支持      | 无 flip/crop/color jitter                |
| 多分辨率训练     | ❌ 不支持      | 仅支持单分辨率 bucketing                  |
| Regularization图 | ❌ 不支持      | Phase 2 特性                             |
| Caption 预处理   | ❌ 手动提供    | 用户需自行生成 .txt 文件                   |

---

## 10. 错误处理

### 10.1 数据集校验失败

```python
if not validation_result.valid:
    print(f"❌ Dataset validation failed with {len(validation_result.errors)} errors:")
    for err in validation_result.errors[:5]:  # 只显示前 5 个
        print(f"  - {err.file_path}: {err.message}")
    raise ValueError("Dataset validation failed. Fix errors and retry.")
```

### 10.2 Latent Cache OOM

```python
try:
    latent_cache = cache_latents(vae, image_paths)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("⚠️ Not enough memory for latent cache. Falling back to on-the-fly encoding.")
        latent_cache = None
    else:
        raise
```

### 10.3 Bucket 为空

```python
buckets = [b for b in buckets if len(b) > 0]
if not buckets:
    raise ValueError("No valid buckets after assignment. Check dataset resolution.")
```

---

## 11. 测试要点

| 测试场景              | 输入                         | 期望输出                          |
|---------------------|-----------------------------|---------------------------------|
| 正常数据集            | 100 张图片 + 100 个 .txt      | 校验通过，生成 5-10 个 bucket      |
| 缺失 caption         | 50 张图片 + 30 个 .txt        | 校验失败，报告 20 个 missing_caption |
| 极端宽高比            | 3000x500 图片                | 分配到 2:1 bucket 或警告超出范围   |
| 小数据集 cache       | 20 张图片 + 10GB 显存         | 成功 cache，训练时直接加载 latent  |
| 大数据集 no cache    | 1000 张图片 + 8GB 显存        | 不 cache，on-the-fly encode      |
| Batch 一致性         | Batch size=4，同一 bucket    | 4 张图片分辨率完全一致             |
