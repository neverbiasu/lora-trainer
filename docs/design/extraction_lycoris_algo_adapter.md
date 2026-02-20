# LyCORIS LoRA 实现提取（mvp v0.1）

## 参考文件
- `LyCORIS/lycoris/modules/base.py` - 基类定义
- `LyCORIS/lycoris/modules/loha.py` - LoHA 实现
- `LyCORIS/lycoris/modules/lokr.py` - LoKr 实现
- `LyCORIS/lycoris/wrapper.py` - 模块包装器

## 1. 统一基类接口（base.py）

```python
# 参考：LyCORIS/lycoris/modules/base.py

class ModuleCustomSD(nn.Module):
    """所有 LoRA 变种的统一基类"""
    
    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,  # 原始模块（要注入的目标）
        multiplier: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.lora_name = lora_name
        self.org_module = org_module
        self.multiplier = multiplier
        
    @abstractmethod
    def make_weight(self, *args, **kwargs):
        """生成 ΔW（权重增量）"""
        pass
    
    def forward(self, x):
        """前向传播：原始权重 + LoRA 增量"""
        # 原始输出
        org_out = self.org_module(x)
        
        # LoRA 增量
        delta_weight = self.make_weight()
        lora_out = F.linear(x, delta_weight)
        
        return org_out + self.multiplier * lora_out
    
    def get_diff_weight(self):
        """获取可训练权重（用于保存/导出）"""
        return self.make_weight()
```

## 2. 标准 LoRA 实现（简化版）

```python
# 基于 LyCORIS 的 LoRA 核心逻辑

class LoRAModule(ModuleCustomSD):
    """标准 LoRA：ΔW = B @ A"""
    
    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        **kwargs
    ):
        super().__init__(lora_name, org_module, **kwargs)
        
        in_dim = org_module.in_features
        out_dim = org_module.out_features
        
        # LoRA 矩阵：低秩分解
        self.lora_down = nn.Linear(in_dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_dim, bias=False)
        
        # 缩放因子
        self.scale = alpha / rank
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
    
    def make_weight(self):
        """ΔW = scale * (B @ A)"""
        return self.scale * (self.lora_up.weight @ self.lora_down.weight)
    
    def forward(self, x):
        org_out = self.org_module(x)
        lora_out = self.lora_up(self.lora_down(x))
        return org_out + self.scale * lora_out
```

## 3. 权重注入策略（wrapper.py）

```python
# 参考：LyCORIS/lycoris/wrapper.py

def inject_lora_to_model(
    model: nn.Module,
    lora_config: dict,
    target_modules: List[str] = None
):
    """将 LoRA 模块注入到目标层"""
    
    # 默认目标：UNet 的所有 Linear/Conv2d
    if target_modules is None:
        target_modules = [
            "to_q", "to_k", "to_v", "to_out",  # Attention
            "ff.net.0", "ff.net.2"  # FeedForward
        ]
    
    replaced_modules = []
    
    for name, module in model.named_modules():
        # 匹配目标模块
        if any(target in name for target in target_modules):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 替换为 LoRA 模块
                lora_module = LoRAModule(
                    lora_name=name,
                    org_module=module,
                    rank=lora_config["rank"],
                    alpha=lora_config["alpha"]
                )
                
                # 替换父模块中的引用
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = get_module_by_name(model, parent_name)
                setattr(parent, child_name, lora_module)
                
                replaced_modules.append(name)
    
    return replaced_modules
```

## 4. 可训练参数管理

```python
def get_trainable_params(model):
    """只返回 LoRA 参数（冻结原始权重）"""
    trainable = []
    
    for name, param in model.named_parameters():
        # 只训练包含 'lora' 的参数
        if 'lora' in name.lower():
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False
    
    return trainable
```

## 5. 导出与合并

```python
def merge_lora_weights(model):
    """将 LoRA 权重合并到原始权重（导出时）"""
    
    for name, module in model.named_modules():
        if isinstance(module, LoRAModule):
            # 获取原始权重
            org_weight = module.org_module.weight.data
            
            # 获取 LoRA 增量
            delta_weight = module.get_diff_weight()
            
            # 合并
            merged_weight = org_weight + delta_weight
            
            # 替换回原始模块
            module.org_module.weight.data = merged_weight
```

## 6. mvp AlgoAdapter 设计映射

### 基类接口（algo_adapter_base.py）

```python
class AlgoAdapterBase(ABC):
    """算法适配器基类"""
    
    @abstractmethod
    def prepare(self, model, config):
        """注入训练模块"""
        pass
    
    @abstractmethod
    def get_trainable_params(self):
        """获取可训练参数"""
        pass
    
    @abstractmethod
    def forward(self, batch):
        """前向传播"""
        pass
    
    @abstractmethod
    def compute_loss(self, pred, target):
        """计算损失"""
        pass
    
    @abstractmethod
    def export_weights(self):
        """导出权重"""
        pass
```

### LoRA 实现（lora_adapter.py）

```python
class LoRAAdapter(AlgoAdapterBase):
    """标准 LoRA 适配器"""
    
    def __init__(self, rank=32, alpha=None):
        self.rank = rank
        self.alpha = alpha if alpha else rank
        self.injected_modules = []
    
    def prepare(self, model, config):
        # 参考 LyCORIS 的注入逻辑
        target_modules = config.get("target_modules", DEFAULT_TARGETS)
        self.injected_modules = inject_lora_to_model(
            model, 
            {"rank": self.rank, "alpha": self.alpha},
            target_modules
        )
    
    def get_trainable_params(self):
        # 只返回 lora 参数
        return [p for n, p in self.model.named_parameters() if 'lora' in n]
    
    # ... forward / compute_loss / export_weights
```

## 7. 关键设计要点

| LyCORIS 设计 | mvp 借鉴 | 说明 |
| --- | --- | --- |
| `ModuleCustomSD` 基类 | `AlgoAdapterBase` | 统一接口，支持多算法 |
| `make_weight()` | `get_diff_weight()` | 生成权重增量 ΔW |
| `inject_lora_to_model()` | `prepare()` | 注入 LoRA 到目标层 |
| `scale = alpha / rank` | 同 | alpha 缩放因子 |
| Kaiming 初始化 down / 零初始化 up | 同 | 标准初始化策略 |
| 目标模块白名单 | `target_modules` | to_q/to_k/to_v/to_out 等 |

## 8. mvp 实现优先级

### 必须实现
1. ✅ `AlgoAdapterBase` 基类（统一接口）
2. ✅ `LoRAAdapter` 标准实现（rank/alpha/inject/forward）
3. ✅ 权重注入逻辑（target_modules 匹配）
4. ✅ 可训练参数过滤（只训练 lora）
5. ✅ 导出与合并（merge_lora_weights）

### Phase_2 扩展
- LoKr（Kronecker 分解）
- LoHA（Hadamard 积）
- LoCon（卷积层 LoRA）
- 动态 rank（DyLoRA）
