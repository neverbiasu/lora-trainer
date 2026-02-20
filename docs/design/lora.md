# LoRA 注入设计（现状对齐 + 缺口清单）

## 1. 目的

这份文档只做三件事：

- 说明 `src/lora_trainer/lora.py` 现在已经实现了什么。
- 明确 LoRA 注入还缺哪些关键能力。
- 给每个待实现点标注可直接参考的来源文件。

## 1.1 本周 V0.2 设计目标

围绕“可恢复、可复用”补齐两项：

1. `load_weights(path)`：支持 `.safetensors/.pt` 回载 LoRA 权重。
2. `export_weights(..., metadata=...)`：输出标准化 metadata（最小 schema）。

本周不引入新算法（如 LoHa/OFT），不扩展新入口。

## 2. 当前代码现状（基于 `src/lora_trainer/lora.py`）

已实现：

1. `LoRAModule`：`down/up` 低秩分支 + `scale=alpha/rank` + device/dtype 对齐。
2. `LoRAAdapter.apply_to(text_encoder, unet, ...)`：参考 sd-scripts 风格，支持选择注入对象。
3. `LoRAAdapter._inject_into_model(model, ...)`：核心注入逻辑，幂等、记录 hook、返回报告。
4. `LoRAAdapter.remove_injection()`：卸载所有 hook。
5. `get_trainable_params()`：返回当前适配器参数。
6. `export_weights()`：支持 `.safetensors` / `.pt` 保存。

现状问题（P0 已解决，剩余缺口）：

- ✅ P0 已完成：精确目标匹配 + device/dtype 对齐 + 幂等注入 + hook 管理 + 注入报告。
- ⏳ P1 待实现：权重加载接口、导出元数据增强。
- ⏳ P2 可选：merge/unmerge 能力。

## 3. M3 必须补齐的实现项（按优先级）

### P0：注入正确性

1. **精确目标匹配**（必须）
   - 从 `ModelAdapter.get_target_modules()` 接收目标模式，不再用模糊匹配。
   - 默认 SD1.5：`to_q/to_k/to_v/to_out.0`。
2. **设备与类型对齐**（必须）
   - 创建 LoRA 模块后立即迁移到 `org_module.weight` 的 device/dtype。
3. **幂等注入**（必须）
   - 避免同一层重复注入；重复 `prepare` 时先检查已注入集合。

### P1：工程可维护性

4. **保存 hook handle 并支持卸载**
   - `self.hook_handles[name] = handle`。
   - 新增 `remove_injection()`，遍历 `handle.remove()`。
5. **注入报告**
   - 返回/记录：`injected_count`、`skipped_count`、`skipped_reasons`。
6. **权重加载接口**
   - 新增 `load_weights(path)` 支持 safetensors/pt。

### P1.5：导出 schema（本周）

7. **导出元信息最小 schema**
   - `format_version`
   - `base_model`
   - `network_module`（固定 `lora_trainer.lora`）
   - `rank`、`alpha`
   - `target_modules`（字符串化列表）

### P2：训练与导出稳定性

7. **导出元数据增强**
   - `base_model`, `rank`, `alpha`, `target_modules`, `format_version`。
8. **可选 merge/unmerge**
   - 提供 `merge_into_base()`（no-grad 下把增量并入原权重）。

## 4. 函数级设计（对应当前文件）

| 函数 | 当前状态 | 需要补充 | 参考来源 |
| --- | --- | --- | --- |
| `LoRAModule.__init__` | 已有线性版 down/up 初始化 | 增加 device/dtype 对齐（从 org module 继承） | `docs/references/sd-scripts/networks/lora.py` 的 `LoRAModule`（[lora.py](../references/sd-scripts/networks/lora.py#L25)） |
| `LoRAModule.forward` | 已返回 `scale * up(down(x))` | 可选 dropout/rank_dropout（非 P0） | `sd-scripts` `LoRAModule.forward`（[lora.py](../references/sd-scripts/networks/lora.py#L89)） |
| `LoRAAdapter._is_attention_module` | 仅模糊匹配 `attn` | 改为“目标名白名单/模式匹配” | `LoRANetwork.apply_to` 的目标控制思路（[lora.py](../references/sd-scripts/networks/lora.py#L1082)） |
| `LoRAAdapter.apply_to` | ✅ 已实现：幂等、handle 记录、报告 | 已完成 P0 需求 | `LoRAInfModule.apply_to`（[lora.py](../references/sd-scripts/networks/lora.py#L85)）、LyCORIS apply_to（[kohya.py](../references/LyCORIS/lycoris/kohya.py#L642)） |
| `LoRAAdapter.get_trainable_params` | 可用 | 保持 `self.parameters()` 即可 | `sd-scripts` `get_trainable_params`（[lora.py](../references/sd-scripts/networks/lora.py#L1252)） |
| `LoRAAdapter.export_weights` | 支持两种格式 | 增加 dtype 归一化和 metadata/hash | `sd-scripts` `save_weights`（[lora.py](../references/sd-scripts/networks/lora.py#L1255)） |
| `LoRAAdapter.load_weights` | 未实现 | 新增，从文件恢复 adapter 参数 | LyCORIS save/load 组织方式（[wrapper.py](../references/LyCORIS/lycoris/wrapper.py#L649)） |
| `LoRAAdapter.remove_injection` | ✅ 已实现 | 完成 | PyTorch hook handle 使用方式 |
| `LoRAAdapter.merge_into_base` | 未实现 | 新增 merge（可选 P2） | `LoRAInfModule.merge_to`（[lora.py](../references/sd-scripts/networks/lora.py#L156)） |

## 4.1 `load_weights(path)` 接口草案

```python
def load_weights(self, path: str, strict: bool = True) -> dict[str, Any]:
   """Load LoRA adapter weights from .safetensors or .pt and return load report."""
```

返回结构建议：

- `loaded_keys_count`
- `missing_keys`
- `unexpected_keys`
- `path`
- `strict`

行为约束：

- 若 suffix 为 `.safetensors`，使用 `safetensors.torch.load_file`。
- 若 suffix 为 `.pt`，使用 `torch.load(map_location="cpu")`。
- 默认 `strict=True`，用于训练续跑；`strict=False` 用于兼容加载。

## 5. 推荐实现顺序（最短闭环）

✅ **P0 已完成**：
1. `apply_to` 完整实现（目标匹配 + 幂等 + hook 管理 + device/dtype 对齐 + 报告）。
2. `remove_injection()` 已提供。

⏳ **P1 接下来实现**：
3. `load_weights(path)`：从文件恢复权重。
4. `export_weights()` 元数据增强：保存 `rank/alpha/base_model/target_modules/format_version`。

⏳ **P2 可选**：
5. `merge_into_base()`：权重合并。

按这个顺序能最快打通：`注入 -> 训练 -> 导出 -> 重载`。

## 6. 与 Trainer / ModelAdapter 的对接约束

1. Trainer 不直接写死目标模块，统一调用 `model_adapter.get_target_modules()`。
2. Trainer 只消费 `lora_adapter.get_trainable_params()` 产物。
3. 训练结束导出统一走 `lora_adapter.export_weights()`。

## 7. 完成判定（Definition of Done）

✅ **P0 已达成**（apply_to 已完整实现）：

1. 同一模型重复调用 `apply_to` 不会重复注入。✅
2. 注入后可输出明确报告（注入/跳过明细）。✅
3. 训练参数只包含 LoRA 参数。✅
4. Hook 可安全移除（`remove_injection()`）。✅

⏳ **P1 待完成**：

5. 导出权重可在新进程加载并继续训练或推理（需 `load_weights()`）。
