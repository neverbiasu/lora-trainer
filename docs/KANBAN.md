# 执行计划（MVP → Next）

## 0. 当前结论

**MVP 阶段：✅ 已完成（核心闭环）**

已满足顶层约束中的最小闭环：`validate → train → export`。

- `validate`：`DataValidator` + `create_data_loader` 可运行。
- `train`：`Trainer.start()/train()/end()` 已打通，LoRA 注入使用 `apply_to`。
- `export`：训练结束可导出 `lora_final.safetensors`。
- 单入口与单配置源：保持 CLI + YAML 路径。

> 说明：`resume` / `export-only` 仍为后续增强，不阻塞 MVP 闭环定义。

---

## 1. 里程碑状态

```
██████████████████████░░░░░░ MVP（M1-M4）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
M1 基础框架    ✅ 完成
M2 数据层      ✅ 完成
M3 训练层      ✅ 完成（核心路径）
M4 导出与工具  ✅ 完成（最小导出路径）
```

### M3 完成项（实际代码）

- `src/lora_trainer/model_adapter.py`
  - SD1.5 模型加载、checkpoint 转换、prompt/image encode/decode。
- `src/lora_trainer/lora.py`
  - `LoRAAdapter.apply_to(...)`、幂等注入、hook 管理、`remove_injection()`。
- `src/lora_trainer/trainer.py`
  - 完整训练生命周期、步数控制、checkpoint、最终导出。
- `src/lora_trainer/cli.py`
  - 主路径已接入 `Trainer`。

### M4（MVP 范围）完成项

- 训练结束导出：`run/export/lora_final.safetensors`。
- run 元数据与 config snapshot 持续记录。

---

## 2. 已知差距（不影响 MVP，但影响工程化）

### P1（建议优先）

1. LoRA 权重回载：`LoRAAdapter.load_weights(path)`。
2. 导出元信息增强：`rank/alpha/base_model/target_modules/format_version`。
3. CLI 能力补齐：`--resume`、`--export-only`。
4. 训练测试补齐：`tests/test_trainer.py`（最小 3~5 条）。

### P2（稳定性/可维护性）

1. 模块化 exporter（从 trainer 中拆出明确导出接口）。
2. metadata 中增加 reproducibility 字段（config_hash/code_version/base_model_hash）。
3. 文档与示例统一（命令、文件名、产物位置）。

### P3（性能与体验）

1. gradient checkpointing / xformers 开关实装。
2. 采样策略完善（固定 prompt + seed）。
3. 训练日志可视化（简版 CSV/TensorBoard 二选一）。

---

## 3. 下一阶段计划（V0.2）

## Sprint A（2~3 天）— 工程完整性

- [ ] A1: `LoRAAdapter.load_weights(path)` + 单测
- [ ] A2: `export_weights` metadata 增强
- [ ] A3: CLI `--export-only` 打通
- [ ] A4: `tests/test_trainer.py` + `tests/test_lora.py` 基础覆盖

**验收标准**

```bash
python -m pytest tests/test_trainer.py tests/test_data_loader.py -v
# 期望: 全通过

python -m src.lora_trainer.cli --config examples/config_fern_test.yaml --dry-run
# 期望: 解析正确
```

## Sprint B（2~3 天）— 运行恢复与稳定性

- [ ] B1: `--resume` 最小恢复（optimizer/scheduler/global_step）
- [ ] B2: run metadata 完整化（hash/version/seed）
- [ ] B3: checkpoint 命名与保留策略

**验收标准**

```bash
# 中断后恢复继续训练
python -m src.lora_trainer.cli --config ... --resume runs/.../checkpoints/step_xxxx.safetensors
# 期望: global_step 连续，loss 正常
```

## Sprint C（可选）— 性能优化

- [ ] C1: gradient checkpointing
- [ ] C2: xformers（有环境时）
- [ ] C3: 采样吞吐和显存指标记录

---

## 4. 下一阶段设计任务（docs/design）

1. `docs/design/trainer.md`
   - 增补 Resume 状态机（init/load/train/end）。
2. `docs/design/lora.md`
   - 增补 `load_weights` 与 metadata schema。
3. `docs/design/run_manager.md`
   - 定义 checkpoint manifest 与保留策略。
4. `docs/design/cli_design.md`
   - 补全 `export-only` / `resume` 参数流与错误码。

---

## 5. 快速命令（当前可用）

```bash
# 配置检查
python -m src.lora_trainer.cli --config examples/config_fern_test.yaml --dry-run

# 仅校验配置
python -m src.lora_trainer.cli --config examples/config_fern_test.yaml --validate-only

# 训练（会生成 run/checkpoints/export）
python -m src.lora_trainer.cli --config examples/config_fern_test.yaml
```

---

## 6. 最近关键提交

| Commit | 内容 |
|---|---|
| `afcf4a3` | 训练主流程接入 `LoRA apply_to`、CLI 连接 Trainer |
| `b1378a3` | 数据层（validator/dataloader）完成 |
| `9d03e8a` | CLI + Config + RunManager 基础框架 |
