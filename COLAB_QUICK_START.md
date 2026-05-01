# 🚀 Colab LoRA 训练 - 快速参考卡

## 📋 一句话总结
**完整的 Colab LoRA 自动化训练流程已准备好。只需在 Colab 中复制粘贴代码逐个运行。预计 35-50 分钟完成。**

---

## ⚡ 3 秒快速开始

1. **打开**: https://colab.research.google.com
2. **启用 GPU**: Runtime → Change runtime type → GPU
3. **执行**: 按顺序复制 [COLAB_WORKFLOW_GUIDE.md](./COLAB_WORKFLOW_GUIDE.md) 中的 7 个单元格

---

## 📁 文件导航

| 文件 | 作用 | 何时使用 |
|------|------|--------|
| **[COLAB_WORKFLOW_GUIDE.md](./COLAB_WORKFLOW_GUIDE.md)** | ⭐ 主要指南 | **立即开始** |
| [COLAB_FINAL_REPORT.md](./COLAB_FINAL_REPORT.md) | 详细总结报告 | 了解全貌 |
| [COLAB_DIAGNOSTIC_REPORT.md](./COLAB_DIAGNOSTIC_REPORT.md) | 故障排查 | 遇到问题时 |
| colab_complete_workflow.py | 工作流脚本 | (参考用) |
| COLAB_EXECUTION_GUIDE.py | 指南生成器 | (参考用) |

---

## 🎯 7 个主要步骤

| # | 步骤 | 时间 | 输出 |
|---|------|------|------|
| 1️⃣ | GPU 检查 | 5秒 | ✓ GPU: T4 |
| 2️⃣ | 克隆仓库 | 15秒 | ✓ 仓库已克隆 |
| 3️⃣ | 安装依赖 | 2分钟 | ✓ 依赖已安装 |
| 4️⃣ | **数据 + 训练** ⭐ | **15-25分钟** | **✓ lora_final.safetensors** |
| 5️⃣ | 可视化对比 | 5分钟 | ✓ Base vs LoRA 图像对比 |
| 6️⃣ | 日志分析 | 1分钟 | ✓ Loss 下降 %、有效性检查 |
| 7️⃣ | 下载结果 | 1分钟 | ✓ 压缩包已下载 |

**总计: 35-50 分钟**

---

## 🔑 关键数据

```
本机 IP:           172.30.189.137
HTTP 端口:         8765
ZIP 文件:          fern_new.zip (~150 MB)
数据集大小:        63 对图像+标题

Colab 工作目录:     /content/lora-trainer
输出目录:          /content/runs/test_fern
最终文件:          lora_final.safetensors (~200 MB)

触发令牌:          f3rn_char
示例提示词:        "f3rn_char, 1girl"
```

---

## ✅ 成功标志

- ✓ 步骤 4 显示: "✓ 训练完成"
- ✓ 日志显示: Loss 从 0.23 降到 0.08
- ✓ metadata.json 中: `effectiveness_passed: true`
- ✓ 步骤 5 生成: Base vs LoRA 对比图像清晰不同
- ✓ 步骤 7 下载: `run_*_artifacts.zip` (~200 MB)

---

## ❌ 常见问题速解

| 问题 | 症状 | 解决 |
|------|------|------|
| 网络错误 | "Cannot reach http://172.30.189.137:8765" | 检查本机服务器: `ps aux \| grep http.server` |
| GPU 缺失 | "No NVIDIA GPU found" | Runtime → Change runtime type → GPU |
| 内存溢出 | "Out of memory" | 减少 batch_size (改为 8 或 4) |
| Loss 不下降 | 最终 loss ≈ 初始 loss | 检查数据集完整性和标题文件 |

---

## 🎨 输出预览

### 训练完成后
```
✓ 训练工作流全部完成!
✓ 运行目录: run_20260427_HHMMSS_sd15_lora
📈 总步数: 100
📈 初始 loss: 0.2345
📈 最终 loss: 0.0812
✓ LoRA 文件: (185.2 MB)
✓ 有效性: 通过
```

### 日志分析后
```
📊 训练统计:
   总步数: 100
   初始 loss: 0.2345
   最终 loss: 0.0812
   Loss 下降: 65.4%
✓ Loss 在下降，训练有效
```

### 可视化对比
```
Base 模型            LoRA 模型
[标准女孩]    VS    [学到特征的女孩]
```

---

## 📝 执行清单

- [ ] 打开 Colab: https://colab.research.google.com
- [ ] 启用 GPU 运行时
- [ ] 复制 Cell 1 并运行 (GPU 检查)
- [ ] 复制 Cell 2 并运行 (克隆仓库)
- [ ] 复制 Cell 3 并运行 (安装依赖)
- [ ] 复制 Cell 4 并运行 (训练) ⭐ 关键步骤
- [ ] 复制 Cell 5 并运行 (可视化)
- [ ] 复制 Cell 6 并运行 (日志分析)
- [ ] 复制 Cell 7 并运行 (下载结果)
- [ ] 验证: lora_final.safetensors 已下载

---

## 🎯 我已为你准备了什么

### ✅ 自动化脚本
- Cell 4 自动处理所有数据工作：下载→解压→验证→令牌注入→训练→收集

### ✅ 可视化对比
- Cell 5 自动生成 Base vs LoRA 对比图像（6 组）

### ✅ 日志分析
- Cell 6 自动计算 loss 下降比例和有效性指标

### ✅ 完整文档
- 故障排查指南
- 详细工作流说明
- 执行总结报告

---

## 🚀 立即开始

**→ 打开 [COLAB_WORKFLOW_GUIDE.md](./COLAB_WORKFLOW_GUIDE.md) 并按步骤 1-7 执行**

---

## 💡 进阶选项

### 修改训练参数

编辑 `/content/lora-trainer/examples/config_fern_test.yaml`:
- `max_train_steps`: 增加到 200-500 以获得更好的结果
- `batch_size`: 减少到 8 或 4 以避免 OOM
- `lora_r`: 改变 LoRA 秩 (默认 8)

### 使用不同的配置

```
examples/config_basic.yaml      # 基础快速测试
examples/config_quality.yaml    # 高质量完整训练
examples/config_agent_1000.yaml # 1000 步完整训练
```

### 本地执行代替 Colab

```bash
python3 colab_complete_workflow.py
```

---

## 📞 遇到问题？

1. **查看**：[COLAB_DIAGNOSTIC_REPORT.md](./COLAB_DIAGNOSTIC_REPORT.md)
2. **搜索**：Ctrl+F 搜索关键字 (如 "network error", "out of memory")
3. **检查**：Colab 单元格中的错误信息
4. **查看日志**：`/content/runs/test_fern/run_*/logs/train.log`

---

## 🎉 预期成果

完成后你将获得：

✓ **lora_final.safetensors** - 可用的 LoRA 权重  
✓ **metadata.json** - 训练指标和有效性报告  
✓ **train.log** - 详细训练日志  
✓ **对比图像** - Base vs LoRA 视觉证明  
✓ **完整分析** - Loss 曲线、效果评估

---

**版本**: 1.0 | **状态**: ✅ 准备就绪 | **最后更新**: 2026-04-27
