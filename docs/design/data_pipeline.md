# 数据集与数据流设计（mvp v0.1）

## 1. 数据集约定

| 项 | 说明 |
| --- | --- |
| 配对 | `image.png` + `image.txt` 一一对应 |
| 兼容性 | 兼容 sd-scripts 目录规范 |
| caption | 使用 utf-8 文本 |

## 2. 数据加载流程

1. `validate_dataset`：检查配对、空 caption、非法路径、重复基名。
2. 分词编码（tokenization）：caption → token_ids。
3. 图像处理：按 bucket 或统一分辨率。
4. 可选 cache_latents：预先 vae encode。

## 3. mvp 策略

1. 默认开启 bucketing（避免强裁为 1:1）。
2. cache_latents 作为自动策略（低显存优先启用）。

## 4. 失败处理

- validate 失败直接阻断训练。
- caption 为空或损坏：记录并报错。
