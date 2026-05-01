#!/usr/bin/env python3
"""
Colab 上传辅助脚本
================

这个脚本帮助处理 Colab 笔记本中的文件上传问题。

使用方法：
---------
在你的 Colab 笔记本中（Cell 3.5），粘贴以下代码：

```python
# 运行此代码来上传文件
from google.colab import files
uploaded = files.upload()
for fname in uploaded.keys():
    print(f'上传完成: {fname}')
```

然后选择 `/Users/nev4rb14su/Downloads/fern_new.zip` 上传。
"""

def show_colab_upload_code():
    """打印可以在 Colab 中运行的上传代码"""
    code = '''# Cell: Upload Dataset (Run this if file not in /content)
from google.colab import files
from pathlib import Path

zip_path = Path('/content/fern_new.zip')
if not zip_path.exists():
    print('⏳ Uploading fern_new.zip to /content...')
    print('请在弹出的对话框中选择: /Users/nev4rb14su/Downloads/fern_new.zip')
    uploaded = files.upload()
    for fname, content in uploaded.items():
        print(f'✓ Uploaded: {fname} ({len(content)} bytes)')
else:
    print(f'✓ Found: {zip_path}')
'''
    return code


if __name__ == '__main__':
    print(__doc__)
    print("\n=" * 60)
    print("复制以下代码到 Colab 单元格中运行：")
    print("=" * 60)
    print(show_colab_upload_code())
    print("=" * 60)
    print("\n完成上传后，继续运行 Cell 4 及之后的单元格。")
