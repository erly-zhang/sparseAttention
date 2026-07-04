# attention_map

在 **RULER** 数据上运行 **Qwen2.5-3B**，按 **(layer_i, head_j)** 记录并保存 attention map。

## 目录结构

```
attention_map/
├── attention_map/          # Python 包
│   ├── data.py             # 读取 ruler_data jsonl
│   ├── model.py            # 加载模型、prefill
│   ├── recorder.py         # 按 (i,j) 保存 attention
│   └── run.py              # CLI 入口
├── outputs/                # 运行后生成（默认）
├── requirements.txt
├── run.sh                  #  smoke test 示例
└── README.md
```

## 输出格式

每个样本目录：

```
outputs/4k/niah_single_1/sample_001179_line0000/
├── manifest.json           # 索引说明 + 各层元数据
├── sample_meta.json        # 样本来源、seq_len 等
├── layer_00/
│   ├── head_00.npy         # (layer=0, head=0)
│   ├── head_01.npy
│   └── ...
├── layer_01/
│   └── ...
└── layer_35/
```

**索引约定**：`(i, j)` → `layer_{i:02d}/head_{j:02d}.npy`

- `query_slice=last`（默认）：每个 head 的 shape 为 `[1, seq_len]`，即**最后一个 query token** 对所有 key 的注意力。
- `query_slice=all`：shape 为 `[seq_len, seq_len]`，完整矩阵（4k 约 18GB/样本，慎用）。
- `query_slice=index`：指定 query 行（可用 `--query_index` 或数据里的 `token_position_answer`）。

## 安装

```bash
cd /home/ubuntu/work/attention_map
pip install -r requirements.txt
```

## 运行

### Smoke test（截断 512 token，1 条样本）

```bash
bash run.sh
```

### 4k 全量输入、所有样本（last-query attention）

```bash
export PYTHONPATH=/home/ubuntu/work/attention_map:$PYTHONPATH

python -m attention_map.run \
  --model_path /home/ubuntu/work/model/Qwen2.5-3B \
  --data_root /home/ubuntu/work/ruler_data \
  --output_dir /home/ubuntu/work/attention_map/outputs \
  --split 4k \
  --task niah_single_1 \
  --query_slice last
```

### 8k 数据

```bash
python -m attention_map.run \
  --split 8k \
  --task niah_single_1 \
  --query_slice last
```

### 限制样本数

```bash
python -m attention_map.run --split 4k --max_samples 2 --query_slice last
```

## 读取示例

```python
import json
import numpy as np
from pathlib import Path

root = Path("outputs/4k/niah_single_1/sample_001179_line0000")
manifest = json.loads((root / "manifest.json").read_text())

# (layer_i=3, head_j=5)
attn = np.load(root / "layer_03" / "head_05.npy")
print(attn.shape)  # [1, seq_len] when query_slice=last
```

## 说明

- 必须使用 `attn_implementation=eager` 才能拿到 attention 权重（已在代码中设置）。
- Qwen2.5-3B：36 层 × 16 heads（见 `config.json`）。
- 长序列建议 `--query_slice last`；需要完整矩阵时再加 `--query_slice all` 并确保显存/磁盘充足。
