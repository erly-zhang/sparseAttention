"""Create an English offline report while retaining original tokens and data."""
from pathlib import Path
import os
import re
import shutil

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'figure5_html'
OUT=HERE/'figure5_html_en'

def copy_asset(src,dst):
    if Path(src).parent.name.startswith('ranks_'):
        os.link(src,dst)
        return dst
    return shutil.copy2(src,dst)

HTML={
 'lang="zh-CN"':'lang="en"',
 'Reasoning 的 J-lens 读出':'J-lens readouts during reasoning',
 '仿 Figure 5 的交互式 slice viewer：查看位置与层，固定 token，追踪它在整个词表中的排名。':'A Figure 5–style interactive slice viewer: explore positions and layers, pin tokens, and track their full-vocabulary ranks.',
 '3 个配对样本':'3 paired samples',
 '保护 prompt + recent64':'Protected prompt + recent64',
 '压缩间隔 64 步':'Compaction every 64 steps',
 '离线可用':'Works offline',
 '选择实验与位置':'Select experiment and position',
 '样本 <select':'Sample <select',
 '读出路径 <select':'Method <select',
 '完整 KV':'Full KV', '压缩 KV':'Compact KV',
 '位置 <select':'Position <select', '层 <select':'Layer <select',
 '前一个捕获位置':'Previous captured position', '后一个捕获位置':'Next captured position',
 '仅 reasoning':'Reasoning only',
 '正在读取本地数据…':'Loading local data…',
 '原始 prompt':'Original prompt',
 '当前捕获位置的上下文 · 所有路径使用同一 Dense 续写':'Context at the captured position · All methods replay the same Dense continuation',
 '固定 token':'Pinned tokens',
 ' 点击读出词可固定 / 取消；最多同时显示 4 个。':' Click a readout token to pin or unpin it; display up to 4 at once.',
 '搜索已导出的 token':'Search exported tokens',
 '如 probably、sum、数字':'e.g. probably, sum, numbers',
 '搜索结果':'Search results',
 '>固定</button>':'>Pin</button>', '>清空</button>':'>Clear</button>',
 '叠加 Dense 排名':'Overlay Dense ranks',
 '位置 × 层：最高排名的可读 token':'Position × layer: top wordlike token',
 '每列是一个真实捕获点；水平滚动查看更多位置。点击单元格选择位置与层。':'Each column is a captured checkpoint. Scroll horizontally for more positions; click a cell to select its position and layer.',
 '与 Dense 的第一读出不同':'Top token differs from Dense',
 '位置与层的第一读出网格':'Top-token grid by position and layer',
 '浅灰：尚未进入对应稀疏阶段':'Light gray: before sparsity begins',
 '橙色描边：当前选择':'Orange outline: current selection',
 '块编号：0–35':'Block indices: 0–35',
 'Pinned tokens 的全词表排名':'Full-vocabulary rank of a pinned token',
 '颜色使用 log 排名；越深表示排名越靠前。并列显示精确排名区间。':'Color uses log rank; darker means a higher rank. Ties retain exact rank intervals.',
 '热图 token':'Heatmap token',
 '词表末尾':'Vocab end',
 '固定token的全词表排名热图':'Pinned-token full-vocabulary rank heatmap',
 '固定位置 → 跨层读出':'Fixed position → layer slice',
 'Pinned tokens 的跨层排名':'Pinned-token ranks across layers',
 '虚线 = Dense':'Dashed = Dense',
 '固定层 → 跨位置读出':'Fixed layer → position slice',
 'Pinned tokens 的跨位置排名':'Pinned-token ranks across positions',
 '横轴 = 真实 decode step':'x-axis = actual decode step',
 '折线只连接已测量的点，空档内没有额外观测。Dashed = Dense；阴影 = 当前路径稀疏开始。':'Lines connect measured checkpoints only; gaps contain no observations. Dashed = Dense; shading marks the selected method’s sparse phase.',
 '当前单元格 · Top 8':'Selected cell · Top 8',
 '显示含字母或数字的 token；排名及概率仍在完整词表上计算。':'Display tokens containing letters or numbers; ranks and probabilities use the full vocabulary.',
 '如何读图 · 方法、边界与来源':'Reading guide · Methods, limitations, and sources',
 '读出公式：':'Readout formula: ',
 '。第 35 个块使用 J = I，对应模型下一 token 分布。未预设 Qwen 的 workspace 层区间。':'. Block 35 uses J = I and gives the model’s next-token distribution. No workspace layer band is assumed for Qwen.',
 '此处使用已完成的 3 个 HMMT 配对样本，每条 Dense 续写最多 4096 tokens。所有 sparse 路径均重放相同 Dense 前缀，比较的是相同位置的读出，不能从本图推断 sparse 自主生成的准确率。prompt 在 prefill 时使用 dense attention。':'This viewer uses three completed, paired HMMT samples, each with up to 4,096 Dense continuation tokens. All sparse methods replay the same Dense prefixes, so comparisons use matched positions. These readouts do not measure sparse free-generation accuracy. Prompt prefill uses dense attention.',
 'Full KV：每步从完整历史选择位置，不永久删除 KV。Compact KV：每 64 步在当时仍存活的 KV 中选择并永久删除其余位置；两次压缩之间长度可以超过 K。两者均使用总预算 K=2048、保护所有 prompt 和 recent64；W1 用当前 query 的代表头分数，Random 使用随机选择。':'Full KV: select positions from the entire history at every step, without permanently deleting KV. Compact KV: select from the surviving cache every 64 steps and permanently delete other positions; cache length may exceed K between compactions. Both use a total budget K=2048 and protect all prompt tokens plus recent64. W1 scores with representative heads at the current query; Random selects randomly.',
 '该图保留每 128 步及稀疏边界附近的实际捕获点，列间隔并不相等；未捕获 token 没有格子。曲线横轴使用真实 step，热图每列是一个捕获点。step 0 表示 prefill 最后一个 prompt token。绝对位置从 0 开始，abs = prompt_tokens + step − 1。':'Actual checkpoints were captured every 128 steps and near sparsity boundaries. Column intervals are unequal; uncaptured tokens have no cells. Trajectory x-axes use actual steps, while each heatmap column represents one checkpoint. Step 0 is the last prompt token at prefill. Absolute positions are zero-based: abs = prompt_tokens + step − 1.',
 'Pinned tokens 的排名在完整词表中独立计算：':'Pinned-token ranks are computed independently over the full vocabulary: ',
 ']，r<sub>max</sub>':'], r<sub>max</sub>',
 '。有限精度相等时显示 [r<sub>min</sub>,r<sub>max</sub>]；曲线和颜色使用几何中点，不是任意的并列排序。可固定的集合覆盖本样本所有路径、位置、层所显示的 Top 8 token。':'. Finite-precision ties are shown as [r<sub>min</sub>, r<sub>max</sub>]. Curves and colors use the geometric midpoint of that interval. Every Top 8 token displayed across this sample’s methods, positions, and layers can be pinned.',
 '公共固定 lens 来源于短 WikiText，与长 HMMT 存在分布差异。此前 J-space 重构探针没有超过随机词典负对照，因此本图展示的是 J-lens 读出差异，并不能证明 workspace 容量损失或推理能力的因果机制。Dense 两种 kernel 控制在这些 K2048 捕获点的 residual 完全一致。':'The public fixed lens was estimated on short WikiText sequences, which differ from long HMMT reasoning. The earlier J-space reconstruction probe did not outperform its random-dictionary negative control. This viewer describes J-lens readout differences; it does not establish workspace capacity loss or a causal mechanism for reasoning performance. The two Dense kernel controls have bitwise-identical residuals at these K2048 checkpoints.',
 '布局参考 ':'Layout inspired by ',
 'Transformer Circuits 2026 workspace 论文 Figure 5':'Figure 5 of the Transformer Circuits 2026 workspace paper',
 '；数据来自本地 Qwen3-4B HMMT pilot。离线查看不需要打开外部链接。显示筛选采用 jacobian-lens 的 meaningful-token 定义（Apache-2.0，Anthropic PBC）。':'. Data come from the local Qwen3-4B HMMT pilot. External links are optional for offline viewing. Display filtering uses the meaningful-token definition from jacobian-lens (Apache-2.0, Anthropic PBC). ',
 '查看数据来源与校验 manifest':'View provenance and validation manifest',
 '</a>。':'</a>.',
 '完整 Dense 续写（所有路径共同前缀）':'Full Dense continuation (shared prefix for all methods)',
 '本地离线包':'Local offline package',
}

JS={
 '完整 KV':'Full KV','压缩 KV':'Compact KV',
 '本地文件加载失败：${src}。请完整解压文件夹后打开 index.html。':'Failed to load local file: ${src}. Extract the complete folder, then open index.html.',
 '此浏览器不支持本地解压，请使用较新的 Chrome、Edge 或 Firefox。':'Local decompression is unsupported. Use a recent Chrome, Edge, or Firefox browser.',
 '排名数据损坏':'Corrupt rank data','排名边界异常':'Invalid rank bounds','排名数据长度异常':'Invalid rank data length',
 '点击固定 / 取消':'Click to pin / unpin',
 '点击右侧任意读出 token，即可查看它的排名轨迹。':'Click any readout token on the right to track its rank.',
 '请固定一个 token':'Pin a token', '无匹配 token':'No matching token',
 '固定一个 token 后显示排名热图':'Pin a token to display its rank heatmap',
 '块 ${l}':'Block ${l}',
 ' · 最终模型输出':' · Final model output',
 ' 个捕获位置':' captured positions',
 ' · 块 <b>':' · Block <b>',
 '与 Dense 的 Recall@25':'Recall@25 vs Dense',
 '全词表排名':'Full-vocabulary rank','概率':'Probability','<th>排名</th>':'<th>Rank</th>',
 "byLayer?'跨层':'跨位置'":"byLayer?'Across-layer ':'Across-position '",
 '}token排名':'}token ranks',
 '词表末尾':'Vocab',
 '固定 token 后显示轨迹':'Pin tokens to display trajectories',
 'prefill / 非 reasoning':'prefill / non-reasoning',
 'Dense 全历史':'Dense full history',
 '已进入稀疏阶段':'Sparse phase active',
 '尚未进入对应稀疏阶段':'Before sparse phase',
 '样本 ${d.idx}':'Sample ${d.idx}',
 ' 个捕获点':' checkpoints','36 个块':'36 blocks',
 '可追踪 ${d.tracked.length.toLocaleString()} 个 token':'${d.tracked.length.toLocaleString()} trackable tokens',
 '正在准备排名…':'Preparing ranks…','排名加载完成':'Ranks ready',
 '正在加载本地样本…':'Loading local sample…',
 ' · 模型输出':' · Model output',
 '已固定 4 个 token；请先取消其中一个。':'Four tokens are pinned. Unpin one before adding another.',
}

def translated(path,replacements):
    s=path.read_text()
    for old,new in replacements.items():
        if old not in s:
            raise ValueError(f'Missing translation in {path.name}: {old}')
        s=s.replace(old,new)
    if re.search('[\u3400-\u9fff]',s):
        raise ValueError(f'Untranslated report text in {path.name}')
    path.write_text(s)

def main():
    if not OUT.exists():
        shutil.copytree(SOURCE,OUT,copy_function=copy_asset)
    if 'lang="en"' not in (OUT/'index.html').read_text():
        translated(OUT/'index.html',HTML)
        translated(OUT/'viewer.js',JS)
    (OUT/'README.txt').write_text('''J-lens Dense / Sparse: a Figure 5–style interactive viewer
2026-10-07

OPENING THE REPORT
1. Extract the entire ZIP, retaining the directory structure.
2. Open index.html in a recent Chrome, Edge, or Firefox browser.
3. No network, Python, model, or local server is needed. Keep all accompanying files.
   Rank files load locally when a token is first pinned.

INTERACTION
- Switch between 3 HMMT samples and 5 methods: Dense, full-KV Random/W1, compact-KV Random/W1.
- Click a grid cell to select a captured position and layer, or use the controls above.
- Each grid column is a measured checkpoint, labeled with decode step and absolute position.
- The right panels show a layer slice, a position slice, and Top 8 readouts at the selected cell.
- Click a readout token to pin/unpin it. Up to 4 tokens can be shown simultaneously.
- Search exported tokens by text or token ID; select the pinned token used in the heatmap.
- Toggle the Dense rank overlay to compare the same token between methods.
- Disable "Reasoning only" to include prefill and captured non-reasoning checkpoints.
- preview.png is a static preview. Open index.html for the interactive report.

DATA AND ALGORITHMS
Model: Qwen3-4B. Dataset: HMMT. Total selection budget K=2048.
All prompt tokens and recent64 are protected.
Full KV selects every step from the entire history without permanently deleting KV.
Compact KV selects from surviving history every 64 steps and deletes unselected KV.
W1 uses representative-head scores at the current query. All sparse methods replay the same Dense prefixes.
There are 3 samples and 158 checkpoints (including non-reasoning), with all 36 blocks per checkpoint.
Checkpoints are spaced mainly every 128 steps, with additional captures around sparsity boundaries.
They are sampled checkpoints, rather than a complete token-by-token activation record.
Each Dense continuation is capped at 4,096 tokens. This is a diagnostic pilot, not a new accuracy benchmark.

Readout: softmax(W_U RMSNorm(J_l h_l)); the final block uses J=I.
Display filtering retains letter/number-containing tokens in the Top 8.
Probabilities and pinned-token ranks still use the full vocabulary.
Every displayed token can be pinned and tracked across all checkpoints/layers/methods in its sample.
Token IDs retain their identities, including different IDs with the same decoded text.
Rank ties are exact intervals: [1+count(z_v>z_t), count(z_v>=z_t)].
Heatmaps and curves use the geometric midpoint of the interval on a logarithmic rank scale.
Lines connect measured points only; gaps have no observations.

INTERPRETATION LIMITS
The layout is inspired by Figure 5, but the data are the local Qwen3-4B HMMT pilot.
The public lens was estimated on short WikiText sequences; transfer to long HMMT reasoning is limited.
The reconstruction probe did not beat its random-dictionary negative control.
This viewer does not establish workspace capacity loss or a causal mechanism for reasoning performance.
The two Dense kernel controls are bitwise identical at the captured K2048 checkpoints.
Sparse free-generation text and formal sparse accuracy are not measured here.
Original model readout tokens and prompt/continuation text are preserved, including multilingual tokens.

PROVENANCE AND VALIDATION
manifest.json: sample/input identities, lens provenance, configuration, data SHA256, and export validation.
browser_validation.json: offline browser interaction and rank-alignment checks.
data_validation.json: independent decoding of all ranks and checks of all 227,520 displayed entries.
SHA256SUMS.txt: checksums for package files.
provenance/: original pilot validation and interpretation records.
source/: copies of export, packing, localization, and validation code.
Reproduction also requires the original project, model, lens, and captured activations.
In the original project, export within a Slurm GPU allocation:
  python export_figure5.py
  python pack_figure5.py
Then create the English version:
  python localize_figure5.py
Browser verification requires Playwright:
  python check_figure5_browser_en.py

REFERENCE AND LICENSE
Figure 5: https://transformer-circuits.pub/2026/workspace/index.html
The meaningful-token filter is from jacobian-lens, Copyright 2026 Anthropic PBC, Apache-2.0.
See LICENSE.jacobian-lens.txt for the complete license.
''')
    checker=(HERE/'check_figure5_browser.py').read_text().replace("HERE/'figure5_html'","HERE/'figure5_html_en'").replace('排名加载完成','Ranks ready').replace('模型输出','model output').replace("HERE/'figure5_mobile_check.png'","HERE/'figure5_mobile_check_en.png'")
    (HERE/'check_figure5_browser_en.py').write_text(checker)
    print(f'English report: {OUT}')

if __name__=='__main__':main()
