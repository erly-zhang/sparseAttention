import json
from pathlib import Path

from validate_results import validate

HERE=Path(__file__).resolve().parent
verified=validate()
source = HERE/'../../experiments/flexprefill_head_usage_retry01_20261002/outputs/independent-audit.json'
data = json.loads(source.read_text())
assert data['status'] == 'PASS' and data['samples'] == 200
labels=[r'Dense',r'Calibrated: Token+Vertical',r'Calibrated: Token+Vertical+Slash',
        r'Random: Token+Vertical',r'Random: Token+Vertical+Slash',
        r'FlexPrefill',r'MInference']
maxima=[max(float(row['scores'][column]) for row in verified['scores']) for column in range(4)]
score_rows=[]
for label,row in zip(labels,verified['scores']):
    values=[r'\textbf{'+v+'}' if float(v)==maxima[i] else v for i,v in enumerate(row['scores'])]
    score_rows.append(label+' & '+' & '.join(values)+r' \\')
layer_rows = []
for layer in range(32):
    s = data['all']['layers'][str(layer)]
    layer_rows.append('{0} & {1:.3f} & {2:.3f} & {3}--{4} \\\\'.format(
        layer, s['vertical_slash_mean'], s['query_aware_mean'], s['query_aware_min'], s['query_aware_max']))
task_rows = []
for task, label in [('kv_retrieval','KV Retrieval'), ('longbook_choice_eng','En.MC'),
                    ('longbook_qa_eng','En.QA'), ('code_debug','Code.Debug')]:
    s = data['by_task'][task]['layers']['0']
    task_rows.append('{0} & {1:.2f} & {2:.2f} & {3}--{4} \\\\'.format(
        label, s['vertical_slash_mean'], s['query_aware_mean'], s['query_aware_min'], s['query_aware_max']))
tex = r'''% Compile with XeLaTeX. Latin text and numerals use Times New Roman.
% Data: flexprefill_head_usage_retry01_20261002/outputs/independent-audit.json
\documentclass[11pt,a4paper]{article}
\usepackage[margin=19mm]{geometry}
\usepackage{fontspec}
\usepackage{booktabs,tabularx,array}
\usepackage[unicode,hidelinks]{hyperref}
\IfFileExists{fonts/TimesNewRoman-Regular.ttf}{
  \setmainfont[Path=fonts/,BoldFont=TimesNewRoman-Bold.ttf,
    ItalicFont=TimesNewRoman-Italic.ttf,BoldItalicFont=TimesNewRoman-BoldItalic.ttf]
    {TimesNewRoman-Regular.ttf}
}{\setmainfont{Times New Roman}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.12}
\hypersetup{pdftitle={Sparse Prefill Results and FlexPrefill Head Usage},pdfauthor={}}
\begin{document}
\begin{center}
  {\Large\bfseries Sparse Prefill Results}
\end{center}

\textbf{Model and samples:} Llama-3.1-8B-Instruct; 32 layers, 32 query heads per layer.\par
KV Retrieval, En.MC, En.QA, and Code.Debug: 50 samples per task, 200 samples in total.

\vspace{5pt}
{\large\bfseries 1. Task Scores}

Mean official task score ×100; the highest scores in each column are bold.
All methods are applied to all 32 layers.

\begingroup
\setlength{\tabcolsep}{5pt}
\begin{tabularx}{\textwidth}{@{}Xrrrr@{}}
\toprule
\textbf{Method} & \textbf{KV Retrieval} & \textbf{En.MC} & \textbf{En.QA} & \textbf{Code.Debug} \\
\midrule
@@SCORE_ROWS@@
\bottomrule
\end{tabularx}
\endgroup

\vspace{3pt}
{\large\bfseries 2. Mask Configuration}

\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{38mm}>{\raggedright\arraybackslash}X@{}}
\toprule
\textbf{Item} & \textbf{Current implementation} \\
\midrule
Custom mask granularity & Individual key tokens; no expansion to full blocks. \\
Base Token mask & TopK=8192; each tile of 128 queries shares one set of key tokens. \\
Vertical & Adds individual key-token columns. \\
Slash & For each query, adds keys at selected offsets \textit{d=i−j}, i.e., \textit{j=i−d}. \\
Sharing within a layer & One representative head per layer constructs a mask shared by all 32 query heads in that layer. \\
Attention computation & Each head uses its own Q/K/V and normalizes independently within the shared mask. \\
FlexPrefill baseline & 128×128 block granularity. \\
\bottomrule
\end{tabularx}

\newpage
\begin{center}
{\Large\bfseries FlexPrefill Algorithm Usage}
\end{center}

\textbf{Parameters:} γ=0.95, τ=0.1, min\_budget=1024, block\_size=128.

\textbf{Source:} Prefill replay of the original 200 inputs with the same configuration;
two observations per input. Counts below refer to one pass.
\par Agreement between the two passes: 100\% across 204,800 head decisions per pass.

\begin{tabularx}{\textwidth}{@{}lXrr@{}}
\toprule
\textbf{Algorithm branch} & \textbf{Selection rule} & \textbf{Decisions} & \textbf{Share} \\
\midrule
Vertical-Slash & √JS ≥ 0.1 & 203750 & 99.49\% \\
Query-Aware Block-Sparse & √JS < 0.1 & 1050 & 0.51\% \\
\bottomrule
\end{tabularx}

Head decisions = samples × layers × heads per layer = 200×32×32 = 204,800.
Classification follows the actual algorithm branch. The Query-Aware branch retains
Vertical-Slash blocks at the minimum budget.

\vspace{4pt}
\textbf{Mean Head Counts by Task at Layer 0}

\begin{center}
\begin{tabular}{@{}lrr@{\hspace{30pt}}r@{}}
\toprule
\textbf{Task} & \textbf{Vertical-Slash} & \textbf{Query-Aware} & \textbf{Query-Aware range} \\
\midrule
@@TASK_ROWS@@
\bottomrule
\end{tabular}
\end{center}

\textbf{Query-Aware Usage at Layers 8 and 31}

\begin{center}
\begin{tabular}{@{}rrrr@{}}
\toprule
\textbf{Layer} & \textbf{Head ID} & \textbf{Samples using Query-Aware} & \textbf{Total samples} \\
\midrule
8 & 16 & 1 & 200 \\
31 & 14 & 4 & 200 \\
\bottomrule
\end{tabular}
\end{center}

\newpage
\begin{center}
  {\Large\bfseries FlexPrefill Mean Head Counts by Layer}
\end{center}
Layer and head IDs are zero-based. Each layer has 32 query heads;
means are computed over 200 inputs.

\begin{center}
\renewcommand{\arraystretch}{1.02}
\begin{tabular}{@{}rrrr@{}}
\toprule
\textbf{Layer} & \textbf{Vertical-Slash} & \textbf{Query-Aware} & \textbf{Query-Aware range} \\
\midrule
@@LAYER_ROWS@@
\bottomrule
\end{tabular}
\end{center}
\end{document}
'''
tex = tex.replace('@@SCORE_ROWS@@','\n'.join(score_rows)).replace('@@LAYER_ROWS@@', '\n'.join(layer_rows)).replace('@@TASK_ROWS@@', '\n'.join(task_rows))
# Audited causal-pair sparsity supplement (2026-10-07).
sparsity = json.loads((HERE/'prefill_sparsity_results.json').read_text())
assert sparsity['status'] == 'PASS' and len(sparsity['rows']) == 28
sparsity_lookup = {(r['method'],r['task']):r for r in sparsity['rows']}
sparsity_tasks = ['kv_retrieval','longbook_choice_eng','longbook_qa_eng','code_debug']
sparsity_methods = ['dense_full','coverage_token_vertical','coverage_token_vertical_slash',
    'random_token_vertical','random_token_vertical_slash','flexprefill','minference']
sparsity_rows = []
for label,method in zip(labels,sparsity_methods):
    values = [format(sparsity_lookup[(method,task)]['mask_sparsity_percent'],'.2f') for task in sparsity_tasks]
    sparsity_rows.append(label+' & '+' & '.join(values)+r' \\')
annex = r'''
\newpage
\begin{center}
{\Large\bfseries Prefill Mask Sparsity}
\end{center}

Percentage of valid causal query--key token pairs removed. Dense causal
attention is the reference; future positions and padding are excluded.

\begingroup
\setlength{\tabcolsep}{5pt}
\begin{tabularx}{\textwidth}{@{}Xrrrr@{}}
\toprule
\textbf{Method} & \textbf{KV Retrieval} & \textbf{En.MC} & \textbf{En.QA} & \textbf{Code.Debug} \\
\midrule
@@SPARSITY_ROWS@@
\bottomrule
\end{tabularx}
\endgroup

\vspace{6pt}
\textbf{Definition:} Sparsity = 100\% ×
(1 − retained valid causal pairs / total valid causal pairs).
For an input of N tokens, the denominator per head is N(N+1)/2.

\textbf{Aggregation:} Equal averages over 32 query heads, 32 layers,
and 50 complete inputs per task. Statistics cover prefill only.

\textbf{Custom methods:} Counts are taken from the saved token masks of
the original scoring runs, including the union of Token, Vertical, and Slash
where applicable.

\textbf{FlexPrefill:} Counts use the final selected 128×128 blocks,
including both Vertical-Slash and Query-Aware additions. Partial edge blocks
are clipped to the input length and the causal region.

\textbf{MInference:} Counts use the actual converted 64-token key intervals
and individual columns passed to the kernel. The union is counted at token
level after causal clipping, including expansion of the selected slash pattern.

\textbf{Baseline measurement:} The original 200 full inputs were replayed
once for each baseline using the frozen scoring configuration. Every one of
409,600 head masks was checked. Baseline sparsity describes these replays;
mask indices were not saved in the original scoring runs.

\textbf{Observer validation:} With and without the observer, attention
outputs matched bitwise at layers 0, 15, and 31 on a 4,097-token test, and
full-model prefill logits matched bitwise on one complete original input
for each baseline.

\textbf{Parameters:} FlexPrefill: γ=0.95, τ=0.1,
min\_budget=1024, block\_size=128. MInference: the frozen
Llama-3.1-8B-Instruct 128k head-pattern file; ratio=1; starting layer=0.
'''
annex = annex.replace('@@SPARSITY_ROWS@@','\n'.join(sparsity_rows))
tex = tex.replace(r'\end{document}',annex+r'\end{document}')

(HERE/'mask_flexprefill_results.tex').write_text(tex, encoding='utf-8')
print('Generated mask_flexprefill_results.tex: seven score rows, mask configuration, FlexPrefill usage, 32 layers.')
