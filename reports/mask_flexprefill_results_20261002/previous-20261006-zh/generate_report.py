import json
from pathlib import Path

from validate_results import validate

HERE=Path(__file__).resolve().parent
verified=validate()
source = HERE/'../../experiments/flexprefill_head_usage_retry01_20261002/outputs/independent-audit.json'
data = json.loads(source.read_text())
assert data['status'] == 'PASS' and data['samples'] == 200
labels=[r'Dense',r'\zh{校准选头：}Token+Vertical',r'\zh{校准选头：}Token+Vertical+Slash',
        r'\zh{随机选头：}Token+Vertical',r'\zh{随机选头：}Token+Vertical+Slash',
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
\IfFileExists{fonts/NotoSerifSC-Report-Regular.ttf}{
  \newfontfamily\CJKfont[Path=fonts/,BoldFont=NotoSerifSC-Report-Bold.ttf]{NotoSerifSC-Report-Regular.ttf}
}{
  \IfFontExistsTF{Source Han Serif CN}{\newfontfamily\CJKfont[AutoFakeBold=2]{Source Han Serif CN}}{
    \IfFontExistsTF{Noto Serif CJK SC}{\newfontfamily\CJKfont[AutoFakeBold=2]{Noto Serif CJK SC}}
      {\newfontfamily\CJKfont[AutoFakeBold=2]{SimSun}}
  }
}
\newcommand{\zh}[1]{{\CJKfont #1}}
\newcommand{\en}[1]{{\rmfamily #1}}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.12}
\hypersetup{pdftitle={稀疏 prefill 实验结果与 FlexPrefill 分层统计},pdfauthor={}}
\begin{document}
\begin{center}
  {\Large\bfseries\zh{稀疏} prefill \zh{实验结果}}
\end{center}

{\bfseries\zh{模型与样本：}} Llama-3.1-8B-Instruct\zh{；}
32 \zh{层，每层} 32 \zh{个} query heads\zh{。}\par
KV Retrieval\zh{、}En.MC\zh{、}En.QA\zh{、}Code.Debug
\zh{各} 50 \zh{个样本，共} 200 \zh{个样本。}

\vspace{5pt}
{\large\bfseries 1. \zh{任务评分}}

\zh{各任务官方评分的样本均值} ×100\zh{；列内最高值加粗。所有方法应用于全部} 32 \zh{层。}

\begingroup
\setlength{\tabcolsep}{5pt}
\begin{tabularx}{\textwidth}{@{}Xrrrr@{}}
\toprule
\textbf{\zh{方法}} & \textbf{KV Retrieval} & \textbf{En.MC} & \textbf{En.QA} & \textbf{Code.Debug} \\
\midrule
@@SCORE_ROWS@@
\bottomrule
\end{tabularx}
\endgroup

\vspace{3pt}
{\large\bfseries 2. Mask \zh{配置}}

\begin{tabularx}{\textwidth}{@{}p{32mm}X@{}}
\toprule
\textbf{\zh{项目}} & \textbf{\zh{当前实现}} \\
\midrule
\zh{自定义} mask \zh{粒度} & Key token\zh{级别，不扩展为整块。} \\
\zh{基础} Token mask & \en{TopK=8192}\zh{；每个包含} \en{128} \zh{个} \en{query} \zh{的} \en{tile} \zh{共用一组} \en{key tokens}\zh{。} \\
Vertical & \zh{加入具体的} \en{key token} \zh{列。} \\
Slash & \zh{逐} \en{query} \zh{加入满足距离} \en{\textit{d=i−j}} \zh{的} \en{key}\zh{，即} \en{\textit{j=i−d}}\zh{。} \\
\zh{同层共享} & \zh{每层选定一个代表头，构造} \en{mask} \zh{并共享给同层全部} \en{32} \zh{个} \en{query heads}\zh{。} \\
\zh{注意力计算} & \zh{各头使用自己的} \en{Q/K/V}\zh{，在共享} \en{mask} \zh{内分别归一化。} \\
FlexPrefill baseline & \en{128×128 block} \zh{粒度。} \\
\bottomrule
\end{tabularx}

\newpage
\begin{center}
{\Large\bfseries FlexPrefill \zh{算法使用统计}}
\end{center}

\zh{参数：}γ=0.95\zh{，}τ=0.1\zh{，}min\_budget=1024\zh{，}block\_size=128\zh{。}

\zh{统计来源：原} 200 \zh{个输入的同配置} prefill \zh{重放，每个输入观测两次。}
\par\zh{计数以单轮为准；两轮} 204800 \zh{次} head \zh{分类的重复一致率为} 100\%\zh{。}

\begin{tabularx}{\textwidth}{@{}lXrr@{}}
\toprule
\textbf{\zh{算法分支}} & \textbf{\zh{选择条件}} & \textbf{\zh{头次}} & \textbf{\zh{占比}} \\
\midrule
Vertical-Slash & √JS ≥ 0.1 & 203750 & 99.49\% \\
Query-Aware Block-Sparse & √JS < 0.1 & 1050 & 0.51\% \\
\bottomrule
\end{tabularx}

\zh{头次：样本数} × \zh{层数} × \zh{每层头数} = 200×32×32 = 204800\zh{。}
\zh{按实际进入的算法分支分类；}Query-Aware \zh{分支保留最小预算的} Vertical-Slash blocks\zh{。}

\vspace{4pt}
\textbf{\zh{第} 0 \zh{层分任务平均头数}}

\begin{center}
\begin{tabular}{@{}lrr@{\hspace{30pt}}r@{}}
\toprule
\textbf{\zh{任务}} & \textbf{Vertical-Slash} & \textbf{Query-Aware} & \textbf{\zh{后者范围}} \\
\midrule
@@TASK_ROWS@@
\bottomrule
\end{tabular}
\end{center}

\textbf{\zh{第} 8\zh{、}31 \zh{层的} Query-Aware \zh{使用记录}}

\begin{center}
\begin{tabular}{@{}rrrr@{}}
\toprule
\textbf{\zh{层}} & \textbf{Head ID} & \textbf{\zh{使用样本数}} & \textbf{\zh{样本总数}} \\
\midrule
8 & 16 & 1 & 200 \\
31 & 14 & 4 & 200 \\
\bottomrule
\end{tabular}
\end{center}

\newpage
\begin{center}
  {\Large\bfseries FlexPrefill \zh{逐层平均头数}}
\end{center}
\zh{层与} Head ID \zh{均从} 0 \zh{开始编号。每层} 32 \zh{个} query heads\zh{；平均值基于} 200 \zh{个输入。}

\begin{center}
\renewcommand{\arraystretch}{1.02}
\begin{tabular}{@{}rrrr@{}}
\toprule
\textbf{\zh{层}} & \textbf{Vertical-Slash} & \textbf{Query-Aware} & \textbf{\zh{后者范围}} \\
\midrule
@@LAYER_ROWS@@
\bottomrule
\end{tabular}
\end{center}
\end{document}
'''
tex = tex.replace('@@SCORE_ROWS@@','\n'.join(score_rows)).replace('@@LAYER_ROWS@@', '\n'.join(layer_rows)).replace('@@TASK_ROWS@@', '\n'.join(task_rows))
(HERE/'mask_flexprefill_results.tex').write_text(tex, encoding='utf-8')
print('Generated mask_flexprefill_results.tex: seven score rows, mask configuration, FlexPrefill usage, 32 layers.')
