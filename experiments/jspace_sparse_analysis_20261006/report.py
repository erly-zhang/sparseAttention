"""Scientific PDF export, distinct from the token exploration viewer."""
import subprocess,json
from pathlib import Path
from common import HERE,SOURCE,TASKS,METHODS,LABELS,TASK_LABELS,read

def main():
    s=read(SOURCE/'outputs/summary.json');rfile=HERE/'outputs/readout-summary.json';jfile=HERE/'outputs/jspace-summary.json'
    font=HERE.parents[1]/'reports/mask_flexprefill_results_20261002/fonts'
    tex=r'''\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}\usepackage{fontspec}\usepackage{graphicx}\usepackage{booktabs}\usepackage{hyperref}
\setmainfont[Path=FONT/,BoldFont=TimesNewRoman-Bold.ttf,ItalicFont=TimesNewRoman-Italic.ttf]{TimesNewRoman-Regular.ttf}
\begin{document}\begin{center}\Large Dense--Sparse Prefill: J-lens and J-space Analysis\end{center}
\noindent Llama-3.1-8B-Instruct; four InfiniteBench tasks, 50 prompts per task; seven methods, all 32 layers. J-lens fitted on 1,000 independent WikiText sequences of 128 tokens.
\paragraph{Scope.} These measurements compare internal representations while the model reads each full long-context prompt. Activations were saved at sampled prefix, tail, uniformly spaced and retrieval-evidence positions. They do not contain generated reasoning or new answers. Historical task scores are association labels, not newly measured accuracy.
\paragraph{Readout differences.} Random-head methods show larger mean tail readout divergence than the corresponding calibrated-head methods on all four tasks. Adding slash reduces calibrated-head tail divergence on all four tasks; for random heads, the KV Retrieval divergence increases slightly. This is a readout comparison, not evidence that a particular reasoning step was lost or restored. Sparse variants are not matched in attention budget.
\begin{center}\begin{tabular}{lrrrr}\toprule Method & KV Retrieval & En.MC & En.QA & Code.Debug\\\midrule
ROWS
\bottomrule\end{tabular}\end{center}
\noindent Table: mean tail JS divergence from Dense in nats, averaged within prompts and over layers. Dense has zero self-divergence.
\paragraph{Historical replay checks.} The prefill audit covers 1,400 paired observations. 1,398 reproduce the historical first predicted token, and 1,357 have bitwise-equal final logits. Replay flags are retained; agreement of one token does not establish agreement of a generated answer.
\paragraph{Readout quality.} The original 551-item concept test shows J-lens outperforming Logit Lens in five categories at Top-10 recovery, but not poetry. Short-text validation does not by itself establish long-context interpretation quality.
JSPACE
\clearpage\includepdf[pages=-]{prefill_analysis.pdf}
\end{document}'''
    tex=tex.replace('\\usepackage{hyperref}','\\usepackage{hyperref}\n\\usepackage{pdfpages}')
    rows=[]
    for mi,m in enumerate(METHODS):
        values=[s['groups'][t][m]['tail']['mean'] for t in TASKS]
        rows.append(LABELS[mi]+' & '+' & '.join('%.4f'%v for v in values)+r' \\')
    if jfile.exists():
        paragraph=r'''\paragraph{J-space decomposition.} Unit-normalized token directions include the model's final RMSNorm gains. A local nonnegative gradient-pursuit implementation uses at most 16 or 25 selected atoms, with all vocabulary tokens eligible. We compare uncentered residual energy reconstructed with a seeded orthogonal-rotation control preserving dictionary norms and Gram matrix. This is not labeled centered variance explained. Independently fitted Dense and Sparse supports need not be unique, and may exchange correlated token directions. The signed projection of Sparse--Dense change onto Dense active directions is reported separately from the nonnegative reconstruction. The decomposition and all original input hashes passed the final audit.'''
    else:paragraph=r'\paragraph{J-space status.} The nonnegative decomposition is still running. The attached figures currently report J-lens readouts only; no J-space conclusion is asserted.'
    tex=tex.replace('FONT',str(font)).replace('ROWS','\n'.join(rows)).replace('JSPACE',paragraph)
    out=HERE/'figures';(out/'analysis_report.tex').write_text(tex)
    p=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','analysis_report.tex'],cwd=out,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'latex-build.log').write_bytes(p.stdout)
    if p.returncode:raise RuntimeError('PDF compilation failed; see latex-build.log')
    print('REPORT READY',flush=True)
if __name__=='__main__':main()
