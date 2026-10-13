"""English Times New Roman results supplement from reviewed measurements."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / 'review'
METHODS = ['dense_full', 'coverage_token_vertical', 'coverage_token_vertical_slash',
           'random_token_vertical', 'random_token_vertical_slash', 'flexprefill', 'minference']
LABELS = ['Dense', 'Calibrated Token+Vertical', 'Calibrated Token+Vertical+Slash',
          'Random Token+Vertical', 'Random Token+Vertical+Slash', 'FlexPrefill', 'MInference']
TASKS = ['kv_retrieval', 'longbook_choice_eng', 'longbook_qa_eng', 'code_debug']
TASK_LABELS = ['KV Retrieval', 'En.MC', 'En.QA', 'Code.Debug']


def main():
    reviewed = json.loads((OUT / 'reviewed-task-summary.json').read_text())
    assert reviewed['status'] == 'PASS'
    readout = json.loads((HERE / 'outputs/readout-summary.json').read_text())
    def value(task, method, metric, k=25):
        rows = [r for r in reviewed['task_means'] if r['task'] == task
                and r['method'] == method and r['metric'] == metric and r['k'] == k]
        assert len(rows) == 1
        return rows[0]['mean']
    def table(metric, scale=100, decimals=2, k=25):
        return '\n'.join(label + ' & ' + ' & '.join(
            f'{value(t, m, metric, k)*scale:.{decimals}f}' for t in TASKS) + r' \\'
            for m, label in zip(METHODS, LABELS))
    def js_table():
        rows = []
        for m, label in zip(METHODS, LABELS):
            values = []
            for t in TASKS:
                matches = [r for r in readout['layer_means'] if r['task'] == t
                           and r['method'] == m and r['group'] == 'tail' and r['metric'] == 'js']
                assert len(matches) == 32
                values.append(sum(r['mean'] for r in matches) / 32)
            rows.append(label + ' & ' + ' & '.join(f'{v:.4f}' for v in values) + r' \\')
        return '\n'.join(rows)
    def contrast_table():
        rows = []
        for task, label in zip(TASKS, TASK_LABELS):
            fields = []
            for name in ['calibrated_slash_minus_vertical', 'random_slash_minus_vertical']:
                r, = [r for r in reviewed['paired_contrasts'] if r['task'] == task
                      and r['k'] == 25 and r['contrast'] == name
                      and r['metric'] == 'dense_active_support_recall']
                fields.append('%.2f [%.2f, %.2f]' % tuple(r[x]*100 for x in
                              ['mean', 'bootstrap_95_low', 'bootstrap_95_high']))
            rows.append(label + ' & ' + ' & '.join(fields) + r' \\')
        return '\n'.join(rows)
    def energy_table():
        return '\n'.join(label + ' & ' + ' & '.join(f'{x:.2f}' for x in [
            value(t, METHODS[0], 'energy_reconstructed')*100,
            value(t, METHODS[2], 'energy_reconstructed')*100,
            value(t, METHODS[0], 'random_energy_reconstructed')*100,
            value(t, METHODS[0], 'excess_energy_reconstructed')*100]) + r' \\'
            for t, label in zip(TASKS, TASK_LABELS))
    def reconstruction_table():
        return '\n'.join(label + ' & ' + ' & '.join(f'{x:.3f}' for x in [
            value(t, METHODS[2], 'reconstruction_relative_l2'),
            value(t, METHODS[2], 'remainder_relative_l2'),
            value(t, METHODS[5], 'reconstruction_relative_l2'),
            value(t, METHODS[5], 'remainder_relative_l2')]) + r' \\'
            for t, label in zip(TASKS, TASK_LABELS))
    font = HERE.parents[1] / 'reports/mask_flexprefill_results_20261002/fonts'
    template = r'''\documentclass[10pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{fontspec,booktabs,hyperref,pdfpages}
\setmainfont[Path=FONT/,BoldFont=TimesNewRoman-Bold.ttf,ItalicFont=TimesNewRoman-Italic.ttf]{TimesNewRoman-Regular.ttf}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\newcommand{\taskheader}{Method & KV Retrieval & En.MC & En.QA & Code.Debug\\}
\begin{document}
\begin{center}{\Large Dense--Sparse Prefill: Reviewed Results}\end{center}
Llama-3.1-8B-Instruct; four InfiniteBench tasks, 50 prompts per task; seven methods; all 32 layers. One Dense J-lens fitted on 1,000 WikiText-103 sequences of 128 tokens. Analysis completed October 6, 2026, 21:01 EDT.

All tables average the 128 sampled prompt-tail positions within each prompt, then average over 32 layers and 50 prompts. Full long-context forwards were used; saved activations contain prompt positions, with no generated response trajectory.

\textbf{Dense active token-direction recall (\%), at most 25 atoms}
\begin{center}\begin{tabular}{lrrrr}\toprule\taskheader\midrule
RECALL25
\bottomrule\end{tabular}\end{center}
Recall counts Dense's active vocabulary token IDs also active in each Sparse fit at the same layer and position. It measures exact dictionary support overlap, rather than semantic concept retention or task accuracy. All vocabulary atoms are eligible; coefficients are nonnegative.

\textbf{Effect of adding Slash on support recall (percentage points)}
\begin{center}\begin{tabular}{lrr}\toprule Task & Calibrated V+S minus V & Random V+S minus V\\\midrule
CONTRASTS
\bottomrule\end{tabular}\end{center}
Brackets show pointwise, unadjusted 95\% paired percentile bootstrap intervals: 5,000 draws of the 50 prompt IDs, after averaging layers within each prompt. Pattern variants are not matched in attention budget.

\textbf{Reconstructed residual energy (\%), at most 25 atoms}
\begin{center}\begin{tabular}{lrrrr}\toprule Task & Dense & Calibrated V+S & Dense rotated control & Dense excess (pp)\\\midrule
ENERGY
\bottomrule\end{tabular}\end{center}
Energy is one minus squared reconstruction error divided by squared residual norm; it is uncentered. The seeded orthogonal-rotation control preserves dictionary size, atom norms and Gram matrix. Excess subtracts control energy from Dense energy.

\clearpage
\textbf{Dense active token-direction recall (\%), at most 16 atoms}
\begin{center}\begin{tabular}{lrrrr}\toprule\taskheader\midrule
RECALL16
\bottomrule\end{tabular}\end{center}

\textbf{Mean J-lens readout JS divergence from Dense (nats)}
\begin{center}\begin{tabular}{lrrrr}\toprule\taskheader\midrule
READOUTROWS
\bottomrule\end{tabular}\end{center}
This readout metric compares full-vocabulary distributions. It is distinct from the support-recall metric above.

\textbf{Changes in decomposition components relative to Dense, at most 25 atoms}
\begin{center}\begin{tabular}{lrrrr}\toprule
& \multicolumn{2}{c}{Calibrated V+S} & \multicolumn{2}{c}{FlexPrefill}\\
Task & Reconstruction & Remainder & Reconstruction & Remainder\\\midrule
COMPONENTS
\bottomrule\end{tabular}\end{center}
Each entry is the L2 norm of the paired component change divided by the corresponding Dense component norm, before averaging. Independently fitted reconstruction and remainder changes do not form an orthogonal, additive energy partition.

\textbf{Checks and metric scope}

All 128 workers and 6,400 sample--layer pairs are complete, covering 44,800 method--sample--layer pairs at both sparsity settings. Source hash verification and final experiment audit: PASS. Independent review reconciles all per-layer means and checks all expected sample IDs, methods and sparsity settings. Direct tensor spot checks of token IDs, active coefficients and support recall pass on one layer-16 case per task. Floating-point arithmetic is FP32 with TF32 disabled; all 128 CPU/GPU pursuit checks pass.

The local nonnegative gradient-pursuit approximation selects at most 16 or 25 atoms. Dictionary directions are correlated and decomposition support is not unique. Setting the atom cap to 25 is not a measurement of J-space occupancy. The energy tables are not the paper's centered variance-explained metric. No new generation, task accuracy, inference speed or causal intervention is measured here.

Source outputs: completed experiment \textit{jspace\_sparse\_analysis\_20261006}; independently reviewed task means and paired intervals: \textit{reviewed-task-summary.json}. Definitions are documented in the frozen experiment protocol. Paper: \href{https://transformer-circuits.pub/2026/workspace/index.html}{\textit{Transformer Circuits: Human-Interpretable Workspace}}.
APPENDIX
\end{document}
'''
    replacements = dict(FONT=str(font), RECALL25=table('dense_active_support_recall'),
                        RECALL16=table('dense_active_support_recall', k=16),
                        CONTRASTS=contrast_table(), ENERGY=energy_table(),
                        READOUTROWS=js_table(), COMPONENTS=reconstruction_table())
    for key, value_ in replacements.items():
        template = template.replace(key, value_)
    for name, appendix in [
        ('results_summary', ''),
        ('results_with_figures', r'\clearpage\includepdf[pages=-]{../figures/prefill_analysis.pdf}'),
    ]:
        (OUT / (name + '.tex')).write_text(template.replace('APPENDIX', appendix))
        result = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                                 name + '.tex'], cwd=OUT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (OUT / (name + '-build.log')).write_bytes(result.stdout)
        if result.returncode:
            raise RuntimeError(name + ' PDF compilation failed')
        result = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                                 name + '.tex'], cwd=OUT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (OUT / (name + '-build.log')).write_bytes(result.stdout)
        if result.returncode:
            raise RuntimeError(name + ' final PDF compilation failed')
    provenance = dict(summary_sha256=hashlib.sha256(
        (OUT / 'reviewed-task-summary.json').read_bytes()).hexdigest(),
        original_figures_sha256=hashlib.sha256(
            (HERE / 'figures/prefill_analysis.pdf').read_bytes()).hexdigest(),
        original_outputs_unchanged=True, font='Times New Roman', language='English')
    (OUT / 'report-provenance.json').write_text(json.dumps(provenance, indent=2))
    print('REVIEW REPORTS READY', flush=True)


if __name__ == '__main__':
    main()
