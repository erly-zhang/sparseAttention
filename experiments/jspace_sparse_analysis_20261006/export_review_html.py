"""Export the reviewed scientific report as one offline HTML file.

The numerical tables use the same reviewed sources as the PDF. Original PNGs
and Times New Roman fonts are embedded unchanged. Frozen outputs are read only.
"""
from pathlib import Path
from html import escape
import base64
import hashlib
import json

HERE = Path(__file__).resolve().parent
OUT = HERE / 'review'
METHODS = ['dense_full', 'coverage_token_vertical', 'coverage_token_vertical_slash',
           'random_token_vertical', 'random_token_vertical_slash', 'flexprefill', 'minference']
LABELS = ['Dense', 'Calibrated Token+Vertical', 'Calibrated Token+Vertical+Slash',
          'Random Token+Vertical', 'Random Token+Vertical+Slash', 'FlexPrefill', 'MInference']
TASKS = ['kv_retrieval', 'longbook_choice_eng', 'longbook_qa_eng', 'code_debug']
TASK_LABELS = ['KV Retrieval', 'En.MC', 'En.QA', 'Code.Debug']
FIGURES = [
    ('tail_js', 'J-lens readout divergence at prompt-tail positions',
     'Full-vocabulary JS divergence from Dense in nats; lower values indicate closer readouts.'),
    ('uniform_js', 'J-lens readout divergence across the full context',
     'Uniformly spaced saved positions from complete long-context prefills; JS divergence in nats.'),
    ('slash_effect', 'Effect of adding Slash on readout divergence',
     'Paired change: Vertical+Slash minus Vertical. Negative values indicate closer Dense readouts. '
     'Intervals are 95% prompt-bootstrap intervals; attention budgets differ.'),
    ('lens_validation', 'Readout validation on the original 551 prompt items',
     'Concept recovery in Top-10 at any layer, comparing J-lens and Logit Lens. '
     'This short-text validation is separate from the InfiniteBench analysis.'),
    ('top10_recall', 'Dense Top-10 readout retention',
     'Fraction of Dense Top-10 vocabulary entries retained in Sparse Top-10 at matched positions and layers.'),
    ('top1_preserved', 'Dense Top-1 readout agreement',
     'Fraction of matched positions with the same highest-ranked readout token as Dense.'),
    ('residual_relative_l2', 'Residual representation change',
     'L2 norm of the Sparse–Dense residual change, divided by the corresponding Dense residual norm.'),
    ('transported_relative_l2', 'Representation change after J-lens transport',
     'Relative L2 change after applying the same Dense-fitted J-lens to both representations.'),
    ('jspace_dense_active_support_recall_16', 'Dense active token-direction recall · k = 16',
     'Exact active vocabulary-token support overlap. Correlated directions can exchange without implying semantic concept loss.'),
    ('jspace_excess_energy_reconstructed_16', 'Energy above the rotated control · k = 16',
     'Uncentered reconstructed residual energy minus the seeded orthogonal-rotation control; this is not centered variance explained.'),
    ('jspace_delta_energy_in_dense_active_span_16', 'Change energy in the Dense active span · k = 16',
     'Signed orthogonal projection of the Sparse–Dense change onto Dense’s selected active-atom span. '
     'This span is distinct from the entire J-space union of cones.'),
    ('jspace_dense_active_support_recall_25', 'Dense active token-direction recall · k = 25',
     'Exact active vocabulary-token support overlap. Correlated directions can exchange without implying semantic concept loss.'),
    ('jspace_excess_energy_reconstructed_25', 'Energy above the rotated control · k = 25',
     'Uncentered reconstructed residual energy minus the seeded orthogonal-rotation control; this is not centered variance explained.'),
    ('jspace_delta_energy_in_dense_active_span_25', 'Change energy in the Dense active span · k = 25',
     'Signed orthogonal projection of the Sparse–Dense change onto Dense’s selected active-atom span. '
     'This span is distinct from the entire J-space union of cones.'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_url(path, mime):
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def table(headers, rows, table_id, caption):
    head = ''.join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = []
    for label, cells in rows:
        values = ''.join(f'<td>{escape(str(c))}</td>' for c in cells)
        body.append(f'<tr><th scope="row">{escape(label)}</th>{values}</tr>')
    return (f'<div class="table-scroll" tabindex="0" role="region" aria-label="{escape(caption)}">'
            f'<table id="{table_id}"><caption>{escape(caption)}</caption>'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def main():
    reviewed_path = OUT / 'reviewed-task-summary.json'
    readout_path = HERE / 'outputs/readout-summary.json'
    reviewed = json.loads(reviewed_path.read_text())
    readout = json.loads(readout_path.read_text())
    assert reviewed['status'] == 'PASS'
    provenance = {'reviewed_summary_sha256': digest(reviewed_path),
                  'readout_summary_sha256': digest(readout_path),
                  'language': 'English', 'font': 'Times New Roman',
                  'sources_modified': False, 'figures': {}, 'fonts': {}}
    for source_name, expected in reviewed['inputs'].items():
        assert digest(HERE / source_name) == expected, source_name

    def value(task, method, metric, k=25):
        row, = [r for r in reviewed['task_means'] if r['task'] == task
                and r['method'] == method and r['metric'] == metric and r['k'] == k]
        return row['mean']

    support_tables = []
    for k in (25, 16):
        rows = [(label, [f'{value(t, m, "dense_active_support_recall", k)*100:.2f}'
                         for t in TASKS]) for m, label in zip(METHODS, LABELS)]
        support_tables.append(
            f'<div data-recall-panel="{k}" id="recall-panel-{k}">'
            + table(['Method'] + TASK_LABELS, rows, f'recall-{k}',
                    f'Dense active token-direction recall (%), at most {k} atoms') + '</div>')

    contrast_rows = []
    for task, label in zip(TASKS, TASK_LABELS):
        cells = []
        for name in ['calibrated_slash_minus_vertical', 'random_slash_minus_vertical']:
            row, = [r for r in reviewed['paired_contrasts'] if r['task'] == task
                    and r['k'] == 25 and r['contrast'] == name
                    and r['metric'] == 'dense_active_support_recall']
            cells.append('%.2f [%.2f, %.2f]' % tuple(row[x]*100 for x in
                         ['mean', 'bootstrap_95_low', 'bootstrap_95_high']))
        contrast_rows.append((label, cells))
    contrasts = table(['Task', 'Calibrated V+S minus V', 'Random V+S minus V'],
                      contrast_rows, 'slash-gains',
                      'Effect of adding Slash on support recall (percentage points), k = 25')

    energy_rows = []
    components_rows = []
    for task, label in zip(TASKS, TASK_LABELS):
        energy_rows.append((label, [f'{x*100:.2f}' for x in [
            value(task, METHODS[0], 'energy_reconstructed'),
            value(task, METHODS[2], 'energy_reconstructed'),
            value(task, METHODS[0], 'random_energy_reconstructed'),
            value(task, METHODS[0], 'excess_energy_reconstructed')]]))
        components_rows.append((label, [f'{x:.3f}' for x in [
            value(task, METHODS[2], 'reconstruction_relative_l2'),
            value(task, METHODS[2], 'remainder_relative_l2'),
            value(task, METHODS[5], 'reconstruction_relative_l2'),
            value(task, METHODS[5], 'remainder_relative_l2')]]))
    energy = table(['Task', 'Dense', 'Calibrated V+S', 'Dense rotated control', 'Dense excess (pp)'],
                   energy_rows, 'energy-table', 'Reconstructed residual energy (%), at most 25 atoms')
    components = table(['Task', 'Calibrated V+S: reconstruction', 'Calibrated V+S: remainder',
                        'FlexPrefill: reconstruction', 'FlexPrefill: remainder'],
                       components_rows, 'components-table',
                       'Changes in decomposition components relative to Dense, at most 25 atoms')

    js_rows = []
    for method, label in zip(METHODS, LABELS):
        means = []
        for task in TASKS:
            rows = [r for r in readout['layer_means'] if r['task'] == task
                    and r['method'] == method and r['group'] == 'tail' and r['metric'] == 'js']
            assert len(rows) == 32 and all(r['n'] == 50 for r in rows)
            means.append(f'{sum(r["mean"] for r in rows)/32:.4f}')
        js_rows.append((label, means))
    js = table(['Method'] + TASK_LABELS, js_rows, 'js-table',
               'Mean J-lens readout JS divergence from Dense (nats)')

    font_dir = HERE.parents[1] / 'reports/mask_flexprefill_results_20261002/fonts'
    font_css = []
    for file, weight, style in [('TimesNewRoman-Regular.ttf', '400', 'normal'),
                                ('TimesNewRoman-Bold.ttf', '700', 'normal'),
                                ('TimesNewRoman-Italic.ttf', '400', 'italic')]:
        path = font_dir / file
        provenance['fonts'][file] = digest(path)
        font_css.append("@font-face{font-family:'Report Times';src:url('" +
                        data_url(path, 'font/ttf') + "') format('truetype');font-weight:" +
                        weight + ';font-style:' + style + ';font-display:swap;}')

    figures = []
    for i, (stem, title, caption) in enumerate(FIGURES, 1):
        path = HERE / 'figures' / (stem + '.png')
        provenance['figures'][stem] = digest(path)
        figures.append(
            f'<figure id="figure-{i}"><div class="figure-heading">'
            f'<span class="figure-number">Figure {i:02d}</span><h3>{escape(title)}</h3></div>'
            f'<button class="figure-zoom" type="button" data-title="{escape(title)}" '
            f'aria-label="Enlarge figure {i}: {escape(title)}">'
            f'<img src="{data_url(path, "image/png")}" alt="{escape(title)}" '
            f'loading="lazy" decoding="async"></button>'
            f'<figcaption>{escape(caption)}</figcaption></figure>')

    html = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Reviewed Dense–Sparse prefill results, J-lens readouts and J-space decomposition across four InfiniteBench tasks.">
<title>Dense–Sparse Prefill: Reviewed Results</title>
<style>
__FONTS__
:root{--ink:#20252b;--muted:#58616b;--accent:#235b86;--line:#dce1e6;--paper:#fff;--page:#f5f6f7;}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:var(--page);color:var(--ink);font:18px/1.55 'Report Times','Times New Roman',serif}
a{color:var(--accent);text-underline-offset:3px}button{font:inherit;color:inherit;cursor:pointer}
button:focus-visible,a:focus-visible,.table-scroll:focus-visible,summary:focus-visible{outline:3px solid #7ca7cb;outline-offset:4px}
.skip{position:absolute;left:10px;top:-80px;padding:10px;background:#fff;z-index:10}.skip:focus{top:10px}
.layout{display:grid;grid-template-columns:200px minmax(0,1fr);gap:36px;max-width:1440px;margin:auto;padding:38px 36px 60px}
aside{position:sticky;top:30px;align-self:start;font-size:16px}.contents-label{font-weight:bold;margin:0 0 16px;color:var(--muted)}
nav{display:flex;flex-direction:column;gap:5px}nav a{text-decoration:none;color:var(--muted);padding:7px 0 7px 12px;border-left:2px solid transparent}
nav a:hover,nav a[aria-current]{color:var(--accent);border-color:var(--accent);background:#edf2f6}
.aside-meta{border-top:1px solid var(--line);margin-top:24px;padding-top:15px;color:var(--muted);font-size:15px}
main{min-width:0;background:var(--paper);border:1px solid var(--line);padding:36px 44px 42px}
.eyebrow{font-size:15px;letter-spacing:.06em;color:var(--accent);margin:0 0 9px;text-transform:uppercase}
h1{font-size:36px;line-height:1.18;font-weight:400;margin:0 0 18px}h2{font-size:27px;font-weight:400;line-height:1.25;margin:0 0 18px}
h3{font-size:20px;line-height:1.3;margin:0;font-weight:400}p{margin:0 0 13px}
.lede{color:var(--muted)}.meta{display:flex;flex-wrap:wrap;gap:8px 22px;margin:18px 0;font-size:16px;color:var(--muted)}
.meta span{white-space:nowrap}.pass{color:#315f3f;font-weight:bold}.toolbar{display:flex;flex-wrap:wrap;gap:9px;margin-top:20px}
.button{border:1px solid #bdc8d2;border-radius:4px;background:#fff;padding:7px 12px;font-size:16px}.button:hover{background:#edf2f6}
section{margin-top:35px;padding-top:29px;border-top:1px solid var(--line)}.note{color:var(--muted);font-size:16px;margin-top:13px}
.table-scroll{overflow-x:auto;margin:18px 0 12px}table{border-collapse:collapse;width:100%;font-size:17px;font-variant-numeric:tabular-nums}
caption{text-align:left;font-size:17px;font-weight:bold;padding:0 0 12px;color:var(--ink)}
th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #e2e6ea;vertical-align:middle}thead th{font-size:16px;border-top:1px solid #8898a6;border-bottom:1px solid #8898a6}
th:first-child{text-align:left}tbody th{font-weight:400}td{white-space:nowrap}tbody tr:hover{background:#f6f8fa}
.selector{display:flex;align-items:center;gap:9px;flex-wrap:wrap;font-size:16px;margin:0 0 8px}
.selector button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:white}
.js-ready [data-recall-panel][hidden]{display:none!important}
.figure-intro{font-size:16px;color:var(--muted)}figure{margin:30px 0 40px;scroll-margin-top:20px}
.figure-heading{display:flex;align-items:baseline;gap:13px;margin-bottom:12px}.figure-number{color:var(--accent);font-size:14px;white-space:nowrap}
.figure-zoom{display:block;padding:0;border:1px solid var(--line);width:100%;background:white;cursor:zoom-in;line-height:0}
.figure-zoom img{display:block;width:100%;height:auto}figcaption{font-size:16px;color:var(--muted);margin-top:10px}
.figure-index{display:grid;grid-template-columns:1fr 1fr;gap:8px 22px;font-size:16px;margin-top:20px}.figure-index a{text-decoration:none}
details{margin:18px 0;border-top:1px solid var(--line);padding-top:14px}summary{cursor:pointer;font-weight:bold;margin-bottom:10px}
.check-list{display:grid;grid-template-columns:1fr 1fr;gap:12px 24px;margin:16px 0}.check-list div{border-left:2px solid #8da799;padding-left:12px}
.check-list strong{display:block;font-size:20px;font-weight:400}.check-list span{font-size:16px;color:var(--muted)}
footer{font-size:15px;color:var(--muted);border-top:1px solid var(--line);padding-top:20px;margin-top:28px}
dialog{border:0;border-radius:5px;max-width:96vw;width:1420px;max-height:94vh;padding:18px;box-shadow:0 20px 80px #0004;background:white}
dialog::backdrop{background:#101924c4}dialog header{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:12px}
dialog img{display:block;width:100%;height:auto}dialog h2{font-size:22px;margin:0}dialog .note{margin-bottom:0}
@media(max-width:1100px){.layout{grid-template-columns:1fr;gap:18px;padding:22px}aside{position:static}.contents-label,.aside-meta{display:none}nav{flex-direction:row;flex-wrap:wrap;gap:8px 16px}nav a{padding:4px 0;border-left:0;border-bottom:2px solid transparent}main{padding:30px}}
@media(max-width:650px){body{font-size:17px}.layout{padding:10px}main{padding:24px 18px}h1{font-size:29px}h2{font-size:24px}.figure-index,.check-list{grid-template-columns:1fr}th,td{padding:9px 10px}figure{margin:24px 0}dialog{padding:12px}nav{font-size:15px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{@page{size:A4;margin:15mm}body{background:white;font-size:11pt}.layout{display:block;padding:0;max-width:none}aside,.toolbar,.selector,.figure-index,.figure-intro,.skip,dialog{display:none!important}main{padding:0;border:0}h1{font-size:24pt}h2{font-size:18pt}h3{font-size:14pt}table{font-size:10pt}thead th,caption{font-size:10pt}th,td{padding:6px}section{margin-top:20px;padding-top:18px}.table-scroll{overflow:visible}thead{display:table-header-group}tr{break-inside:avoid}.js-ready [data-recall-panel][hidden]{display:block!important}figure{break-inside:avoid;break-before:page;margin:0 0 16px}.figure-zoom{border:0}figcaption,.note,.lede,.figure-number{font-size:10pt}.figure-heading{margin-top:12px}.check-list span{font-size:10pt}details>*{display:block!important}footer{font-size:9pt}a{color:inherit;text-decoration:none}}
</style>
</head>
<body>
<a class="skip" href="#report">Skip to report</a>
<div class="layout">
<aside><p class="contents-label">CONTENTS</p><nav aria-label="Report contents">
<a href="#overview">Experiment scope</a><a href="#support">Token-direction recall</a>
<a href="#slash">Effect of adding Slash</a><a href="#energy">Reconstructed energy</a>
<a href="#readouts">J-lens readouts</a><a href="#components">Component changes</a>
<a href="#figures">All 14 figures</a><a href="#checks">Checks and definitions</a></nav>
<p class="aside-meta">Llama-3.1-8B-Instruct<br>InfiniteBench · 4 tasks<br>50 prompts per task<br>32 layers · 7 methods</p></aside>
<main id="report">
<header id="overview"><p class="eyebrow">J-lens and J-space analysis</p>
<h1>Dense–Sparse Prefill: Reviewed Results</h1>
<p class="lede">Llama-3.1-8B-Instruct; four InfiniteBench tasks, 50 prompts per task; seven methods; all 32 layers. One Dense J-lens fitted on 1,000 WikiText-103 sequences of 128 tokens.</p>
<div class="meta"><span>Completed October 6, 2026 · 21:01 EDT</span><span class="pass">Audit and independent review: PASS</span></div>
<p>All tables average the 128 sampled prompt-tail positions within each prompt, then average over 32 layers and 50 prompts. Full long-context forwards were used; saved activations contain prompt positions, with no generated response trajectory.</p>
<div class="toolbar"><button class="button" id="print-report" type="button">Print / Save PDF</button><button class="button" id="download-data" type="button">Download reviewed data</button></div>
</header>
<section id="support"><h2>Dense active token-direction recall</h2>
<div class="selector" role="group" aria-label="Support recall atom cap"><span>Atom cap</span>
<button class="button" type="button" data-k="25" aria-pressed="true" aria-controls="recall-panel-25">k = 25</button>
<button class="button" type="button" data-k="16" aria-pressed="false" aria-controls="recall-panel-16">k = 16</button></div>
__SUPPORT__
<p class="note">Recall counts Dense’s active vocabulary token IDs also active in each Sparse fit at the same layer and position. It measures exact dictionary support overlap, rather than semantic concept retention or task accuracy. All vocabulary atoms are eligible; coefficients are nonnegative. This selector changes only the support-recall table.</p>
</section>
<section id="slash"><h2>Effect of adding Slash</h2>__CONTRASTS__
<p class="note">Brackets show pointwise, unadjusted 95% paired percentile bootstrap intervals: 5,000 draws of the 50 prompt IDs, after averaging layers within each prompt. Pattern variants are not matched in attention budget.</p></section>
<section id="energy"><h2>Reconstructed residual energy</h2>__ENERGY__
<p class="note">Energy is one minus squared reconstruction error divided by squared residual norm; it is uncentered. The seeded orthogonal-rotation control preserves dictionary size, atom norms and Gram matrix. Excess subtracts control energy from Dense energy.</p></section>
<section id="readouts"><h2>J-lens readout divergence</h2>__READOUTS__
<p class="note">This readout metric compares full-vocabulary distributions. It is distinct from the support-recall metric above.</p></section>
<section id="components"><h2>Changes in decomposition components</h2>__COMPONENTS__
<p class="note">Each entry is the L2 norm of the paired component change divided by the corresponding Dense component norm, before averaging. Independently fitted reconstruction and remainder changes do not form an orthogonal, additive energy partition.</p></section>
<section id="figures"><h2>Layerwise figures and validation</h2>
<p class="figure-intro">All 14 original figures are included. Click a figure to enlarge it; press Escape to close. Layerwise shaded intervals resample prompts, rather than treating positions as independent observations.</p>
<div class="figure-index">__FIGURE_INDEX__</div>__FIGURES__</section>
<section id="checks"><h2>Checks and metric scope</h2>
<div class="check-list"><div><strong>128 / 128</strong><span>Completed decomposition workers</span></div><div><strong>6,400</strong><span>Sample–layer pairs</span></div><div><strong>44,800</strong><span>Method–sample–layer pairs at both atom caps</span></div><div><strong>PASS</strong><span>Source hashes, experiment audit and independent review</span></div></div>
<p>Independent review reconciles all per-layer means and checks all expected sample IDs, methods and sparsity settings. Direct tensor spot checks of token IDs, active coefficients and support recall pass on one layer-16 case per task. Floating-point arithmetic is FP32 with TF32 disabled; all 128 CPU/GPU pursuit checks pass.</p>
<p>The local nonnegative gradient-pursuit approximation selects at most 16 or 25 atoms. Dictionary directions are correlated and decomposition support is not unique. Setting the atom cap to 25 is not a measurement of J-space occupancy. The energy tables are not the paper’s centered variance-explained metric. No new generation, task accuracy, inference speed or causal intervention is measured here.</p>
<details><summary>Sources and reproducibility</summary>
<p>Source outputs: completed experiment <em>jspace_sparse_analysis_20261006</em>. Independently reviewed task means and paired intervals: <em>reviewed-task-summary.json</em>. Readout means: <em>readout-summary.json</em>. Definitions are documented in the frozen experiment protocol.</p>
<p>The tables use the same reviewed measurements as the English PDF report. Original figure PNGs and Times New Roman regular, bold and italic font files are embedded. The report opens offline; no external assets are needed. The reviewed data download retains task means, paired contrasts, aggregation scope, bootstrap settings, source hashes and tensor-check scope.</p>
<p>Reference: <a href="https://transformer-circuits.pub/2026/workspace/index.html" target="_blank" rel="noopener noreferrer">Transformer Circuits workspace paper</a>.</p>
</details></section>
<footer>English · Times New Roman · Reviewed results and all 14 figures · Full-context prefill observations</footer>
</main></div>
<dialog id="figure-dialog" aria-labelledby="zoom-heading"><header><h2 id="zoom-heading"></h2><button class="button" id="close-dialog" type="button">Close</button></header><img id="zoom-image" alt=""><p class="note">Press Escape to close.</p></dialog>
<script type="application/json" id="reviewed-data">__DATA__</script>
<script type="application/json" id="report-provenance">__PROVENANCE__</script>
<script>
(() => {
  'use strict';
  document.documentElement.classList.add('js-ready');
  const panels = Array.from(document.querySelectorAll('[data-recall-panel]'));
  const controls = Array.from(document.querySelectorAll('[data-k]'));
  function selectCap(cap) {
    panels.forEach(panel => { panel.hidden = panel.dataset.recallPanel !== cap; });
    controls.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.k === cap)));
  }
  controls.forEach(button => button.addEventListener('click', () => selectCap(button.dataset.k)));
  selectCap('25');
  const dialog = document.getElementById('figure-dialog');
  const zoomImage = document.getElementById('zoom-image');
  const heading = document.getElementById('zoom-heading');
  const closeButton = document.getElementById('close-dialog');
  let previousFocus = null;
  document.querySelectorAll('.figure-zoom').forEach(button => {
    button.addEventListener('click', () => {
      if (typeof dialog.showModal !== 'function') return;
      const source = button.querySelector('img');
      previousFocus = button;
      zoomImage.src = source.src;
      zoomImage.alt = source.alt;
      heading.textContent = button.dataset.title;
      dialog.showModal();
      closeButton.focus();
    });
  });
  closeButton.addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    zoomImage.removeAttribute('src');
    if (previousFocus) previousFocus.focus();
  });
  document.getElementById('print-report').addEventListener('click', () => window.print());
  let printDetails = [];
  window.addEventListener('beforeprint', () => {
    printDetails = Array.from(document.querySelectorAll('details')).map(detail => [detail, detail.open]);
    printDetails.forEach(([detail]) => { detail.open = true; });
    document.querySelectorAll('figure img').forEach(img => { img.loading = 'eager'; });
  });
  window.addEventListener('afterprint', () => {
    printDetails.forEach(([detail, wasOpen]) => { detail.open = wasOpen; });
  });
  document.getElementById('download-data').addEventListener('click', () => {
    const blob = new Blob([document.getElementById('reviewed-data').textContent], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = 'reviewed-task-summary.json';
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  if ('IntersectionObserver' in window) {
    const navigation = Array.from(document.querySelectorAll('nav a'));
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        navigation.forEach(link => {
          if (link.hash === '#' + entry.target.id) link.setAttribute('aria-current','location');
          else link.removeAttribute('aria-current');
        });
      }
    }, {rootMargin:'0px 0px -65% 0px'});
    document.querySelectorAll('header#overview, main > section').forEach(section => observer.observe(section));
  }
})();
</script>
</body></html>'''
    replacements = {
        '__FONTS__': '\n'.join(font_css), '__SUPPORT__': '\n'.join(support_tables),
        '__CONTRASTS__': contrasts, '__ENERGY__': energy, '__READOUTS__': js,
        '__COMPONENTS__': components, '__FIGURES__': '\n'.join(figures),
        '__FIGURE_INDEX__': ''.join(f'<a href="#figure-{i}">{i:02d}. {escape(title)}</a>'
                                  for i, (_, title, _) in enumerate(FIGURES, 1)),
        '__DATA__': json.dumps(reviewed, ensure_ascii=False).replace('<', '\\u003c'),
        '__PROVENANCE__': json.dumps(provenance, ensure_ascii=False).replace('<', '\\u003c'),
    }
    for marker, content in replacements.items():
        assert marker in html
        html = html.replace(marker, content)
    destination = OUT / 'results_with_figures.html'
    destination.write_text(html)
    provenance['html_sha256'] = digest(destination)
    provenance['html_bytes'] = destination.stat().st_size
    (OUT / 'html-provenance.json').write_text(json.dumps(provenance, indent=2))
    print('HTML READY', destination, destination.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
