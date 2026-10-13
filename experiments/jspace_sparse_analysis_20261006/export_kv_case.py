"""Supplementary trace of one known-value prefix from saved observations only."""
from pathlib import Path
from html import escape
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgba

HERE = Path(__file__).resolve().parent
REVIEW = HERE / 'review'
SOURCE = HERE.parent / 'jlens_prefill_analysis_20261005'
METHODS = ['dense_full', 'coverage_token_vertical_slash', 'flexprefill']
LABELS = ['Dense', 'Calibrated V+S', 'FlexPrefill']


def main():
    case_path = REVIEW / 'kv_retrieval_case_check.json'
    case = json.loads(case_path.read_text())
    manifest = json.loads((SOURCE / 'input-manifest.json').read_text())
    item, = [i for i in manifest['evaluation'] if i['key'] == case['sample_key']]
    case['selection'] = 'Existing representative case: median tail-JS among calibrated V+S historical score-decline samples.'
    case['source_files_sha256'] = {}
    for method in METHODS:
        readout = SOURCE / 'outputs' / f"replay-{method}-{item['analysis_shard']}" / (item['key'] + '.json')
        history = Path(item['history'][method]['path'])
        assert case['methods'][method]['historical_score'] == json.loads(history.read_text())['score']
        case['methods'][method]['historical_prediction'] = json.loads(history.read_text())['prediction']
        for path in [readout, history]:
            case['source_files_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    case_path.write_text(json.dumps(case, indent=2))

    width, cell, left = 1080, 27, 174
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="205" viewBox="0 0 {width} 205" role="img" aria-labelledby="title desc">',
           '<title id="title">Correct-value prefix ca: Top-10 rank by layer</title>',
           '<desc id="desc">Dense reads ca in its Top-10 from layer 23, FlexPrefill from layer 27, and Calibrated V+S at no layer. Dash means outside the Top-10.</desc>',
           '<rect width="1080" height="205" fill="white"/>',
           '<g font-family="Times New Roman,serif" font-size="15" fill="#20252b">',
           '<text x="8" y="25">Layer (block output)</text>']
    for layer in range(32):
        svg.append(f'<text x="{left+layer*cell+12}" y="25" text-anchor="middle">{layer}</text>')
    for mi, (method, label) in enumerate(zip(METHODS, LABELS)):
        y = 44 + mi*39
        svg.append(f'<text x="8" y="{y+21}">{escape(label)}</text>')
        ranks = case['methods'][method]['target_first_piece_top10_rank_by_layer']
        for layer, rank in enumerate(ranks):
            color = '#edf0f3' if rank is None else ('#285f88' if rank <= 3 else '#bed4e6')
            ink = '#54606a' if rank is None else ('white' if rank <= 3 else '#20252b')
            x = left + layer*cell
            label_ = '—' if rank is None else str(rank)
            svg.append(f'<g><title>{escape(label)}, layer {layer}: {"outside Top-10" if rank is None else "rank " + str(rank)}</title>'
                       f'<rect x="{x}" y="{y}" width="25" height="31" rx="2" fill="{color}"/>'
                       f'<text x="{x+12}" y="{y+21}" text-anchor="middle" fill="{ink}">{label_}</text></g>')
    svg.extend(['<text x="174" y="189" fill="#58616b">Cell number = Top-10 rank; — = outside Top-10. One prompt, final prefill position.</text>', '</g></svg>'])
    font = HERE.parents[1] / 'reports/mask_flexprefill_results_20261002/fonts/TimesNewRoman-Regular.ttf'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Times New Roman', 'svg.fonttype': 'path'})
    colors = np.empty((3, 32, 4))
    for mi, method in enumerate(METHODS):
        for layer, rank in enumerate(case['methods'][method]['target_first_piece_top10_rank_by_layer']):
            colors[mi, layer] = to_rgba('#edf0f3' if rank is None else '#285f88' if rank <= 3 else '#bed4e6')
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.imshow(colors, aspect='auto')
    ax.set_xticks(range(32), range(32), fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Layer (block output)', fontsize=12, labelpad=12)
    ax.set_yticks(range(3), LABELS, fontsize=13)
    ax.tick_params(axis='both', length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for mi, method in enumerate(METHODS):
        for layer, rank in enumerate(case['methods'][method]['target_first_piece_top10_rank_by_layer']):
            ax.text(layer, mi, '—' if rank is None else str(rank), ha='center', va='center',
                    fontsize=11, color='white' if rank is not None and rank <= 3 else '#20252b')
    fig.text(.5, .03, 'Cell number = Top-10 rank; — = outside Top-10. One prompt, final prefill position.', ha='center', fontsize=11)
    fig.tight_layout(rect=(0, .11, 1, 1))
    fig.savefig(REVIEW / 'kv_correct_value_prefix_trace.svg', bbox_inches='tight')
    fig.savefig(REVIEW / 'kv_correct_value_prefix_trace.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    rows = []
    for method, label in zip(METHODS, LABELS):
        data = case['methods'][method]
        layers = [(l, rank) for l, rank in enumerate(data['target_first_piece_top10_rank_by_layer']) if rank is not None]
        first = str(layers[0][0]) if layers else 'Outside Top-10 at all layers'
        final = str(data['target_first_piece_top10_rank_by_layer'][31] or 'Outside Top-10')
        coefficients = data['target_first_piece_selected_coefficients_layer31_k25']
        coefficient = ', '.join(f'{c:.3f}' for c in coefficients) if coefficients else 'Not selected'
        rows.append(f'<tr><th scope="row">{escape(label)}</th><td>{data["historical_score"]:.0f}</td><td>{first}</td><td>{final}</td><td>{coefficient}</td></tr>')
    predictions = ''.join(f'<h3>{escape(label)}</h3><p class="prediction">{escape(case["methods"][method]["historical_prediction"])}</p>' for method, label in zip(METHODS, LABELS))
    generation = json.loads((REVIEW / 'kv_generation_history_check.json').read_text())
    generation_methods = ['dense_full', 'coverage_token_vertical', 'coverage_token_vertical_slash',
                          'random_token_vertical', 'random_token_vertical_slash', 'flexprefill', 'minference']
    generation_labels = ['Dense', 'Calibrated V', 'Calibrated V+S', 'Random V', 'Random V+S', 'FlexPrefill', 'MInference']
    generation_rows = ''.join(f'<tr><th scope="row">{escape(label)}</th>' + ''.join(
        f'<td>{generation["method_summary"][method][metric]}</td>' for metric in
        ['correct', 'token_limit_stops', 'failed_limit_stop_with_correct_prefix_ge8']) + '</tr>'
        for method, label in zip(generation_methods, generation_labels))
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KV Retrieval: Correct-Value Prefix Trace</title><link rel="stylesheet" href="assets/report.css">
<style>.layout{{display:block;max-width:1280px}}.prediction{{overflow-wrap:anywhere;padding:12px;background:#f5f7f9}}.case-table{{min-width:800px}}.trace{{overflow-x:auto}}.trace img{{min-width:950px;width:100%;height:auto}}</style></head>
<body><div class="layout"><main><header><p><a href="index.html">← Back to reviewed results</a></p>
<p class="eyebrow">Supplementary observation · one historical failure case</p><h1>KV Retrieval: Correct-Value Prefix Trace</h1>
<p>Sample <strong>{escape(case['sample_key'])}</strong>; full context: {item['input_tokens']:,} tokens; final prefill position: {case['position']:,}; all 32 layers.</p>
<p>Correct value: <strong>{escape(case['correct_value'])}</strong>. Its isolated tokenization has {case['target_token_count']} tokens. This trace tracks only the first piece <strong>“ca”</strong> (token ID {case['target_first_token_id']}).</p>
<p class="note">This is an existing representative case, selected as the median tail-JS case among calibrated V+S historical score declines. It is a case study, not a population estimate.</p></header>
<section><h2>Correct-value prefix in the layerwise J-lens readout</h2><div class="trace"><img src="assets/figures/kv_correct_value_prefix_trace.svg" alt="Dense reads ca in Top-10 from layer 23; FlexPrefill from layer 27; Calibrated V+S at no layer."></div>
<div class="table-scroll"><table class="case-table"><caption>Final-position prefix readout and layer-31 decomposition</caption><thead><tr><th>Method</th><th>Historical score</th><th>First Top-10 layer</th><th>Layer-31 rank</th><th>Layer-31 k=25 coefficient for ca</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">Missing rank means outside the saved Top-10, rather than absent from the full hidden state. “Not selected” means the greedy decomposition did not select this token atom; it does not establish that its information is absent. FlexPrefill answers this case correctly despite not selecting the ca atom at layer 31.</p></section>
<section><h2>Historical generated answers</h2>{predictions}<p class="note">These predictions come from the earlier task evaluation. No generation was rerun for this trace.</p></section>
<section><h2>Output-limit check across all 50 KV Retrieval prompts</h2>
<div class="table-scroll"><table><caption>Original 50-token generation limit; existing records only</caption><thead><tr><th>Method</th><th>Official correct / 50</th><th>Output-limit stops / 50</th><th>Failed capped outputs with ≥8 characters of correct-value prefix</th></tr></thead><tbody>{generation_rows}</tbody></table></div>
<p class="note">Prefix matches are literal substrings and do not establish complete retrieval or correct key-value binding. Output-limit stops alone do not establish that increasing the limit would recover the answer.</p>
<p><a href="data/kv_generation_history_check.json" download>Download all historical generation checks</a></p></section>
<section><h2>Scope of this observation</h2><p>The trace compares readouts at one fixed prefill position. The full correct value and its binding to the queried key are not measured by the presence of its first token. UUID fragments occur in many other values in the context. Generation-time activations were not saved.</p>
<p>The next diagnostic requires comparing the correct value with competing values across query positions and response-token steps. Actual mask retention, attention routing and causal interventions are separate measurements.</p>
<p><a href="data/kv_retrieval_case_check.json" download>Download this case’s measurements and provenance</a></p></section>
<footer>English · Times New Roman · Saved observations only · No new model execution</footer></main></div></body></html>'''
    (REVIEW / 'kv_retrieval_case.html').write_text(html)
    print('KV PREFIX CASE PAGE READY')


if __name__ == '__main__':
    main()
