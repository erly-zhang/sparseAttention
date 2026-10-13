"""Portable folder and ZIP of the verified HTML report and all local assets."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
REVIEW = HERE / 'review'
DEST = REVIEW / 'html_report'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = REVIEW / 'results_with_figures.html'
    validation = json.loads((REVIEW / 'html-validation.json').read_text())
    assert validation['status'] == 'PASS' and validation['html_sha256'] == sha(source)
    html = source.read_text()
    css_match, = list(re.finditer(r'<style>(.*?)</style>', html, re.S))
    js_match, = list(re.finditer(r'<script>\n(.*?)\n</script>', html, re.S))
    css, js = css_match.group(1), js_match.group(1)
    provenance_match = re.search(r'<script type="application/json" id="report-provenance">(.*?)</script>', html, re.S)
    provenance = json.loads(provenance_match.group(1))
    for subdir in ['assets/figures', 'assets/fonts', 'data', 'reports']:
        (DEST / subdir).mkdir(parents=True, exist_ok=True)
    font_source = HERE.parents[1] / 'reports/mask_flexprefill_results_20261002/fonts'
    font_uris = re.findall(r"data:font/ttf;base64,[^']+", css)
    assert len(font_uris) == len(provenance['fonts']) == 3
    for uri, (filename, expected) in zip(font_uris, provenance['fonts'].items()):
        source_font = font_source / filename
        assert sha(source_font) == expected
        shutil.copyfile(source_font, DEST / 'assets/fonts' / filename)
        css = css.replace(uri, 'fonts/' + filename)
    css += '\n.toolbar a{text-decoration:none}\n'
    image_uris = re.findall(r'data:image/png;base64,[^\"]+', html)
    assert len(image_uris) == len(provenance['figures']) == 14
    for uri, (stem, expected) in zip(image_uris, provenance['figures'].items()):
        filename = stem + '.png'
        source_figure = HERE / 'figures' / filename
        assert sha(source_figure) == expected
        shutil.copyfile(source_figure, DEST / 'assets/figures' / filename)
        html = html.replace(uri, 'assets/figures/' + filename)
    # Copy literal styles/scripts unchanged, except the font asset paths above.
    html = re.sub(r'<style>.*?</style>', '<link rel="stylesheet" href="assets/report.css">', html, count=1, flags=re.S)
    html = re.sub(r'<script>\n.*?\n</script>', '<script src="assets/report.js" defer></script>', html, count=1, flags=re.S)
    html = html.replace('<div class="toolbar">', '<div class="toolbar"><a class="button" href="reports/results_with_figures.pdf" download>PDF version</a>', 1)
    html = html.replace('Original figure PNGs and Times New Roman regular, bold and italic font files are embedded.',
                        'Original figure PNGs and Times New Roman regular, bold and italic font files are included in this folder.')
    (DEST / 'assets/report.css').write_text(css)
    (DEST / 'assets/report.js').write_text(js)
    (DEST / 'index.html').write_text(html)
    for name in ['reviewed-task-summary.json', 'html-provenance.json', 'html-validation.json', 'report-provenance.json']:
        shutil.copyfile(REVIEW / name, DEST / 'data' / name)
    shutil.copyfile(HERE / 'outputs/readout-summary.json', DEST / 'data/readout-summary.json')
    for name in ['results_summary.pdf', 'results_with_figures.pdf']:
        shutil.copyfile(REVIEW / name, DEST / 'reports' / name)
    if (REVIEW / 'kv_retrieval_case.html').exists():
        shutil.copyfile(REVIEW / 'kv_retrieval_case.html', DEST / 'kv_retrieval_case.html')
        shutil.copyfile(REVIEW / 'kv_retrieval_case_check.json', DEST / 'data/kv_retrieval_case_check.json')
        shutil.copyfile(REVIEW / 'kv_generation_history_check.json', DEST / 'data/kv_generation_history_check.json')
        shutil.copyfile(REVIEW / 'kv_correct_value_prefix_trace.svg',
                        DEST / 'assets/figures/kv_correct_value_prefix_trace.svg')
        html = html.replace('<div class="toolbar">',
                            '<div class="toolbar"><a class="button" href="kv_retrieval_case.html">KV case trace</a>', 1)
        (DEST / 'index.html').write_text(html)
    if (REVIEW / 'figure5_prefill.html').exists():
        shutil.copyfile(REVIEW / 'figure5_prefill.html', DEST / 'figure5_prefill.html')
        for name in ['figure5_prefill_data.json', 'figure5-prefill-provenance.json']:
            shutil.copyfile(REVIEW / name, DEST / 'data' / name)
        for ext in ['png', 'svg']:
            name = 'kv_full_value_token_ranks.' + ext
            shutil.copyfile(REVIEW / name, DEST / 'assets/figures' / name)
        html = html.replace('<div class="toolbar">',
                            '<div class="toolbar"><a class="button" href="figure5_prefill.html">Figure 5-style explorer</a>', 1)
        (DEST / 'index.html').write_text(html)
    (DEST / 'README.txt').write_text('''DENSE-SPARSE PREFILL: REVIEWED RESULTS

Open index.html in a browser. No server, installation, or internet access is
required. Keep this entire folder together; asset paths are relative.

Contents
  index.html          English report, all numerical tables, and 14 figures
  assets/report.css   Page and print styles
  assets/report.js    Table selector, figure enlargement, print and download
  assets/figures/     All 14 original scientific figures
  assets/fonts/       Times New Roman regular, bold, and italic
  data/               Reviewed measurements and source provenance
  reports/            Two-page summary and full 16-page PDF
  kv_retrieval_case.html  Supplementary correct-value prefix trace (one case)
  figure5_prefill.html  Layer-position explorer; full V / first 50% token probes
  MANIFEST.json       Folder file hashes and packaging provenance

Use the atom-cap buttons to select k=16 or k=25 for the support-recall table.
Click a figure to enlarge it. Press Escape or Close to dismiss it.
Print / Save PDF includes both recall tables and all figure pages.

These observations cover full-context prefill, sampled positions, all 32
layers and 50 prompts per InfiniteBench task. They do not contain generated
response trajectories or a causal explanation of task failures.
The supplementary case page uses saved readouts and historical predictions;
no new model execution was performed. It tracks the first value token only.
The Figure 5-style explorer additionally tracks all 25 correct-value tokens
(18 unique vocabulary IDs), or the first 13 tokens as a rounded-up 50% prefix.
These ranks were recomputed from saved residuals with the existing J-lens,
without rerunning the LLM. Repeated token IDs repeat the same probe. Token
readouts do not establish sequence probability or key-to-value binding.

Data tables, image/font hashes, JavaScript syntax, and relative resource paths
were verified. Browser visual preview was not performed because this runtime
does not have a browser. The original experiment outputs were not modified.
''')
    # Verify all image/style/script paths against the completed folder.
    packaged_html = html
    if (DEST / 'kv_retrieval_case.html').exists():
        packaged_html += (DEST / 'kv_retrieval_case.html').read_text()
    if (DEST / 'figure5_prefill.html').exists():
        packaged_html += (DEST / 'figure5_prefill.html').read_text()
    paths = re.findall(r'(?:src|href)="([^"]+)"', packaged_html)
    for path in paths:
        if path.startswith(('#', 'https://', 'http://')):
            continue
        assert (DEST / path).is_file(), path
    for path in re.findall(r"src:url\('([^']+)'\)", css):
        assert (DEST / 'assets' / path).is_file(), path
    assert 'data:image/' not in html and 'data:font/' not in css
    manifest = dict(source_html_sha256=sha(source), language='English', font='Times New Roman',
                    entry_point='index.html', offline=True, external_asset_requests=False,
                    relative_resource_paths='PASS', figures=14,
                    files={str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob('*'))
                           if p.is_file() and p.name != 'MANIFEST.json'})
    (DEST / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    archive = REVIEW / 'html_report.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                z.write(path, Path(DEST.name) / path.relative_to(DEST))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.namelist() and all(p.startswith('html_report/') for p in z.namelist())
    print('FOLDER READY', DEST)
    print('ZIP VERIFIED', archive, archive.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
