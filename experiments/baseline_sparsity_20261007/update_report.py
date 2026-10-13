"""Preserve old report, append audited sparsity, compile before replacing PDFs."""
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPORT=HERE.parents[1]/'reports/mask_flexprefill_results_20261002'
CUSTOM=HERE.parent/'jspace_sparse_analysis_20261006/review/prefill_sparsity_20261007.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.loads((HERE/'outputs/independent-audit.json').read_text());assert audit['status']=='PASS' and audit['samples']==400
custom=json.loads(CUSTOM.read_text());assert custom['status']=='PASS'
for p,digest in custom['source_record_sha256'].items():assert sha(p)==digest
rows=[r for r in custom['rows'] if r['method'] not in ['flexprefill','minference']]+audit['rows']
assert len(rows)==28
(REPORT/'prefill_sparsity_results.json').write_text(json.dumps(dict(status='PASS',rows=rows,
    baseline_audit_path=str(HERE/'outputs/independent-audit.json'),baseline_audit_sha256=sha(HERE/'outputs/independent-audit.json'),
    custom_audit_path=str(CUSTOM),custom_audit_sha256=sha(CUSTOM)),indent=2)+'\n')
backup=REPORT/'previous-20261007-before-sparsity';backup.mkdir(exist_ok=True)
for name in ['generate_report.py','mask_flexprefill_results.tex','mask_flexprefill_results_en.tex','mask_flexprefill_results.pdf','mask_flexprefill_results_en.pdf','report-validation.json']:
    if (REPORT/name).exists() and not (backup/name).exists():shutil.copy2(REPORT/name,backup/name)
p=REPORT/'generate_report.py';original=p.read_text()
marker='# Audited causal-pair sparsity supplement (2026-10-07).'
if marker not in original:
    code='''# Audited causal-pair sparsity supplement (2026-10-07).
sparsity = json.loads((HERE/'prefill_sparsity_results.json').read_text())
assert sparsity['status'] == 'PASS' and len(sparsity['rows']) == 28
sparsity_lookup = {(r['method'],r['task']):r for r in sparsity['rows']}
sparsity_tasks = ['kv_retrieval','longbook_choice_eng','longbook_qa_eng','code_debug']
sparsity_methods = ['dense_full','coverage_token_vertical','coverage_token_vertical_slash',
    'random_token_vertical','random_token_vertical_slash','flexprefill','minference']
sparsity_rows = []
for label,method in zip(labels,sparsity_methods):
    values = [format(sparsity_lookup[(method,task)]['mask_sparsity_percent'],'.2f') for task in sparsity_tasks]
    sparsity_rows.append(label+' & '+' & '.join(values)+r' \\\\')
annex = r\'''\n\\newpage
\\begin{center}
{\\Large\\bfseries Prefill Mask Sparsity}
\\end{center}

Percentage of valid causal query--key token pairs removed. Dense causal
attention is the reference; future positions and padding are excluded.

\\begingroup
\\setlength{\\tabcolsep}{5pt}
\\begin{tabularx}{\\textwidth}{@{}Xrrrr@{}}
\\toprule
\\textbf{Method} & \\textbf{KV Retrieval} & \\textbf{En.MC} & \\textbf{En.QA} & \\textbf{Code.Debug} \\\\
\\midrule
@@SPARSITY_ROWS@@
\\bottomrule
\\end{tabularx}
\\endgroup

\\vspace{6pt}
\\textbf{Definition:} Sparsity = 100\\% $\\times$
(1 $-$ retained valid causal pairs / total valid causal pairs).
For an input of $N$ tokens, the denominator per head is $N(N+1)/2$.

\\textbf{Aggregation:} Equal averages over 32 query heads, 32 layers,
and 50 complete inputs per task. Statistics cover prefill only.

\\textbf{Custom methods:} Counts are taken from the saved token masks of
the original scoring runs, including the union of Token, Vertical, and Slash
where applicable.

\\textbf{FlexPrefill:} Counts use the final selected 128$\\times$128 blocks,
including both Vertical-Slash and Query-Aware additions. Partial edge blocks
are clipped to the input length and the causal region.

\\textbf{MInference:} Counts use the actual converted 64-token key intervals
and individual columns passed to the kernel. The union is counted at token
level after causal clipping, including expansion of the selected slash pattern.

\\textbf{Baseline measurement:} The original 200 full inputs were replayed
once for each baseline using the frozen scoring configuration. Every one of
409,600 head masks was checked. Baseline sparsity describes these replays;
mask indices were not saved in the original scoring runs.

\\textbf{Observer validation:} With and without the observer, attention
outputs matched bitwise at layers 0, 15, and 31 on a 4,097-token test, and
full-model prefill logits matched bitwise on one complete original input
for each baseline.

\\textbf{Parameters:} FlexPrefill: $\\gamma=0.95$, $\\tau=0.1$,
min\\_budget=1024, block\\_size=128. MInference: the frozen
Llama-3.1-8B-Instruct 128k head-pattern file; ratio=1; starting layer=0.
\'''
annex = annex.replace('@@SPARSITY_ROWS@@','\\n'.join(sparsity_rows))
tex = tex.replace(r'\\end{document}',annex+r'\\end{document}')
'''
    needle="(HERE/'mask_flexprefill_results.tex').write_text(tex, encoding='utf-8')"
    assert original.count(needle)==1
    p.write_text(original.replace(needle,code+'\n'+needle))
subprocess.run([sys.executable,str(p)],cwd=REPORT,check=True)
tex=REPORT/'mask_flexprefill_results.tex'
with tempfile.TemporaryDirectory(prefix='sparsity-pdf-',dir=REPORT) as temp:
    stage=Path(temp);shutil.copy2(tex,stage/tex.name);(stage/'fonts').symlink_to(REPORT/'fonts',target_is_directory=True)
    for _ in range(2):
        run=subprocess.run(['/usr/bin/xelatex','-interaction=nonstopmode','-halt-on-error',tex.name],cwd=stage,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (REPORT/'sparsity-xelatex.log').write_text(run.stdout)
        assert run.returncode==0,run.stdout[-2000:]
    pdf=stage/'mask_flexprefill_results.pdf'
    text=subprocess.check_output(['/usr/bin/pdftotext','-layout',str(pdf),'-'],text=True)
    assert 'Prefill Mask Sparsity' in text and 'MInference' in text and '409,600' in text
    for row in rows:assert format(row['mask_sparsity_percent'],'.2f') in text
    fonts=subprocess.check_output(['/usr/bin/pdffonts',str(pdf)],text=True)
    assert 'TimesNewRoman' in fonts
    info=subprocess.check_output(['/usr/bin/pdfinfo',str(pdf)],text=True)
    assert 'Pages:           4' in info,info
    for name in ['mask_flexprefill_results.pdf','mask_flexprefill_results_en.pdf']:
        staged=REPORT/(name+'.tmp');shutil.copy2(pdf,staged);staged.replace(REPORT/name)
    shutil.copy2(tex,REPORT/'mask_flexprefill_results_en.tex')
    (REPORT/'sparsity-pdf-text.txt').write_text(text);(REPORT/'sparsity-pdf-fonts.txt').write_text(fonts)
    subprocess.run(['/usr/bin/pdftoppm','-r','100','-png',str(REPORT/'mask_flexprefill_results.pdf'),str(REPORT/'sparsity-page')],check=True)
result=dict(status='GENERATED',pdf=str(REPORT/'mask_flexprefill_results.pdf'),pdf_sha256=sha(REPORT/'mask_flexprefill_results.pdf'),
    tex=str(tex),tex_sha256=sha(tex),pages=4,font='Times New Roman',table_cells_verified=28,visual_review='pending',baseline_audit_sha256=sha(HERE/'outputs/independent-audit.json'))
(HERE/'report-update.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
