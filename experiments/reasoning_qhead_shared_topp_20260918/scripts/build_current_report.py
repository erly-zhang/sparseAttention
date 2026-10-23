"""Build the English PDF/HTML report from the validated current result catalog.

CPU only. Raw experiments are never modified. ReportLab4.4.3 is used for PDF;
the local report-runtime and the original Times New Roman fonts are reusable.
Use --snapshot report_data.json to rebuild an already frozen report.
"""
import argparse
import base64
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

E = Path(__file__).resolve().parents[1]
WORKSPACE = E.parents[2]
sys.path.insert(0,str(E/"scripts"))
from current_report_layout import primary, name as display_name, current_sections, history_sections, validation_sections


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def budget(r):
    return int(r['budget'].split('=')[1]) if r['budget'].startswith('total K=') else 0


def label(r):
    return display_name(r)


def f(v, digits=1):
    return '—' if v is None else format(v, ',.'+str(digits)+'f')


def freeze():
    catalog = read(E/'results/current/summary.json')
    history = read(E/'results/current/implementation_history.json')
    rows = sorted(catalog['results'], key=lambda r: (
        r['dataset'], budget(r), r['protection'], r['storage'],
        r['compaction_interval'], r['method'], r['score']))
    counts = {}
    for r in rows:
        d = r['dataset']; counts[d] = counts.get(d, 0)+1
        r.pop('row_id',None)
        assert sha(E/'results'/r['source']) == r['source_sha256']
    audit = WORKSPACE/'.codex-persist/hmmt_dense_paper_audit_20261014'
    grading = read(audit/'canonical_grading.json')
    # The official random_pp parent summary contains two different K cells.
    # Its path alone is not a unique result identifier.
    compared = {(g['source'], 'total K='+str(g['budget']) if isinstance(g['budget'],int)
                 else g['budget']):g for g in grading['groups']}
    reviewed = [r for r in rows if r['dataset']=='hmmt' and (r['source'],r['budget']) in compared]
    assert all(compared[(r['source'],r['budget'])]['canonical_correct']==r['correct'] for r in reviewed)
    assert grading['status']=='passed' and not grading['changes']
    v3 = WORKSPACE/'.codex-persist/w1_qk_fused_20261014_validationfix'
    assert read(v3/'correctness.json')['status']=='passed'
    return dict(as_of='2026-10-14', language='English', results=rows, history=history,
        counts=counts, supersession_proofs=catalog['supersession_proofs'],
        validation=read(E/'results/current/validation.json'),
        comparability=read(audit/'internal_comparability.json'),
        grading=dict(complete_groups=grading['complete_result_groups'],
                     unique_pred_gold_pairs=grading['unique_prediction_gold_pairs'],
                     changed_grades=0, current_hmmt_groups_covered=len(reviewed),
                     policy=grading['grading_policy']),
        uncertainty=read(audit/'best_vs_dense_uncertainty.json'),
        v3_microbenchmark=read(v3/'microbenchmark.json'),
        v3_correctness=read(v3/'correctness.json'),
        source_hashes={str(p.relative_to(WORKSPACE)):sha(p) for p in (
            E/'results/current/summary.json', E/'results/current/validation.json',
            E/'results/current/implementation_history.json', audit/'canonical_grading.json',
            audit/'internal_comparability.json', audit/'best_vs_dense_uncertainty.json',
            v3/'microbenchmark.json', v3/'correctness.json')})


def sections(data):
    return current_sections(data)


def pdf_report(path, content, fonts, as_of='14 October 2026'):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.pdfgen.canvas import Canvas
    from functools import partial
    for name,file in [('TNR','Times.TTF'),('TNR-Bold','Timesbd.TTF'),('TNR-Italic','Timesi.TTF')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNR-Bold',italic='TNR-Italic',boldItalic='TNR-Bold')
    body = ParagraphStyle('Body',fontName='TNR',fontSize=10,leading=13,spaceAfter=8,textColor=colors.HexColor('#26313b'))
    title = ParagraphStyle('Title',parent=body,fontName='TNR-Bold',fontSize=19,leading=23,spaceAfter=13)
    small = ParagraphStyle('Table',parent=body,fontSize=8.5,leading=10.5,spaceAfter=0)
    header = ParagraphStyle('Header',parent=small,fontName='TNR-Bold',textColor=colors.white)
    doc = SimpleDocTemplate(str(path),pagesize=A4,leftMargin=48,rightMargin=48,
        topMargin=46,bottomMargin=43,title=content[0]['title'],author='SparseAttention experiments')
    story=[]
    for i,s in enumerate(content):
        if i: story.append(PageBreak())
        story.append(Paragraph(s['title'],title))
        for text in s['paragraphs']: story.append(Paragraph(text,body))
        for t in s['tables']:
            if t.get('title'):
                story.append(Paragraph('<b>'+html.escape(t['title'])+'</b>',body))
            matrix=[[Paragraph(html.escape(str(v)),header) for v in t['headers']]]
            matrix += [[Paragraph(html.escape(str(v)),small) for v in row] for row in t['rows']]
            widths=[w*doc.width/sum(t['widths']) for w in t['widths']]
            table=Table(matrix,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#244c59')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f4f5')]),
                ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#244c59')),
                ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#b7c4c8'))]))
            story.extend([Spacer(1,7),table,Spacer(1,10)])
    def page(canvas,doc):
        canvas.saveState();canvas.setFont('TNR',8);canvas.setFillColor(colors.HexColor('#68777e'))
        canvas.drawString(48,24,'SparseAttention · Local results · '+as_of)
        canvas.drawRightString(A4[0]-48,24,str(doc.page));canvas.restoreState()
    doc.build(story,onFirstPage=page,onLaterPages=page,canvasmaker=partial(Canvas,initialFontName='TNR'))


def html_report(path, content, data, fonts):
    font = base64.b64encode((fonts/'Times.TTF').read_bytes()).decode()
    css = """@font-face{font-family:TNR;src:url(data:font/ttf;base64,FONT)}
    *{box-sizing:border-box}body{margin:0;background:#f2f4f4;color:#26313b;font:17px/1.55 TNR,'Times New Roman',serif}
    main{max-width:1140px;margin:auto;padding:30px}section{background:white;padding:30px;margin:20px 0;border:1px solid #d5dfe2}
    h1{font-size:30px}h2{font-size:25px;color:#244c59}nav{display:flex;flex-wrap:wrap;gap:14px}
    a{color:#166475}table{width:100%;border-collapse:collapse;font-size:15px;margin:18px 0}
    th{text-align:left;background:#244c59;color:white;padding:9px}td{padding:8px;vertical-align:top;border-bottom:1px solid #d5dfe2}
    tr:nth-child(even){background:#f0f4f5}.table-wrap{overflow:auto}small{font-size:14px}details{margin:15px 0}code{font-size:13px;overflow-wrap:anywhere}
    @media(max-width:650px){main{padding:10px}section{padding:16px}table{min-width:790px}}
    @media print{body{background:white}main{max-width:none;padding:0}section{border:none;break-before:page}nav{display:none}}
    """.replace('FONT',font)
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>Current shared KV selection results</title><style>'+css+'</style><main>',
           '<h1>Current shared KV selection results</h1><p>English report · 14 October 2026</p>',
           '<nav><a href="hmmt_results_en.pdf">Current PDF</a><a href="historical_results_en.pdf">Historical PDF</a><a href="w1_v3_validation_en.pdf">v3 validation only</a><a href="results.csv">Current comparison CSV</a><a href="report_data.json">Source-backed data</a></nav>',
           '<nav>'+''.join('<a href="#s'+str(i)+'">'+html.escape(s['title'])+'</a>' for i,s in enumerate(content))+'</nav>']
    for i,s in enumerate(content):
        parts.append('<section id="s'+str(i)+'"><h2>'+html.escape(s['title'])+'</h2>')
        parts += ['<p>'+p+'</p>' for p in s['paragraphs']]
        for t in s['tables']:
            if t.get('title'): parts.append('<h3>'+html.escape(t['title'])+'</h3>')
            parts.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in t['headers'])+'</tr></thead><tbody>')
            parts += ['<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in t['rows']]
            parts.append('</tbody></table></div>')
        parts.append('</section>')
    parts.append('<section><h2>Sources for the current comparison</h2><p>Method names and configurations map to exact local summaries below. Full grades, metric-file hashes, input fingerprints and raw pair totals are retained in the accompanying JSON.</p>')
    for r in [r for r in data['results'] if primary(r)]:
        relative=os.path.relpath(E/'results'/r['source'],path.parent)
        parts.append('<details><summary>'+html.escape(label(r)+' · '+r['budget']+' · '+r['storage']+' · update '+str(r['compaction_interval']))+'</summary><p><a href="'+html.escape(relative,quote=True)+'">Open local source summary</a></p><code>'+html.escape(r['source'])+'</code><p><small>SHA-256: '+r['source_sha256']+'</small></p></details>')
    parts.append('</section></main></html>')
    path.write_text('\n'.join(parts))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=WORKSPACE/'report_build/hmmt_results_20261014')
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--html-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    data=read(args.snapshot) if args.snapshot else freeze()
    assert not re.search('[\u3400-\u9fff]',json.dumps(data['as_of']))
    for r in data['results']+data['history']: r.pop('row_id',None)
    selected=[r for r in data['results'] if primary(r)]
    historical=[r for r in data['results'] if not primary(r)]
    data['presentation']={'current_comparison_groups':len(selected),'historical_reference_groups':len(historical),'show_lookup_ids':False,'v3_full_hmmt_result':False}
    content=sections(data)
    fonts=WORKSPACE/'report_build/hmmt_compact_results_20261006/fonts'
    (out/'report_data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    columns=['dataset','method','score','budget','storage','protection','compaction_interval',
             'questions','answers','correct','accuracy','prefill_seconds','decode_seconds',
             'decode_forward_calls_mean','decode_ms_per_forward','output_tokens_mean',
             'output_tokens_median','cap_hits','peak_memory_gib','selected_pairs','causal_pairs',
             'global_pair_sparsity','input_alignment','source','source_sha256','notes']
    for filename,records in [('results.csv',selected),('historical_results.csv',historical),('all_results.csv',data['results'])]:
        with (out/filename).open('w',newline='') as target:
            writer=csv.DictWriter(target,fieldnames=columns,extrasaction='ignore')
            writer.writeheader();writer.writerows(records)
    html_report(out/'index.html',content,data,fonts)
    if not args.html_only:
        sys.path.insert(0,str(WORKSPACE/'.tools/report-runtime'))
        pdf_report(out/'hmmt_results_en.pdf',content,fonts)
        pdf_report(out/'historical_results_en.pdf',history_sections(data),fonts)
        pdf_report(out/'w1_v3_validation_en.pdf',validation_sections(data),fonts)
    (out/'README.md').write_text('# Current HMMT report\n\nOpen `hmmt_results_en.pdf` (3 pages) or `index.html` for the8 latest protected raw-QK W1 / Dense / official random_pp configurations. No lookup IDs appear in the report.\n\n`historical_results_en.pdf` contains earlier probability-scored W1, Shared Random,500-step and K4096 controls, unprotected runs, older budgets and other datasets. `w1_v3_validation_en.pdf` contains only GPU correctness/single-layer timings; no formal v3 HMMT result is claimed.\n\n`results.csv` contains the8 main configurations, `historical_results.csv` the52 separate references, and `all_results.csv` the full60-result archive. `report_data.json` retains sources and implementation history.\n\nRebuild from the experiment directory with `python scripts/build_current_report.py --snapshot '+str(out/'report_data.json')+' --output '+str(out)+'`. Times New Roman is embedded; HTML is standalone. Raw evidence is preserved.\n')

    print(json.dumps(dict(output=str(out),counts=data['counts'],current_comparison=len(selected),historical_references=len(historical),sections=len(content),language='English',status='built')))


if __name__=='__main__':
    main()
