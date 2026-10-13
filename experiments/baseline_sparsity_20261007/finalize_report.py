"""Finish document typography and metadata without changing experiment sources."""
import hashlib,json,shutil,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPORT=HERE.parents[1]/'reports/mask_flexprefill_results_20261002'
replacements={r'$\times$':'×',r'$-$':'−',r'$N$':'N',r'$N(N+1)/2$':'N(N+1)/2',r'$\gamma=0.95$':'γ=0.95',r'$\tau=0.1$':'τ=0.1'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['generate_report.py','mask_flexprefill_results.tex','mask_flexprefill_results_en.tex']:
 p=REPORT/name;s=p.read_text()
 for a,b in replacements.items():s=s.replace(a,b)
 p.write_text(s)
with tempfile.TemporaryDirectory(prefix='sparsity-font-',dir=REPORT) as temp:
 stage=Path(temp);(stage/'fonts').symlink_to(REPORT/'fonts',target_is_directory=True)
 shutil.copy2(REPORT/'mask_flexprefill_results.tex',stage/'mask_flexprefill_results.tex')
 for _ in range(2):
  p=subprocess.run(['/usr/bin/xelatex','-interaction=nonstopmode','-halt-on-error','mask_flexprefill_results.tex'],cwd=stage,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
  assert p.returncode==0,p.stdout[-2000:]
  (REPORT/'sparsity-xelatex.log').write_text(p.stdout)
 pdf=stage/'mask_flexprefill_results.pdf'
 info=subprocess.check_output(['/usr/bin/pdfinfo',str(pdf)],universal_newlines=True)
 assert next(line.split(':',1)[1].strip() for line in info.splitlines() if line.startswith('Pages:'))=='4'
 fonts=subprocess.check_output(['/usr/bin/pdffonts',str(pdf)],universal_newlines=True)
 assert all('TimesNewRoman' in line for line in fonts.splitlines()[2:] if line.strip()),fonts
 text=subprocess.check_output(['/usr/bin/pdftotext','-layout',str(pdf),'-'],universal_newlines=True)
 rows=json.loads((REPORT/'prefill_sparsity_results.json').read_text())['rows']
 for row in rows:assert format(row['mask_sparsity_percent'],'.2f') in text
 assert '409,600' in text and 'Prefill Mask Sparsity' in text
 for name in ['mask_flexprefill_results.pdf','mask_flexprefill_results_en.pdf']:
  tmp=REPORT/(name+'.tmp');shutil.copy2(pdf,tmp);tmp.replace(REPORT/name)
 (REPORT/'sparsity-pdf-text.txt').write_text(text);(REPORT/'sparsity-pdf-fonts.txt').write_text(fonts)
 subprocess.run(['/usr/bin/pdftoppm','-r','100','-png',str(REPORT/'mask_flexprefill_results.pdf'),str(REPORT/'sparsity-page')],check=True)
p=HERE/'report-update.json';r=json.loads(p.read_text());r.update(status='GENERATED',pdf_sha256=sha(REPORT/'mask_flexprefill_results.pdf'),tex_sha256=sha(REPORT/'mask_flexprefill_results.tex'),font='Times New Roman for all embedded text and symbols',visual_review='pending_final_page_check',report_export_failure_repaired=True)
p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
