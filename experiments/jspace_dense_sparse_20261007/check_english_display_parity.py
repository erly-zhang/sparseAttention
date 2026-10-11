"""Verify the display-only export preserves inputs, metrics and shared token ranks."""
from pathlib import Path
import json,re
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'figure5_html_en_v2'

def read(directory,idx):
    s=(directory/f'case_{idx}.js').read_text()
    return json.loads(s[s.index(']=')+2:].removesuffix(';\n'))

def main():
    checks=[]
    for idx in range(3):
        old=read(HERE/'figure5_html_en',idx);new=read(OUT,idx)
        for field in ['idx','P','L','N','arms','vocab_size','labels','prompt_tokens','continuation_tokens','prompt','continuation','snippets','metrics','input_ids_sha256','prefix_sha256']:
            assert old[field]==new[field],field
        assert new['display_filter']=='english'
        assert all(s.isascii() and re.search('[A-Za-z0-9]',s) for s in new['vocab'].values())
        def cells(d,name):return np.array(d[name]).reshape(-1,8)
        same=cells(new,'top_ids')[:,:,None]==cells(old,'top_ids')[:,None,:]
        shared=same.any(-1);where=same.argmax(-1)
        for field in ['top_low','top_high','top_probs']:
            prior=np.take_along_axis(cells(old,field),where,axis=1)
            assert np.array_equal(cells(new,field)[shared],prior[shared]),field
        checks.append(dict(idx=idx,all_displayed_tokens_ascii=True,inputs_and_full_vocab_metrics_unchanged=True,shared_token_entries_validated=int(shared.sum())))
    result=dict(status='passed',report_language='en',display_filter='Top8 ASCII letter/number-containing tokens from the full vocabulary',experiment_unchanged=True,samples=checks)
    (OUT/'english_display_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
