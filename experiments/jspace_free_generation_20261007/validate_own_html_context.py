"""Verify per-method HTML text, context and phase labels against raw traces."""
import json
from pathlib import Path
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
OUT=HERE/'figure5_html_en'


def main():
    tok=AutoTokenizer.from_pretrained('/scratch/yl258/jz648/Qwen3-4B',local_files_only=True)
    checked=0;traces=0
    for idx in range(3):
        text=(OUT/f'case_{idx}.js').read_text()
        d=json.loads(text[text.index(']=')+2:].removesuffix(';\n'))
        directory=HERE/'captures'/f'sample_{idx}'
        validation=json.loads((directory/'validation.json').read_text())
        prefixes={r['arm']:r['common_prefix_tokens_vs_dense'] for r in validation['arms']}
        original={}
        for line in (directory/'readouts.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['layer']==0:original[r['arm'],r['step']]=r
        for a,arm in enumerate(d['arms']):
            t=json.loads((directory/f'{arm}.json').read_text())
            assert d['continuations'][a]==t['continuation_text']
            assert d['prefix_sha256_by_arm'][a]==t['prefix_sha256']
            assert (OUT/f'raw/sample_{idx}/{arm}.txt').read_text()==t['continuation_text']
            assert d['trajectory_stats'][a]['think_close_token']==t['think_close_token']
            context=t['prompt_ids']+t['continuation_ids']
            for p,label in enumerate(d['labels_by_arm'][a]):
                s=label['step'];absolute=t['input_tokens']+s-1
                assert label['absolute_position']==absolute
                assert label['same_token_prefix_as_dense']==(s<=prefixes[arm])
                assert label['reasoning']==original[arm,s]['reasoning']
                expected=[tok.decode(context[max(0,absolute-25):absolute]),
                          tok.decode([context[absolute]]),
                          tok.decode(context[absolute+1:absolute+26])]
                assert d['snippets_by_arm'][a][p]==expected
                checked+=1
            traces+=1
    result=dict(status='passed',original_texts_exact=traces,own_context_snippets_exact=checked,
                own_reasoning_and_prefix_labels='passed',current_token_positions='passed',
                raw_download_texts='passed')
    (OUT/'own_context_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
