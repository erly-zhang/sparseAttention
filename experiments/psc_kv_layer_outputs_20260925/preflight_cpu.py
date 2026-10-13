"""No GPU compute: validate fixed data/IDs, original imports and source identity."""
import ast,hashlib,importlib.metadata as md,json,struct,sys,subprocess
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer,AutoConfig
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path[:0]=[str(ROOT),str(HERE)]
from adapters import METHODS,FLEX
MODEL=Path('/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/models/Llama-3.1-8B-Instruct')

def main():
    import flash_attn,minference
    import experiments.run_shareprefill_autoencoder_clustering as ae
    from flex_prefill import patch_model
    from selector6040 import Selector
    manifest=json.loads((HERE/'inputs/manifest.json').read_text());assert manifest['test_indices']==[3]
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    selected=[json.loads(x) for x in (HERE/'inputs/selected_records.jsonl').read_text().splitlines()]
    for item,record in zip(manifest['records'],selected):
        i=item['source_index'];row=record['record']
        prompt='Extract the value corresponding to the specified key in the JSON object below.\n\n'+row['context']+'\n\n'+row['input']
        rendered=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        ids=tok.encode(rendered,add_special_tokens=False);limit=130944
        if len(ids)>limit:ids=ids[:limit//2]+ids[-(limit-limit//2):]
        raw=np.asarray(ids,dtype='<i8').tobytes()
        assert raw==(HERE/'inputs'/f'{i}.ids.i64le').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==item['input_ids_sha256']
        if i<3:
            prefix=tok(prompt,truncation=True,max_length=1936)['input_ids']
            assert np.asarray(prefix,dtype='<i8').tobytes()==(HERE/'inputs'/f'{i}.ae_ids.i64le').read_bytes()
    cfg=AutoConfig.from_pretrained(MODEL,local_files_only=True)
    assert (cfg.num_hidden_layers,cfg.num_attention_heads,cfg.num_key_value_heads,cfg.hidden_size)==(32,32,8,4096)
    files=list(HERE.glob('*.py'))+[ROOT/'experiments/run_shareprefill_autoencoder_clustering.py',ROOT/'experiments/token_compacted_sparse.py',ROOT/'experiments/benchmark_shareprefill_ae3.py',ROOT/'experiments/baseline_sparsity.py']+list((ROOT/'experiments/residual6040_20260908').glob('*.py'))
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    record={'status':'passed_cpu_only','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'versions':{name:md.version(name) for name in ['torch','transformers','tokenizers','flash-attn','minference','triton','scipy','numpy']},
        'model_config':cfg.to_dict(),'gqa_mapping':[h//4 for h in range(32)],'methods':METHODS,
        'fixed_ids_match_compatibility_tokenizer':True,'ae_prefix_ids_match_original_pipeline':True,
        'dataset_revision':manifest['dataset_revision'],'data_sha256':manifest['data_sha256'],
        'grouping_origin':'PSC regenerated; not Wulver historical', 'grouping_independent_of_validation':False,
        'test_index_used_for_training':False,'gpu_compute_performed':False}
    (HERE/'cpu_preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record['versions']))
if __name__=='__main__':main()
