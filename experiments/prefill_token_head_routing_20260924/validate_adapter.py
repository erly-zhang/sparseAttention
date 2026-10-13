"""Synthetic fixture only: exercise installed official task loader + scorer.
No pretrained model, benchmark dataset, or accuracy result is represented.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
import lm_eval
from test_routing import small_model
from generation_bridge import GenerationBridge
from smoke import require_allocation

ROOT = Path(__file__).resolve().parent
FLEX = Path('/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/FlexPrefill')
OUT = ROOT/'records/adapter_fixture'


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve()
    if (ROOT/'records').resolve() not in OUT.parents:
        parser.error('fixture output must be a new directory below experiment records/')
    require_allocation()
    OUT.mkdir(exist_ok=False)
    tasks = OUT/'tasks'; tasks.mkdir()
    official = FLEX/'experiments/benchmark/infinitebench'
    for name in ('metrics.py','rouge.py','utils.py'):
        shutil.copy2(official/name,tasks/name)
    fixture = {'input':'What is the pass key?', 'context':'The pass key is 12345.', 'answer':['12345']}
    (OUT/'passkey.jsonl').write_text(json.dumps(fixture)+'\n')
    yaml = (official/'passkey.yaml').read_text().replace('experiments/benchmark/infinitebench/data/InfiniteBench.py',str(official/'data/InfiniteBench.py')).replace('data_dir: experiments/benchmark/infinitebench/data',f'data_dir: {OUT}')
    (tasks/'passkey.yaml').write_text(yaml)
    sys.path.insert(0,str(FLEX/'experiments/benchmark'))
    from utils import fixed_generate_until, tok_encode_middle_trunc, tok_batch_encode_middle_trunc
    HFLM.generate_until = fixed_generate_until
    HFLM.tok_encode = tok_encode_middle_trunc
    HFLM.tok_batch_encode = tok_batch_encode_middle_trunc
    tokenizer = Tokenizer(WordLevel({'<pad>':0,'<unk>':1,'<eos>':42,'12345':2,'The':3,'pass':4,'key':5,'is':6,'.':7},unk_token='<unk>'))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='<pad>',unk_token='<unk>',eos_token='<eos>')
    tok.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    rows = []
    for mode in ('dense','q_only','q_kv'):
        model = small_model().cuda()
        wrapped = HFLM(pretrained=model,tokenizer=tok,batch_size=1,max_length=128,backend='causal',device='cuda')
        bridge = GenerationBridge(model,mode,OUT/f'{mode}_generations.jsonl',tok)
        result = lm_eval.simple_evaluate(model=wrapped,tasks=['passkey'],task_manager=TaskManager(include_path=str(tasks),include_defaults=False),
            apply_chat_template=True,gen_kwargs='do_sample=False,max_gen_toks=3,max_new_tokens=3',log_samples=True,
            limit=1,random_seed=42,numpy_random_seed=42,torch_random_seed=42,fewshot_random_seed=42,bootstrap_iters=0)
        assert len(bridge.rows)==1
        assert len(result['samples']['passkey'])==1
        rows.append({'condition':mode,'fixture_only':True,'official_metric_keys':list(result['results']['passkey']),
                     'doc_id':result['samples']['passkey'][0]['doc_id'], 'generation':bridge.rows[0]})
        bridge.close()
    assert len({r['generation']['input_ids_sha256'] for r in rows}) == 1
    (OUT/'validation.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('SYNTHETIC_OFFICIAL_ADAPTER_A_B_C_PASSED; NOT A BENCHMARK RESULT')

if __name__ == '__main__':
    main()
