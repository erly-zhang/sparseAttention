"""Freeze upstream random_pp with matched local inputs, runner and metrics."""
import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
E=Path(__file__).resolve().parents[1];root=Path(sys.argv[1]).resolve();prior=E/'results/full_kv_w1_p095_recent64_gpu_20260924'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
root.mkdir();(root/'source/scripts').mkdir(parents=True);(root/'logs').mkdir();(root/'formal').mkdir()
for name in ('ra_harness','tests'):shutil.copytree(E/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('prepare_official_random_pp.py','run_official_random_pp.py','collect_official_random_pp.py','official_random_pp_smoke.sh','official_random_pp_worker.sh','official_random_pp_array.sbatch','official_random_pp_collect.sbatch','official_random_pp_preflight.py','submit_official_random_pp.py'):
 shutil.copyfile(E/'scripts'/name,root/'source/scripts'/name)
for name in ('inputs.jsonl','model_assets_sha256.json','dataset_metadata.jsonl'):shutil.copyfile(prior/name,root/name)
shutil.copytree(prior/'reference_baselines',root/'reference_baselines')
cfg=json.loads((prior/'config.json').read_text())
for k in ('recent_window','recent_semantics','random_keep_ratio','gather_backend','inference_change','windows','query_semantics'):cfg.pop(k,None)
cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='official_random_pp',experiment='official_random_pp_matched_MATH500',token_budget=1024,residual_length=64,query_window=None,selection_p=None,mask_seed=42,cache_semantics='persistent_eviction',
 auto_full_submission=True,full_run_authorized=True,scope='Official random_pp default MATH budget; three-input smoke then full500, two rollouts per question',
 source_repository='https://github.com/SalesforceAIResearch/Random-Attention',source_commit='64db9688a12f0926db45fec039370f6ceb1ab4fe',
 mask_rng_adapter='Unmodified upstream sampler executed in private per-sample/layer RNG context; generation global RNG restored; adapter overhead included in decode',
 comparison_caveat='Matched model/inputs/preprocessing/generation/hardware model/timing; official permanent per-KV-head eviction and fixed1024 budget differ from shared full-KV masks and their retention budgets. No algorithm-isolated speed claim at matched sparsity.',smoke_source=str(root/'smoke'))
for name,h in json.loads((root/'model_assets_sha256.json').read_text()).items():assert sha(Path(cfg['model'])/name)==h,name
assert sha(Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl'))==cfg['dataset_sha256']
# Reconstruct ALL500 tokenized inputs with the same chat template/preprocessing.
sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from transformers import AutoTokenizer
from Utils.data_loader import load_data
from Utils.parser import parse_question,parse_ground_truth
import numpy as np
tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
expected={x['idx']:x for x in map(json.loads,(root/'inputs.jsonl').read_text().splitlines())}
examples=load_data('math','test','/project/yl258/jz648/Random-Attention/data')
assert len(examples)==500 and {x['idx'] for x in examples}==set(range(500))
for ex in examples:
 messages=[dict(role='user',content=parse_question(ex,'math')+'\nPlease reason step by step, and put your final answer within \\boxed{}.')]
 ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True)['input_ids']
 old=expected[ex['idx']]
 assert len(ids)==old['input_tokens'] and hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()==old['input_ids_sha256']
 assert tok.decode(ids,skip_special_tokens=True)==old['prompt_text'] and parse_ground_truth(ex,'math')[1]==old['gold']
 assert len(ids)<=1024-64-1,'Upstream prompt-protection cap would differ on this input'
(root/'input_preflight.json').write_text(json.dumps(dict(status='passed/full',questions=500,prompt_min=min(x['input_tokens'] for x in expected.values()),prompt_max=max(x['input_tokens'] for x in expected.values()),inputs_sha256=sha(root/'inputs.jsonl')),indent=2)+'\n')
(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
linkage={str(p.relative_to(root)):sha(p) for name in ('ra_harness','tests') for p in (root/'source'/name).rglob('*') if p.is_file()}
(root/'inference_source_hashes.json').write_text(json.dumps(linkage,indent=2)+'\n')
(root/'README.md').write_text('''# Official Random-Attention matched MATH500 baseline

Upstream commit64db9688a12f0926db45fec039370f6ceb1ab4fe, verified against GitHub main. Official cache_utils.py is vendored byte-for-byte with provenance and license. random_pp: independent per-KV-head uniform scores, protected prompt, TopK of older cached positions plus64recent positions, chronological permanent compaction. MATH defaults1024total budget/residual64. Attention runs BEFORE eviction; first eviction after attention at cache length1088, then every64appends, compacting back to1024. Thus actual decode attention can use1025..1088keys after saturation; sparsity is not simply1-1024/history. Each KV head's Q-head group uses its own retained history. RoPE uses original cumulative positions. None of our shared-mask algorithms is changed into eviction.

Adapter subclasses upstream and calls its eviction code verbatim. The RNG context uses a private per-input/layer stream(seed42 with stable identity hash), restoring global generation RNG. This is an explicit evaluation-protocol adaptation, not exact reproduction of the upstream launcher's coupled global RNG sequence. Its overhead is included. Counters record actual pre-eviction pairs and complete causal-history denominators, with no per-step diagnostic logging. Independent integer recurrence validates final cache lengths, eviction counts and total selected pairs.

All non-algorithm controls match the existing runner: Qwen3-4B identical weights/tokenizer, BF16, same500MATH inputs and reconstructed input hashes, same chat template/preprocessing, batch2(two rollouts), sample seed42, temperature0.6, generation Top-p0.95, no top_k, max total length32768, same EOS/active-row filtering and2-token untimed warmup. GPU-synchronized prefill/decode timing includes selection, compaction, model and sampling; loading/warmup/serialization excluded. OneA10080GB per worker; no upstream multiworker sharing of a GPU. Historical dense timing still has its documented limitations.

Correctness checks compare exact cache contents to upstream under matched RNG, verify per-head diversity and prompt/recent protection, global generation RNG restoration, independent GQA attention, pre-eviction pair counts, batch filtering and absolute sequence lengths. Then same3inputs0,1,2 generate completely. On technical success the authorized pipeline submits all500questions in8shards(max4concurrentGPUs),48h limit, and a dependent CPU collector. Low smoke accuracy is an effect measurement, not suppressed. Formal source is the same frozen source that passed smoke.

Track pipeline_status.json, correctness.log, smoke/summary.json, submission.json and collection_status.json. formal/REPORT.md, summary.json, question_metrics.jsonl/csv and rollout_metrics.csv report accuracy, per-run/per-level and paired outcomes, EOS/length limits, token lengths, synchronized time distributions, aggregate throughput, actual selected/causal pairs, weighted sparsity, eviction events, peak memory and hardware. Completion requires500unique inputs,1000outputs and input_alignment=passed/full. Comparison is at each algorithm's stated budget, not matched retained-pair density.
''')
manifest={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(root,flush=True)
