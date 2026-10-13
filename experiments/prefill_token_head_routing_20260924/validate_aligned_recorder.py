"""Bounded live CUDA integration of canonical recorder and deferred bridge."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.benchmark_shareprefill_ae3 import GenerationMetricsRecorder
from aligned_recorder import PairStats, make_recorder
from generation_bridge import GenerationBridge
from smoke import require_allocation
from test_routing import small_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    allocation = require_allocation()
    args.out.mkdir(exist_ok=False)
    records = []
    for mode in ('dense', 'q_only', 'q_kv'):
        model = small_model().cuda()
        patch = SimpleNamespace(selector=SimpleNamespace(stats=PairStats()))
        def record(row):
            stats = patch.selector.stats
            stats.selected_token_pairs += sum(x['eligible_pairs'] for x in row['prefill_pairs_by_layer'])
            n = row['valid_prompt_tokens']
            stats.causal_token_pairs += len(model.model.layers)*model.config.num_attention_heads*n*(n+1)//2
        sidecar = args.out/f'{mode}_sidecar.jsonl'
        bridge = GenerationBridge(model, mode, sidecar, retain_input_ids=False,
                                  defer_output=True, on_record=record, event_timing=True)
        inputs = dict(input_ids=torch.tensor([[2,3,4,5,6]],device='cuda'),
                      attention_mask=torch.ones(1,5,device='cuda',dtype=torch.long),
                      do_sample=False, max_new_tokens=3, min_new_tokens=3)
        model.generate(**inputs)  # warmup pending row is discarded on install
        Recorder = make_recorder(GenerationMetricsRecorder, lambda: bridge)
        recorder = Recorder(model, patch, args.out/f'{mode}_metrics.jsonl', method=mode)
        recorder.reset(); recorder.install()
        try:
            for _ in range(2):
                model.generate(**inputs)
            assert len(recorder.rows) == len(bridge.rows) == 2
            assert len(sidecar.read_text().splitlines()) == 2
            assert bridge.pending_row is None
            for metric, diagnostic in zip(recorder.rows,bridge.rows):
                assert metric['selected_token_pairs'] == sum(x['eligible_pairs'] for x in diagnostic['prefill_pairs_by_layer'])
                assert metric['causal_token_pairs'] == 120
                assert metric['prefill_latency_sec'] > 0
                assert diagnostic['decode_calls'] == 2
                assert diagnostic['decode_forward_seconds'] > 0
            records.append({'mode':mode, 'calls':len(recorder.rows), 'status':'passed'})
        finally:
            recorder.uninstall(); bridge.close()
    (args.out/'result.json').write_text(json.dumps({'allocation':allocation,'results':records,
        'backend':'sdpa','real_FA2_validated':False},indent=2)+'\n')
    print(json.dumps(records))

if __name__ == '__main__':
    main()
