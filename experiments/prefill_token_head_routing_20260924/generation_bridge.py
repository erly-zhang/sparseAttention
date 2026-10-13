"""Keep HF/InfiniteBench generation & stopping intact; route only full prefill."""
from functools import wraps
import hashlib
import json
from pathlib import Path
import time
import torch
from routing import RoutingController


class GenerationBridge:
    def __init__(self, model, mode, output=None, tokenizer=None, retain_input_ids=True, defer_output=False, on_record=None, event_timing=False):
        self.model, self.mode, self.tokenizer = model, mode, tokenizer
        self.output = Path(output) if output else None
        self.retain_input_ids = retain_input_ids
        self.defer_output, self.on_record = defer_output, on_record
        self.event_timing = event_timing
        self.pending_row = None
        self.original_generate = model.generate
        self.rows = []
        model.generate = self.generate

    def flush_pending(self):
        if self.output and self.pending_row is not None:
            with self.output.open('a') as stream:
                stream.write(json.dumps(self.pending_row)+'\n')
        self.pending_row = None

    def close(self):
        self.model.generate = self.original_generate

    def generate(self, *args, **kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is None or ids.shape[0] != 1:
            raise ValueError('InfiniteBench bridge requires input_ids with batch=1')
        if self.event_timing and ids.shape[1] <= 1:
            raise ValueError('canonical timing requires prompt length > 1; use development tests for single-token prompts')
        if kwargs.get('past_key_values') is not None:
            raise ValueError('external/nonempty generation cache unsupported')
        config = kwargs.get('generation_config', self.model.generation_config)
        if (kwargs.get('num_beams', config.num_beams) or 1) != 1 or kwargs.get('do_sample', config.do_sample):
            raise ValueError('audited InfiniteBench path requires greedy num_beams=1')
        if self.pending_row is not None:
            raise RuntimeError('unflushed generation record')
        events = []
        ctl = RoutingController(self.model, self.mode)
        ctl.guard.remove()  # validate inside the explicit forward phase below
        original_forward = self.model.forward
        prefill_calls = 0
        prefill_seconds = decode_seconds = 0.0
        decode_calls = 0
        cache_shapes = None
        device = ids.device
        def sync():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        @wraps(original_forward)
        def forward(*fargs, **fkwargs):
            nonlocal prefill_calls, prefill_seconds, decode_seconds, decode_calls, cache_shapes
            token_ids = fkwargs.get('input_ids', fargs[0] if fargs else None)
            cache = fkwargs.get('past_key_values')
            size = cache.get_seq_length() if cache is not None else 0
            phase = 'prefill' if prefill_calls == 0 else 'decode'
            if phase == 'prefill' and (size != 0 or token_ids.shape[1] != ids.shape[1]):
                raise ValueError('full prompt in a single prefill call required')
            if phase == 'decode' and (size == 0 or token_ids.shape[1] != 1):
                raise ValueError('single-token cached dense decode required')
            mask = fkwargs.get('attention_mask')
            if mask is None:
                mask = torch.ones(ids.shape[0], size + token_ids.shape[1], device=device, dtype=torch.long)
                fkwargs['attention_mask'] = mask
            pos = fkwargs.get('position_ids')
            if pos is None:
                pos = (mask.long().cumsum(-1) - 1).clamp(min=0)[:, -token_ids.shape[1]:]
                fkwargs['position_ids'] = pos
            if self.event_timing:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            else:
                sync()
            start = time.perf_counter()
            with ctl.phase(phase, mask.bool() if phase == 'prefill' else None, pos if phase == 'prefill' else None):
                ctl._guard_inputs(self.model, fargs, fkwargs)
                result = original_forward(*fargs, **fkwargs)
            if self.event_timing:
                end.record()
                events.append((phase, begin, end))
                elapsed = 0.0
            else:
                sync()
                elapsed = time.perf_counter() - start
            if phase == 'prefill':
                prefill_calls += 1
                prefill_seconds += elapsed
                cache_shapes = [list(layer.keys.shape) for layer in result.past_key_values.layers]
            else:
                decode_calls += 1
                decode_seconds += elapsed
            return result

        self.model.forward = forward
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        sync(); start = time.perf_counter()
        try:
            with torch.no_grad():
                result = self.original_generate(*args, **kwargs)
            sync(); elapsed = time.perf_counter() - start
            if self.event_timing:
                prefill_seconds = sum(a.elapsed_time(b) / 1000 for phase, a, b in events if phase == 'prefill')
                decode_seconds = sum(a.elapsed_time(b) / 1000 for phase, a, b in events if phase == 'decode')
            sequence = result.sequences if hasattr(result, 'sequences') else result
            generated = sequence[0, ids.shape[1]:].tolist()
            eos = kwargs.get('eos_token_id', config.eos_token_id)
            eos = eos if isinstance(eos, (list, tuple)) else [eos]
            max_new = kwargs.get('max_new_tokens', config.max_new_tokens)
            max_length = kwargs.get('max_length', config.max_length)
            if generated and generated[-1] in eos:
                reason = 'eos'
            elif (max_new is not None and len(generated) >= max_new) or (max_new is None and sequence.shape[1] >= max_length):
                reason = 'length_limit'
            else:
                reason = 'stopping_criteria'
            row = dict(mode=self.mode, input_ids_sha256=hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                       input_tokens=ids.shape[1],
                       valid_prompt_tokens=int(kwargs['attention_mask'].sum()) if kwargs.get('attention_mask') is not None else ids.shape[1],
                       input_ids=ids[0].tolist(), generated_token_ids=generated,
                       generated_length=len(generated), stop_reason=reason,
                       stop_reason_policy='observed EOS first, then length limit, otherwise custom criteria; simultaneous triggers not disambiguated',
                       generation_config=config.to_dict(),
                       generation_overrides={key: kwargs[key] for key in
                           ('do_sample','num_beams','max_new_tokens','max_length','eos_token_id','pad_token_id','temperature','top_p')
                           if key in kwargs},
                       prefill_seconds=prefill_seconds, decode_forward_seconds=decode_seconds,
                       total_seconds=elapsed, decode_calls=decode_calls,
                       prefill_cache_shapes=cache_shapes, prefill_pairs_by_layer=ctl.counts,
                       kernel_backend=ctl.backend,
                       routed_prefill_backend=('sdpa_explicit_mask' if ctl.backend == 'flash_attention_2' and self.mode == 'q_only' else ctl.backend),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
            if self.tokenizer is not None:
                row['prediction'] = self.tokenizer.decode(generated, skip_special_tokens=True)
            # Full input IDs remain on disk; avoid retaining 128k Python ints
            # for every benchmark document in RAM as well.
            self.rows.append(row if self.retain_input_ids else
                             {key: value for key, value in row.items() if key != 'input_ids'})
            if self.on_record:
                self.on_record(row)
            self.pending_row = row
            if not self.defer_output:
                self.flush_pending()
            return result
        finally:
            self.model.forward = original_forward
            ctl.close()
