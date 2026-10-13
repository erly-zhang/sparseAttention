"""Speed pilot: native full short fits; exact-estimator long-context row probes.

No formal lens is produced by this pilot. Long projections explicitly do not
claim all 4096 Jacobian rows were measured. Original jlens code is unchanged.
"""
import argparse
import gc
import math
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / 'jlens_prefill_analysis_20261005'
sys.path.insert(0, str(ANALYSIS))
from common import read, save, sha, runtime, load_model, Llama44LensModel
from progress import PeriodicSaver


def authorize():
    authorization = read(HERE/'authorization.json')
    if authorization.get('approved') is not True or authorization['scope'] != 'speed pilot only':
        raise PermissionError('Speed pilot not authorized')
    if 'SLURM_JOB_ID' not in os.environ:
        raise PermissionError('Requires the speed-pilot Slurm allocation')
    for path, expected in authorization['files'].items():
        if sha(path) != expected:
            raise PermissionError('Changed speed-pilot input or source: '+path)


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class FixedInput(Llama44LensModel):
    def __init__(self, hf, tokenizer, ids):
        super().__init__(hf, tokenizer)
        self.ids = ids

    def encode(self, text, *, max_length=128):
        import torch
        if len(self.ids) > max_length:
            raise ValueError('Refusing to truncate fixed full-context input')
        return torch.tensor([self.ids], dtype=torch.long, device=self.hf.device)


def probe_rows(wrapper, input_ids, *, dim_batch, batches, skip_first=16,
               source_layers=None, target_layer=None, progress=None):
    """Same cotangent/averaging as jacobian_for_prompt, a prefix of its rows.

    Hooks are removed after forward so activation-checkpoint recomputation
    does not overwrite captured forward tensors or retain recomputed tensors.
    """
    import torch
    from jlens.hooks import ActivationRecorder
    from jlens.fitting import valid_position_mask
    sources = list(range(wrapper.n_layers-1)) if source_layers is None else list(source_layers)
    target = wrapper.n_layers-1 if target_layer is None else target_layer
    seq_len = input_ids.shape[1]
    valid = valid_position_mask(seq_len, skip_first=skip_first).nonzero(as_tuple=True)[0].to(input_ids.device)
    n_passes = min(batches, math.ceil(wrapper.d_model/dim_batch))
    output = {l:torch.empty(min(n_passes*dim_batch, wrapper.d_model), wrapper.d_model)
              for l in sources}
    sync()
    t0 = time.perf_counter()
    if progress: progress(dict(phase='forward_started', seq_len=seq_len, dim_batch=dim_batch))
    with torch.enable_grad():
        with ActivationRecorder(wrapper.layers, at=[*sources,target], start_graph_at=min(sources)) as recorder:
            wrapper.forward(input_ids.expand(dim_batch,-1))
        final = recorder.activations[target]
        inputs = [recorder.activations[l] for l in sources]
        sync()
        forward_seconds = time.perf_counter()-t0
        if progress: progress(dict(phase='forward_finished', forward_seconds=forward_seconds,
                                   gpu_allocated_gib=torch.cuda.memory_allocated()/2**30))
        cotangent = torch.zeros_like(final)
        bi = torch.arange(dim_batch,device=final.device)
        times = []
        for index in range(n_passes):
            start = index*dim_batch
            count = min(dim_batch,wrapper.d_model-start)
            sync()
            begin = time.perf_counter()
            cotangent.zero_()
            cotangent[bi[:count,None],valid[None,:],start+bi[:count,None]] = 1.
            grads = torch.autograd.grad(final, inputs, grad_outputs=cotangent,
                retain_graph=index<n_passes-1)
            for l, grad in zip(sources,grads,strict=True):
                rows = grad[:count,valid,:].float().mean(dim=1)
                output[l][start:start+count] = rows.cpu()
            del grads, rows
            sync()
            elapsed = time.perf_counter()-begin
            times.append(elapsed)
            if progress: progress(dict(phase='backward_finished', batch=index+1,
                measured_dimensions=start+count, backward_seconds=elapsed,
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    if any(not torch.isfinite(j).all() for j in output.values()):
        raise ValueError('Nonfinite sampled Jacobian rows')
    # First pass includes lazy kernel setup; central estimate uses later passes.
    steady = times[1:] if len(times)>1 else times
    total_passes = math.ceil(wrapper.d_model/dim_batch)
    projected = forward_seconds + times[0] + (total_passes-1)*statistics.median(steady)
    bounds = [forward_seconds+times[0]+(total_passes-1)*f(steady) for f in (min,max)]
    return output, dict(status='measured_partial', input_tokens=seq_len,
        dim_batch=dim_batch, measured_dimensions=min(n_passes*dim_batch,wrapper.d_model),
        full_dimensions=wrapper.d_model, full_backward_batches=total_passes,
        forward_seconds=forward_seconds, backward_batch_seconds=times,
        projected_per_prompt_seconds=projected, projected_range_seconds=bounds,
        projection='forward + first batch + remaining batches times median/min/max of later measured batches; full fit NOT completed')


def set_checkpointing(model, enabled):
    if enabled:
        # HF 4.44 activates this path only in training mode. Llama has no other
        # train/eval-dependent operation when its attention dropout is zero.
        if model.config.attention_dropout != 0:
            raise ValueError('Checkpoint benchmark requires zero attention dropout')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
        model.train()
    else:
        model.gradient_checkpointing_disable()
        model.eval()


def cleanup():
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    sync()


def checkpoint_validation(model, tokenizer, row):
    import torch
    ids = torch.tensor([row['input_ids'][:128]],device=model.device)
    wrapper = FixedInput(model,tokenizer,ids[0].tolist())
    results = []
    for enabled in (False,True):
        set_checkpointing(model, enabled)
        j, result = probe_rows(wrapper,ids,dim_batch=1,batches=2)
        results.append(j)
        cleanup()
    max_abs = max(float((results[0][l]-results[1][l]).abs().max()) for l in results[0])
    for l in results[0]:
        torch.testing.assert_close(results[0][l],results[1][l],atol=.03,rtol=.03)
    set_checkpointing(model,False)
    cleanup()
    return dict(status='PASS', input_tokens=128, dimensions=2, source_layers=31,
                max_abs_row_difference=max_abs, atol=.03, rtol=.03,
                estimator_unchanged=True, checkpoint_mode='use_reentrant=False')


def run(worker):
    authorize()
    import torch
    import jlens
    runtime()
    c = read(HERE/'config.json')
    data = read(HERE/'inputs.json')
    directory = HERE/'outputs'/('worker-'+str(worker))
    directory.mkdir(parents=True,exist_ok=True)
    def progress(event):
        save(directory/'progress.json',dict(event, updated_unix=time.time(),worker=worker))
        print(event,flush=True)
    report = dict(worker=worker,job_id=os.environ['SLURM_JOB_ID'],
        config_sha256=sha(HERE/'config.json'),inputs_sha256=sha(HERE/'inputs.json'),
        formal_pipeline_run=False)
    with PeriodicSaver(directory):
        progress(dict(phase='model_loading'))
        start = time.perf_counter()
        model, tokenizer = load_model()
        sync()
        report.update(model_load_seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),
            gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            torch_version=torch.__version__,cpu_threads=torch.get_num_threads())
        progress(dict(phase='model_loaded',**{k:report[k] for k in ('gpu','model_load_seconds')}))
        try:
            # Same two-row warmup and checkpoint correctness check for every worker.
            report['checkpoint_validation'] = checkpoint_validation(model,tokenizer,data['wikitext'][0])
            save(directory/'validation.json',report['checkpoint_validation'])
            if worker == 0:
                records = []
                cases = [('wikitext',data['wikitext'][0]),('infinitebench_128',data['infinitebench'][0]),
                         ('wikitext',data['wikitext'][1]),('infinitebench_128',data['infinitebench'][1]),
                         ('infinitebench_128',data['infinitebench'][2]),('infinitebench_128',data['infinitebench'][3])]
                for index,(kind,row) in enumerate(cases):
                    # Matched-length check is supplemental; the selection uses full contexts.
                    ids = row['input_ids'][:128]
                    wrapper = FixedInput(model,tokenizer,ids)
                    cleanup()
                    torch.cuda.reset_peak_memory_stats()
                    progress(dict(phase='native_full_fit_started',index=index,kind=kind))
                    sync()
                    begin=time.perf_counter()
                    lens=jlens.fit(wrapper,['fixed-input'],source_layers=c['source_layers'],
                        target_layer=c['target_layer'],dim_batch=c['short_dim_batch'],
                        max_seq_len=128,skip_first=c['skip_first'],checkpoint_path=None,resume=False)
                    sync()
                    elapsed=time.perf_counter()-begin
                    peak=torch.cuda.max_memory_allocated()/2**30
                    if lens.n_prompts!=1 or any(not torch.isfinite(j).all() for j in lens.jacobians.values()):
                        raise ValueError('Invalid native full fit')
                    record=dict(status='completed_full_fit',kind=kind,key=row.get('key','wiki-'+str(index)),
                        input_tokens=128,dim_batch=32,dimensions=4096,backward_batches=128,
                        per_prompt_seconds=elapsed,gpu_peak_allocated_gib=peak)
                    records.append(record)
                    save(directory/('sample-'+str(index)+'.json'),record)
                    del lens
                    cleanup()
                    progress(dict(phase='native_full_fit_finished',**record))
                report['records']=records
            else:
                row = data['infinitebench'][worker-1]
                wrapper = FixedInput(model,tokenizer,row['input_ids'])
                ids = wrapper.encode('fixed-input',max_length=row['original_tokens'])
                report.update(key=row['key'],task=row['task'],input_tokens=row['original_tokens'],
                    full_context=True,selected_dim_batch_32_source_residual_gib=
                    31*32*len(row['input_ids'])*4096*2/2**30)
                attempts=[]
                for checkpoint in (False,True):
                    set_checkpointing(model,checkpoint)
                    cleanup()
                    torch.cuda.reset_peak_memory_stats()
                    attempt=dict(checkpointing=checkpoint,dim_batch=1,full_context=True)
                    progress(dict(phase='long_probe_started',key=row['key'],**attempt))
                    begin=time.perf_counter()
                    try:
                        matrices,result=probe_rows(wrapper,ids,dim_batch=1,
                            batches=c['long_backward_batches'],progress=progress)
                        attempt.update(result)
                        # A small diagnostic is sufficient; never present rows as a fitted lens.
                        attempt['sampled_row_norms']={str(l):float(j.norm()) for l,j in matrices.items()}
                        del matrices
                    except torch.cuda.OutOfMemoryError as exc:
                        attempt.update(status='cuda_out_of_memory',error=str(exc))
                    attempt.update(attempt_elapsed_seconds=time.perf_counter()-begin,
                        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                    attempts.append(attempt)
                    save(directory/('attempt-'+str(int(checkpoint))+'.json'),attempt)
                    cleanup()
                    if attempt['status']=='measured_partial':
                        break
                report['attempts']=attempts
                report['status']='complete'
            report['status']='complete'
            save(directory/'complete.json',report)
            progress(dict(phase='worker_complete'))
        except Exception as exc:
            save(directory/'failure.json',dict(report,error=repr(exc)))
            raise
        finally:
            del model
            cleanup()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,required=True,choices=range(5))
    run(parser.parse_args().worker)
