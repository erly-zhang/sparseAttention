"""Isolated adapter for the canonical runner; no historical source edits."""
import hashlib
import inspect
import json
import time
import types
from pathlib import Path

import torch
import experiments.run_dense_multimodel_infinitebench as runner
from experiments.shared_flexprefill_20260911.adapter import METHOD, SharedFlexPrefill


runner.METHODS = tuple(runner.METHODS) + (METHOD,)
runner.SHAREPREFILL_METHODS = set(runner.SHAREPREFILL_METHODS) | {METHOD}
original_install = runner.install_benchmark_method
original_parse = runner.parse_args
REFERENCE_IDENTITIES = set()
EXPECTED_FLEX_CONFIG = {
    "block_size": 128, "flex_prefill_gamma": .95, "flex_prefill_tau": .1,
    "flex_prefill_min_budget": 512, "flex_prefill_max_budget": None,
}


def parse_args():
    args = original_parse()
    destination = Path(args.output_dir) / args.method / args.task
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    if not args.reference_metrics:
        raise ValueError("An explicit historical input reference is required")
    reference = [json.loads(line) for line in Path(args.reference_metrics).read_text().splitlines()
                 if line.strip()]
    identities = {(int(row["input_tokens"]), row["input_ids_sha256"]) for row in reference}
    if len(identities) != len(reference):
        raise ValueError("Reference contains duplicate input identities")
    REFERENCE_IDENTITIES.update(identities)
    return args


runner.parse_args = parse_args


def install(model, method, **kwargs):
    kwargs["flexprefill_min_budget"] = 512
    if method != METHOD:
        result = original_install(model, method, **kwargs)
        if method == "flexprefill" and result[2]["online_config"] != EXPECTED_FLEX_CONFIG:
            raise ValueError("Official FlexPrefill configuration drift")
        return result
    if kwargs.get("selector_dump_path") is not None:
        raise ValueError("Primitive dumps need shared-head remapping; use record_sparsity only")
    group_path = Path(kwargs["group_config_path"])
    config = json.loads(group_path.read_text())
    record = kwargs.get("record_sparsity", False)
    base_kwargs = dict(kwargs, record_sparsity=False, selector_dump_path=None)
    result, _, metadata = original_install(model, "flexprefill", **base_kwargs)
    if metadata["online_config"] != EXPECTED_FLEX_CONFIG:
        raise ValueError("Official FlexPrefill configuration drift")
    adapter = SharedFlexPrefill(result, config)
    # Wrap AFTER sharing so instrumentation counts all member masks, not three reps.
    from experiments.baseline_sparsity import install_baseline_sparsity_instrumentation
    instrumentation = (install_baseline_sparsity_instrumentation(result, "flexprefill")
                       if record else None)
    result._shared_flexprefill_adapter = adapter
    metadata.update(implementation=METHOD, num_groups_per_layer=3,
                    group_config_path=str(group_path),
                    group_config_sha256=hashlib.sha256(group_path.read_bytes()).hexdigest(),
                    shared_object="complete representative final block index set",
                    member_qkv="unchanged, correct member GQA mapping",
                    autoblock_projection=False, fixed_token_cap=None, residual=False,
                    sparsity_definition=("final-kernel causal pairs / dense causal pairs"
                                         if record else None),
                    inapplicable_runner_defaults=["fixed_topk_budget", "block_f_beta"],
                    official_selector_sha256=hashlib.sha256(
                        inspect.getsource(adapter.original).encode()).hexdigest())
    return result, instrumentation, metadata


runner.install_benchmark_method = install


class Recorder(runner.GenerationMetricsRecorder):
    def install(self):
        super().install()
        inner = self.model.generate
        recorder = self
        seen = set()

        def generate(_model, *args, **kwargs):
            from experiments.benchmark_shareprefill_ae3 import input_ids_sha256
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is None:
                raise ValueError("Missing generation input_ids")
            identity = (int(ids.shape[1]), input_ids_sha256(ids))
            if identity not in REFERENCE_IDENTITIES or identity in seen:
                raise ValueError("Unaligned or duplicate generation input")
            seen.add(identity)
            stamps = {}
            def before(_module, _args, named):
                ids = named.get("input_ids")
                if ids is not None and ids.shape[1] > 1 and "start" not in stamps:
                    torch.cuda.synchronize()
                    stamps["start"] = time.perf_counter()
            def after(_module, _args, named, output):
                if "start" in stamps and "end" not in stamps:
                    torch.cuda.synchronize()
                    stamps["end"] = time.perf_counter()
            handles = [_model.register_forward_pre_hook(before, with_kwargs=True),
                       _model.register_forward_hook(after, with_kwargs=True)]
            try:
                output = inner(*args, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            row = recorder.rows[-1]
            row["prefill_cuda_event_sec"] = row["prefill_latency_sec"]
            row["prefill_latency_sec"] = stamps["end"] - stamps["start"]
            row["decode_latency_sec"] = row["total_latency_sec"] - row["prefill_latency_sec"]
            row["timing_scope"] = "synchronized prefill incl selector, broadcast, kernel and pair statistics"
            recorder.output_path.write_text("".join(json.dumps(r) + "\n" for r in recorder.rows))
            return output

        self.model.generate = types.MethodType(generate, self.model)


runner.GenerationMetricsRecorder = Recorder
original_write_summary = runner.write_benchmark_summary


def write_summary(path, **kwargs):
    extra = dict(kwargs.get("extra") or {})
    extra["timing_scope"] = (
        "synchronized model.generate; prefill synchronized wall-clock including "
        "selector, broadcast, kernel and pair statistics; excludes model loading "
        "and the canonical 4096-token warmup; smoke may retain shape compilation"
    )
    kwargs["extra"] = extra
    return original_write_summary(path, **kwargs)


runner.write_benchmark_summary = write_summary


if __name__ == "__main__":
    runner.main()
