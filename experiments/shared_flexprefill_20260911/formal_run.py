"""Formal-only warmup wrapper; preserve the smoke-tested selector and runner."""
import json
import time
import types

import torch
from experiments.shared_flexprefill_20260911 import run


class WarmRecorder(run.Recorder):
    def install(self):
        super().install()
        measured_generate = self.model.generate
        recorder = self
        warmed = False

        def generate(model, *args, **kwargs):
            nonlocal warmed
            if not warmed:
                from experiments.benchmark_shareprefill_ae3 import input_ids_sha256
                ids = kwargs.get("input_ids", args[0] if args else None)
                identity = (int(ids.shape[1]), input_ids_sha256(ids))
                if identity not in run.REFERENCE_IDENTITIES:
                    raise ValueError("Long warmup input is not in the reference")
                warm_kwargs = dict(kwargs, max_new_tokens=2, do_sample=False)
                warm_kwargs.pop("max_length", None)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    output = recorder._original_generate(*args, **warm_kwargs)
                    del output
                    torch.cuda.synchronize()
                if recorder.patch is not None:
                    recorder.patch.flush_stats()
                    recorder.patch.reset_stats()
                evidence = dict(input_tokens=identity[0], input_ids_sha256=identity[1],
                                seconds=time.perf_counter() - start,
                                scope="first real long input, two generated tokens, outside metrics",
                                caveat="Other input shapes or data-dependent selector branches may compile later")
                with (recorder.output_path.parent / "long_warmup.json").open("x") as stream:
                    json.dump(evidence, stream, indent=2)
                print("LONG_WARMUP_COMPLETE", json.dumps(evidence), flush=True)
                warmed = True
            return measured_generate(*args, **kwargs)

        self.model.generate = types.MethodType(generate, self.model)


run.runner.GenerationMetricsRecorder = WarmRecorder
base_summary = run.runner.write_benchmark_summary


def summary(path, **kwargs):
    result = base_summary(path, **kwargs)
    data = json.loads(path.read_text())
    data["timing_scope"] = (
        "GPU-synchronized wall-clock prefill including selector, shared-mask broadcast, "
        "official attention kernel and pair statistics. Excludes loading, canonical 4K "
        "warmup and first real long-input warmup. Later unseen shapes/branches may compile."
    )
    path.write_text(json.dumps(data, indent=2))
    return result


run.runner.write_benchmark_summary = summary

if __name__ == "__main__":
    run.runner.main()
