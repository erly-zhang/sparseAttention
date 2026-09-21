"""Only enable evaluation sample logging; preserve the prior inference path."""
import lm_eval
from experiments.shared_flexprefill_20260911 import formal_run

original_evaluate = lm_eval.simple_evaluate


def evaluate(*args, **kwargs):
    kwargs["log_samples"] = True
    result = original_evaluate(*args, **kwargs)
    assert result.get("samples", {}).get("kv_retrieval"), "Missing prediction records"
    return result


lm_eval.simple_evaluate = evaluate

if __name__ == "__main__":
    formal_run.run.runner.main()
