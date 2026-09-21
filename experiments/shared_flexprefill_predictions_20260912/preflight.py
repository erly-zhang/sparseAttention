"""Validate saved-prediction checks, then a three-input full-generation smoke."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from experiments.shared_flexprefill_predictions_20260912 import launch
from experiments.shared_flexprefill_predictions_20260912.predictions import export


def main():
    base = launch.base
    root = launch.OUT.parent
    root.mkdir(exist_ok=True)
    old_smoke = base.ROOT / "experiments/outputs/shared_flexprefill_20260911/smoke_flexprefill_ae3_shared_pattern_attempt2" / base.METHOD / "kv_retrieval"
    original = json.loads((old_smoke / "lm_eval_results.json").read_text())
    data = base.DATA / "filtered_data/kv_retrieval.jsonl"
    score = launch.scorer()
    for case in ("valid", "missing", "duplicate", "wrong_score", "wrong_target"):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            shutil.copy2(old_smoke / "online_metrics.jsonl", folder)
            result = copy.deepcopy(original)
            samples = result["samples"]["kv_retrieval"]
            if case == "missing": samples.pop()
            if case == "duplicate": samples[1] = samples[0]
            if case == "wrong_score": samples[0]["get_score_one_kv_retrieval"] = 1 - samples[0]["get_score_one_kv_retrieval"]
            if case == "wrong_target": samples[0]["target"] = "wrong"
            (folder / "lm_eval_results.json").write_text(json.dumps(result))
            try:
                audit = export(folder, data, score, 3)
            except (AssertionError, KeyError):
                assert case != "valid"
            else:
                assert case == "valid", case
            print("PREDICTION_TEST_PASS", case, flush=True)
    import lm_eval
    from experiments.shared_flexprefill_predictions_20260912 import run
    seen = {}
    def fake(**kwargs):
        seen.update(kwargs)
        return {"samples": {"kv_retrieval": [{}]}}
    run.original_evaluate = fake
    run.evaluate(log_samples=False, limit=None)
    assert seen["log_samples"] is True and seen["limit"] is None
    print("FULL_RUN_LOGGING_OVERRIDE_PASS", flush=True)
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    smoke = root / "smoke"
    command = json.loads((launch.OLD / "command.json").read_text())["command"]
    command[command.index("-m") + 1] = "experiments.shared_flexprefill_predictions_20260912.run"
    command[command.index("--output_dir") + 1] = str(smoke)
    command[command.index("--limit") + 1] = "3"
    folder = smoke / base.METHOD / "kv_retrieval"
    if not (folder / "lm_eval_results.json").exists():
        with (root / "smoke.log").open("x") as stream:
            subprocess.run(command, cwd=base.ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    audit = export(folder, data, score, 3, "predictions_input_aligned.jsonl")
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["input_alignment"]["status"] == "passed"
    assert summary["run_args"]["max_new_tokens"] == 128
    base.save(root / "preflight_identity_passed.json", dict(status="passed", count=3, prediction_audit=audit,
              input_alignment=summary["input_alignment"], full_run_logging_override=True))
    print("PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
