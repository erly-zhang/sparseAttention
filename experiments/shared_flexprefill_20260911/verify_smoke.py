"""Verify both three-input smoke outputs without modifying their source artifacts."""
import hashlib
import json
from pathlib import Path
import statistics

root = Path("/project/yl258/jz648/sparseAttention/experiments/outputs/shared_flexprefill_20260911")
methods = {"flexprefill_ae3_shared_pattern": 2, "flexprefill": 1}
results = {}
identities = []
for method, attempt in methods.items():
    run = root / f"smoke_{method}_attempt{attempt}"
    assert json.loads((run / "exit.json").read_text())["returncode"] == 0
    path = run / method / "kv_retrieval"
    summary = json.loads((path / "summary.json").read_text())
    rows = [json.loads(x) for x in (path / "online_metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    identity = [(r["input_tokens"], r["input_ids_sha256"]) for r in rows]
    assert len(set(identity)) == 3
    identities.append(identity)
    alignment = summary["input_alignment"]
    assert alignment["status"] == "passed" and alignment["count"] == 3
    assert alignment["reference_count"] == 497
    assert alignment["alignment_scope"] == "ordered_subsequence"
    metadata = summary["method_metadata"]
    expected_config = {"block_size": 128, "flex_prefill_gamma": .95, "flex_prefill_tau": .1,
                       "flex_prefill_min_budget": 512, "flex_prefill_max_budget": None}
    assert metadata["online_config"] == expected_config
    if method != "flexprefill":
        assert metadata["num_groups_per_layer"] == 3
        assert metadata["fixed_token_cap"] is None
    for row in rows:
        n = row["input_tokens"]
        assert row["causal_token_pairs"] == 32 * 32 * n * (n + 1) // 2
        assert 0 < row["selected_token_pairs"] <= row["causal_token_pairs"]
        assert 0 < row["generated_tokens"] <= 128
        assert 0 < row["prefill_latency_sec"] < row["total_latency_sec"]
        assert row["timing_scope"].startswith("synchronized prefill")
    selected = sum(r["selected_token_pairs"] for r in rows)
    causal = sum(r["causal_token_pairs"] for r in rows)
    score = summary["official_lm_eval_results"]["kv_retrieval"]["get_score_one_kv_retrieval,none"]
    assert 0 <= score <= 1
    assert summary["run_args"]["max_new_tokens"] == 128
    assert not summary["run_args"]["profile_one_token"]
    result = dict(count=3, accuracy=score, correct=round(3 * score),
                  input_alignment=alignment, hardware=summary["hardware"], online_config=expected_config,
                  selected_token_pairs=selected, causal_token_pairs=causal,
                  global_pair_sparsity=1-selected/causal,
                  peak_memory_bytes=max(r["peak_memory_bytes"] for r in rows),
                  prefill_each=[r["prefill_latency_sec"] for r in rows],
                  mean_prefill=statistics.mean(r["prefill_latency_sec"] for r in rows),
                  mean_decode=statistics.mean(r["decode_latency_sec"] for r in rows),
                  mean_total=statistics.mean(r["total_latency_sec"] for r in rows),
                  source_summary_sha256=hashlib.sha256((path / "summary.json").read_bytes()).hexdigest(),
                  timing_scope=rows[0]["timing_scope"],
                  caveat="Smoke may include new-shape compilation; inherited summary timing label is stale. Use row wall-clock fields.")
    results[method] = result
assert identities[0] == identities[1], "Control and treatment input order differs"
results["verification"] = "passed: identical 3 inputs, full generation, 32 layers x 32 Q heads counted"
destination = root / "verified_smoke_comparison.json"
with destination.open("x") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
