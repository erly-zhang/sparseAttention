"""Rebuild the curated result index from sealed local evidence (no GPU work).

Only presentation files under results/current and RESULTS.md are replaced.
Raw summaries, grades, inference sources and scheduler jobs are never changed.
Run with --check to audit without writing. Local result data is not in Git.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

E = Path(__file__).resolve().parents[1]
B = E / "results"
ROWS = []
PROOFS = []


def read(p):
    return json.loads(p.read_text())


def jsonl(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mean_field(s, *names):
    for name in names:
        value = s.get(name)
        if value is not None:
            return value.get("mean") if isinstance(value, dict) else value
    return None


def close(a, b, label):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (label, a, b)


def reference(root, dataset):
    for p in [root / "inputs" / f"{dataset}.jsonl", root / "inputs.jsonl"]:
        if p.exists():
            rows = jsonl(p)
            return {r["idx"]: (r["input_tokens"], r["input_ids_sha256"]) for r in rows}
    raise AssertionError(f"Missing canonical input references: {root}, {dataset}")


def add(path, *, dataset, method, storage, budget, protection, interval=0,
        score="none", window=None, phase=None, grade=None, summary=None,
        config_root=None, metric_files=None, grade_cell=None, notes=""):
    p = B / path
    s = read(p) if summary is None else summary
    q, n = s["questions"], s.get("answers", s.get("rollouts", s.get("graded_outputs")))
    assert n == 2 * q
    assert s.get("status", "completed") == "completed" and not s.get("unresolved")
    assert s["input_alignment"] == "passed/full"
    phase = B / phase if phase else p.parent
    grade = B / grade if grade else phase / "grades.jsonl"
    gs = jsonl(grade)
    if grade_cell:
        gs = [g for g in gs if g["cell"] == grade_cell]
    expected = {(r, i) for r in (0, 1) for i in range(q)}
    keys = [(g["run"], g["idx"]) for g in gs]
    assert len(keys) == n and set(keys) == expected, (path, "grades coverage")
    correct = sum(g["correct"] for g in gs)
    assert correct == s["correct"]
    close(correct / n, s["accuracy"], "accuracy")
    config_root = B / (config_root or Path(path).parts[0])
    cfg = read(config_root / "config.json")
    refs = reference(config_root, dataset)
    assert len(refs) == q and set(refs) == set(range(q))
    if metric_files is None:
        metric_files = sorted(phase.glob("shard_*/cache_metrics.jsonl"))
    else:
        metric_files = [B / f for f in metric_files]
    metrics = [m for f in metric_files for m in jsonl(f)]
    assert len(metrics) == q and {m["idx"] for m in metrics} == set(refs), (path, "metric coverage")
    for m in metrics:
        assert (m["input_tokens"], m["input_ids_sha256"]) == refs[m["idx"]], (path, m["idx"])
        assert len(m["generated_tokens_per_rollout"]) == 2
        assert m["decode_forward_calls"] == max(m["generated_tokens_per_rollout"]) - 1
    decode = statistics.mean(m["decode_seconds"] for m in metrics)
    close(decode, mean_field(s, "mean_decode_seconds", "decode_seconds", "decode_seconds_per_question_batch"), "decode mean")
    tokens = [t for m in metrics for t in m["generated_tokens_per_rollout"]]
    close(statistics.mean(tokens), mean_field(s, "mean_generated_tokens", "generated_tokens"), "output mean")
    selected, causal = [sum(m[k] for m in metrics) for k in ("selected_pairs", "causal_pairs")]
    assert selected == s["selected_pairs"] and causal == s["causal_pairs"]
    close(1 - selected / causal, s["global_pair_sparsity"], "pair sparsity")
    record = dict(dataset=dataset, model=Path(cfg["model"]).name, method=method,
        score=score, window=window, storage=storage, budget=budget,
        protection=protection, compaction_interval=interval, questions=q,
        answers=n, correct=correct, accuracy=correct / n, decode_seconds=decode,
        prefill_seconds=statistics.mean(m["prefill_seconds"] for m in metrics),
        decode_forward_calls_total=sum(m["decode_forward_calls"] for m in metrics),
        decode_forward_calls_mean=statistics.mean(m["decode_forward_calls"] for m in metrics),
        decode_ms_per_forward=1000 * sum(m["decode_seconds"] for m in metrics) / sum(m["decode_forward_calls"] for m in metrics),
        output_tokens_mean=statistics.mean(tokens), output_tokens_median=statistics.median(tokens),
        cap_hits=sum(not eos for m in metrics for eos in m["eos_reached"]),
        peak_memory_gib=max(m["peak_memory_allocated"] for m in metrics) / 2**30,
        selected_pairs=selected, causal_pairs=causal, global_pair_sparsity=1-selected/causal,
        input_alignment="passed/full", verification="grades+metrics+input_hashes",
        seed=cfg["seed"], batch_size=cfg["batch_size"], max_tokens=cfg["max_tokens"],
        generation_temperature=cfg["generation_temperature"], generation_top_p=cfg["generation_top_p"],
        input_fingerprint=hashlib.sha256(json.dumps(sorted(refs.items())).encode()).hexdigest(),
        source=path, source_sha256=sha(p), status="current", notes=notes,
        evidence=dict(grades=str(grade.relative_to(B)), grades_sha256=sha(grade),
            metric_files=[dict(path=str(f.relative_to(B)),sha256=sha(f)) for f in metric_files],
            config=str((config_root / "config.json").relative_to(B)), config_sha256=sha(config_root / "config.json")))
    ROWS.append(record)
    return record


def build():
    grid = "hmmt_k1024_k2048_full_compact_k4096_full_20261001"
    for p in sorted((B / grid / "cells").glob("*/formal/summary.json")):
        c = read(p)["cell"]
        add(str(p.relative_to(B)), dataset="hmmt", method="W1" if c["scorer"] == "w1" else "Shared Random",
            storage=c["storage"], budget=f"total K={c['K']}", protection=c["protection"],
            interval=64 if c["storage"] == "compact" else 0,
            score="probability" if c["scorer"] == "w1" else "Uniform TopK", window=c["window"] if c["scorer"] == "w1" else None)
    grid500 = "hmmt_compact_interval500_k1024_k2048_k4096_20261007"
    for p in sorted((B / grid500 / "cells").glob("*/formal/summary.json")):
        c = read(p)["cell"]
        add(str(p.relative_to(B)), dataset="hmmt", method="W1" if c["scorer"] == "w1" else "Shared Random",
            storage="compact", budget=f"total K={c['K']}", protection=c["protection"], interval=500,
            score="probability" if c["scorer"] == "w1" else "Uniform TopK", window=1 if c["scorer"] == "w1" else None)
    original = "shared_selector_grid_20260929"
    report = f"{original}/schedule_hmmt_first_20260930/reports/hmmt_compact_random_w1_four_cells"
    for p in sorted((B / report).glob("*_summary.json")):
        c = read(p)["cell"]
        add(str(p.relative_to(B)), dataset="hmmt", method="W1" if c["scorer"] == "w1" else "Shared Random",
            storage="compact", budget="total K=4096", protection=c["protection"], interval=64,
            score="probability" if c["scorer"] == "w1" else "Uniform TopK", window=1 if c["scorer"] == "w1" else None,
            phase=f"{original}/cells/{c['name']}/formal", grade=f"{report}/{c['name']}_grades.jsonl", config_root=original)
    for root in ["hmmt_w1_raw_qk_k1024_full_compact500_20261009", "hmmt_w1_raw_qk_k1024_hybrid4096_compact1000_20261011"]:
        for p in sorted((B / root / "cells").glob("*/formal/summary.json")):
            c = read(p)["cell"]
            hybrid = "hybrid" in c["name"]
            full = c["storage"] == "full"
            add(str(p.relative_to(B)), dataset="hmmt", method="W1 + random4096" if hybrid else "W1",
                storage="hybrid" if hybrid else c["storage"], budget="total K=1024", protection="protected",
                interval=0 if full else (1000 if "1000" in root else 500), score="raw-QK", window=1)
    # The maintenance replacement changes only orchestration. Publish these
    # new K2048 algorithm cells after BOTH formal collections pass, never from
    # partially written worker metrics or reused smoke answers.
    short = "hmmt_w1_raw_qk_k2048_hybrid4096_compact1000_shortshards_20261012"
    collected = B / short / "summary.json"
    if collected.exists() and read(collected).get("status") == "completed":
        for p in sorted((B / short / "cells").glob("*/formal/summary.json")):
            c = read(p)["cell"]
            hybrid = c["storage"] == "hybrid"
            add(str(p.relative_to(B)), dataset="hmmt", method="W1 + random4096" if hybrid else "W1",
                storage=c["storage"], budget="total K=2048", protection="protected", interval=1000,
                score="raw-QK", window=1, notes="Maintenance replacement: same inference sources;20 short shards/method instead of4.")
    add("hmmt_dense_and_timing_diagnosis_20261008/formal/summary.json", dataset="hmmt", method="Dense",
        storage="full", budget="all", protection="all")
    fast = add("hmmt_full_w1_optimized_k2048_protected_20261009/summary.json", dataset="hmmt", method="W1",
        storage="full", budget="total K=2048", protection="protected", score="probability", window=1,
        phase="hmmt_full_w1_optimized_k2048_protected_20261009/formal", grade="hmmt_full_w1_optimized_k2048_protected_20261009/grades.jsonl",
        notes="Implementation optimization v1; identical 120 completions to previous implementation. Different A100 allocations.")
    old = next(r for r in ROWS if r["source"] == f"{grid}/cells/hmmt_k2048_grid_full_w1_topk_protected/formal/summary.json")
    # Compare every complete text, beyond merely agreeing on correctness counts.
    def completions(phase):
        pairs = [(int(p.parent.name.split("_")[1]), r["idx"], r["completion"])
                 for p in sorted(phase.glob("shard_*/run_*/completions.jsonl")) for r in jsonl(p)]
        assert len(pairs) == 120 and len({(a,b) for a,b,_ in pairs}) == 120
        return {(a,b):c for a,b,c in pairs}
    assert old["input_fingerprint"] == fast["input_fingerprint"]
    assert completions((B / old["source"]).parent) == completions(B / "hmmt_full_w1_optimized_k2048_protected_20261009/formal")
    assert (old["selected_pairs"], old["causal_pairs"]) == (fast["selected_pairs"], fast["causal_pairs"])
    assert fast["decode_seconds"] < old["decode_seconds"]
    old["status"] = "superseded"
    old["superseded_by"] = fast["source"]
    PROOFS.append(dict(old=old["source"], new=fast["source"], identical_completions=120,
        identical_input_hashes=True, identical_pairs=True,
        decode_reduction_pct=100*(1-fast["decode_seconds"]/old["decode_seconds"])))
    # Replace only an equivalent implementation, not an algorithm with a
    # higher observed score. The v3 layer benchmark is not a formal HMMT run.
    fast_root = "hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_packed32"
    collected = B / fast_root / "summary.json"
    if collected.exists() and read(collected).get("status") == "completed":
        for p in sorted((B / fast_root / "cells").glob("*/formal/summary.json")):
            c = read(p)["cell"]
            hybrid = c["storage"] == "hybrid"
            method = "W1 + random4096" if hybrid else "W1"
            newer = add(str(p.relative_to(B)), dataset="hmmt", method=method,
                storage=c["storage"], budget="total K=1024", protection="protected",
                interval=1000 if hybrid else 0, score="raw-QK", window=1,
                notes="Optimized v2; all120 complete output texts and pair counts match the previous same-storage raw-QK implementation. Different A100 allocations.")
            older = next(r for r in ROWS if r is not newer and
                r["dataset"] == "hmmt" and r["method"] == method and
                r["score"] == "raw-QK" and r["storage"] == c["storage"] and
                r["budget"] == "total K=1024" and r["protection"] == "protected")
            assert older["input_fingerprint"] == newer["input_fingerprint"]
            assert completions((B / older["source"]).parent) == completions(p.parent)
            assert (older["selected_pairs"], older["causal_pairs"]) == (newer["selected_pairs"], newer["causal_pairs"])
            older["status"] = "superseded"
            older["superseded_by"] = newer["source"]
            PROOFS.append(dict(old=older["source"], new=newer["source"],
                identical_completions=120, identical_input_hashes=True, identical_pairs=True,
                decode_reduction_pct=100*(1-newer["decode_seconds"]/older["decode_seconds"])))
    official = "hmmt_official_random_pp_k1024_k2048_20261008"
    for s in read(B / official / "summary.json")["cells"]:
        cell = f"official_random_pp_k{s['K']}_recent64"
        phase = f"{official}/cells/{cell}/formal"
        files = sorted((B / phase).glob("shard_*/cache_metrics.jsonl"))
        files += sorted((B / official / "cells" / cell / "smoke").glob("*/cache_metrics.jsonl"))
        add(f"{official}/summary.json", summary=s, dataset="hmmt", method="Official random_pp",
            storage="compact", budget=f"total K={s['K']}", protection="protected", interval=64,
            score="Uniform TopK", phase=phase, grade=f"{official}/grades.jsonl", grade_cell=cell,
            metric_files=[str(f.relative_to(B)) for f in files],
            notes="Independent per-KV-head masks; native cache-length modulo64 phase; first3 smoke inputs reused once.")
    math_specs = [
        ("full_kv_random_r040_recent64_gpu_20260924/random/summary.json", "Shared Random", "full", "40% of reasoning", 0, "Uniform TopK", None),
        ("full_kv_bernoulli_r020_recent64_recovery_20260929/random/summary.json", "Shared Random", "full", "Bernoulli p=0.20", 0, "Bernoulli", None),
        ("full_kv_random_reason1024_memory_repair_20260928/random/summary.json", "Shared Random", "full", "prompt + reasoning1024", 0, "Uniform TopK", None),
        ("compact_kv_random_reason1024_recent64_20260926/random/summary.json", "Shared Random", "compact", "prompt + reasoning1024", 64, "Uniform TopK", None),
        ("representative_w1_compact_math_20260926/math/formal/summary.json", "W1", "compact", "prompt + reasoning1024", 64, "probability", 1),
        ("official_random_pp_k1024_r64_20260924/formal/summary.json", "Official random_pp", "compact", "total K=1024", 64, "Uniform TopK", None),
        ("full_kv_w1_p095_20260924/formal/summary.json", "W1", "full", "TopP=0.95 uncapped", 0, "probability", 1),
    ]
    for path, method, storage, budget, interval, score, window in math_specs:
        notes = ""
        metrics = None
        if "memory_repair" in path:
            metrics = [str(Path(path).parent / "question_metrics.jsonl")]
            notes = "429 historical +71 repaired inputs. Mixed-source timing is not a fresh full500 repaired speed measurement."
        add(path, dataset="math", method=method, storage=storage, budget=budget,
            protection="prompt only" if "p095_20260924" in path else "protected", interval=interval,
            score=score, window=window, metric_files=metrics, notes=notes)
    root = "full_kv_w16_w64_20260922"
    comparison = read(B / root / "comparison_summary.json")
    for w, s in comparison["methods"].items():
        s = dict(s, status="completed", input_alignment=comparison["input_alignment"])
        add(f"{root}/comparison_summary.json", summary=s, dataset="math", method=f"W{w}",
            storage="full", budget="TopP=0.95 uncapped", protection="prompt only", score="probability", window=int(w), phase=f"{root}/w{w}")
    root = "compact_multibench_20260926"
    for task in ["gpqa", "aime25", "aime26", "hmmt", "livecodebench"]:
        add(f"{root}/{task}/formal/summary.json", dataset=task, method="Shared Random", storage="compact",
            budget="prompt + reasoning1024", protection="protected", interval=64, score="Uniform TopK",
            notes="LCB release_v6 medium383, test-based pass@1 estimate." if task == "livecodebench" else "")
    # Legacy Dense: retain the complete accuracy reference; do not invent decode
    # timing or input-ID hashes that its runner never recorded.
    path = "full_kv_random_r040_recent64_gpu_20260924/reference_baselines/dense/summary.json"
    s = read(B / path)
    root = B / "dense_baseline_20260920/sharded8"
    comps = [(int(p.parent.name.split("_")[1]), r["idx"]) for p in root.glob("shard_*/*/run_*/completions.jsonl") for r in jsonl(p)]
    assert len(comps) == 1000 and set(comps) == {(run, i) for run in (0,1) for i in range(500)}
    assert s["checks"]["duplicates"] == s["checks"]["missing"] == 0
    ROWS.append(dict(dataset="math", model="Qwen3-4B", method="Dense (legacy)", score="none", window=None,
        storage="full", budget="all", protection="all", compaction_interval=0, questions=500, answers=1000,
        correct=s["pooled"]["correct"], accuracy=s["pooled"]["flag_acc"], decode_seconds=None,
        batch_generation_seconds=s["batch_generation_seconds"]["mean"], output_tokens_mean=s["pooled"]["generated_tokens"]["mean"],
        output_tokens_median=s["pooled"]["generated_tokens"]["median"],
        input_alignment="legacy decoded-prompt check; token hash unavailable", verification="complete outputs+historical graded summary",
        source=path, source_sha256=sha(B / path), status="current",
        notes="Historical batch generation wall 284.8s; no separated synchronized decode time, pair counters, peak memory or saved input-ID hashes."))


def complete_build():
    build()
    primary = [r for r in ROWS if r["status"] == "current"]
    keys = [(r["dataset"], r["method"], r["score"], r["window"], r["storage"], r["budget"],
             r["protection"], r["compaction_interval"]) for r in primary]
    assert len(keys) == len(set(keys)), "Duplicate primary method/settings"
    assert len({r["input_fingerprint"] for r in ROWS if r["dataset"] == "hmmt"}) == 1, "Mismatched HMMT inputs"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def fmt(value, digits=1):
    return "—" if value is None else f"{value:,.{digits}f}"


def table(records):
    lines = ["| Method / score | Budget | KV storage | Interval | Protection | Correct | Accuracy | Decode s/question | Step ms | Output tokens | Source |",
             "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in records:
        method = r["method"] + (f" ({r['score']})" if r["score"] != "none" else "")
        lines.append(f"| {method} | {r['budget']} | {r['storage']} | {r['compaction_interval']} | {r['protection']} | {r['correct']}/{r['answers']} | {r['accuracy']:.2%} | {fmt(r.get('decode_seconds'))} | {fmt(r.get('decode_ms_per_forward'))} | {fmt(r.get('output_tokens_mean'))} | [summary](results/{r['source']}) |")
    return lines


def inventory():
    source_roots = {Path(r["source"]).parts[0] for r in ROWS}
    source_roots.add("dense_baseline_20260920")
    failed = {
        "full_kv_random_reason1024_recent64_20260925": "OOM before all500 inputs; replaced in current table by 429+71 merged repair.",
        "full_kv_random_reason1024_repaired_full500_20260928": "Fresh full500 rerun incomplete: shard4 failed; merged repair remains current.",
        "full_kv_w1_p095_recent64_gpu_20260924": "Incomplete: collector failed on shard6; no full500 accuracy.",
        "hmmt_w1_raw_qk_k2048_hybrid4096_compact1000_20261011": "Failed correctness gate; superseded by validationfix run.",
        "hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012": "Failed GPU correctness gate; superseded by validationfix/packed32 runs.",
    }
    result = []
    roots = [p for p in B.iterdir() if p.is_dir() and p.name not in ("current", "archive")]
    known = {p.name for p in roots}
    if (B / "archive/runs").exists():
        roots += [p for p in (B / "archive/runs").iterdir() if p.is_dir() and p.name not in known]
    for p in sorted(roots, key=lambda p: p.name):
        if not p.is_dir():
            continue
        category = "historical_or_pilot"
        reason = "Distinct historical setup, partial run or diagnostic; excluded from full-dataset comparison."
        if p.name in source_roots:
            category, reason = "formal_source", "Contains one or more curated complete results."
        elif p.name in failed:
            category, reason = "failed_or_superseded", failed[p.name]
        elif "incomplete_" in p.name:
            category, reason = "duplicate_snapshot", "Incomplete preparation snapshot; no experiment was submitted."
        elif "smoke" in p.name or "validationfix" in p.name:
            category, reason = "smoke", "Small correctness/smoke validation; not full-dataset accuracy."
        supersession = p / "scheduler_supersession.json"
        if supersession.exists():
            category = "superseded_orchestration"
            reason = read(supersession)["reason"]
        result.append(dict(root=p.name, category=category, reason=reason, physical_path=str(p.resolve().relative_to(B))))
    return result


def write():
    current = sorted((r for r in ROWS if r["status"] == "current"), key=lambda r: (r["dataset"], r["budget"], r["protection"], r["storage"], r["compaction_interval"], r["method"], r["score"]))
    history = [r for r in ROWS if r["status"] != "current"]
    latest_hmmt_roots = {"hmmt_dense_and_timing_diagnosis_20261008", "hmmt_full_w1_optimized_k2048_protected_20261009",
        "hmmt_official_random_pp_k1024_k2048_20261008", "hmmt_w1_raw_qk_k1024_full_compact500_20261009",
        "hmmt_w1_raw_qk_k1024_hybrid4096_compact1000_20261011",
        "hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_packed32",
        "hmmt_w1_raw_qk_k2048_hybrid4096_compact1000_shortshards_20261012"}
    for r in current:
        r["display_role"] = "historical_control" if r["dataset"] == "hmmt" and Path(r["source"]).parts[0] not in latest_hmmt_roots else "primary"
    inv = inventory()
    out = B / "current"
    out.mkdir(exist_ok=True)
    built = datetime.now(timezone.utc).isoformat()
    write_json(out / "summary.json", dict(updated_utc=built, scope="Local Qwen3-4B reasoning experiments", results=current, supersession_proofs=PROOFS))
    write_json(out / "implementation_history.json", history)
    write_json(out / "run_inventory.json", inv)
    write_json(out / "validation.json", dict(status="passed", current_complete_results=len(current), superseded_results=len(history),
        grades_checked=sum(r["answers"] for r in ROWS if r["verification"] == "grades+metrics+input_hashes"),
        metric_records_checked=sum(r["questions"] for r in ROWS if r["verification"] == "grades+metrics+input_hashes"),
        checks=["unique grade IDs and complete coverage", "recomputed correctness", "exact input tokens/SHA256", "unique metrics",
                "decode/output means recomputed", "pair sums and sparsity recomputed", "120 exact completion pairs for supersession"],
        legacy_dense_limit="Complete1000 outputs; historical grading reused; input-ID hashes and separated decode timing unavailable."))
    columns = ["dataset", "display_role", "model", "method", "score", "window", "storage", "budget", "protection", "compaction_interval", "questions", "answers", "correct", "accuracy", "prefill_seconds", "decode_seconds", "decode_forward_calls_total", "decode_forward_calls_mean", "decode_ms_per_forward", "output_tokens_mean", "output_tokens_median", "cap_hits", "peak_memory_gib", "selected_pairs", "causal_pairs", "global_pair_sparsity", "input_alignment", "source", "notes"]
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(current)
    lines = ["# Complete result archive and current comparisons", "", f"Updated: {built[:10]}. Fixed entry point; replaces dated comparison tables.", "",
        f"{len(current)} current complete result groups; {len(history)} equivalent older implementations are listed only in history. All methods use Qwen3-4B.", "",
        "[Current English PDF report](../../../report_build/current/hmmt_results_en.pdf) · [Code and method guide](CODE_INDEX.md) · [Report directory](REPORTS.md)", "",
        "The main PDF selects8 latest protected raw-QK W1 / Dense / official random_pp configurations. This complete catalog intentionally retains earlier validated controls and other datasets; use the separate historical PDF for those references.", "",
        "## Reading the results", "",
        "Accuracy is correct sampled outputs / total outputs, not pass@2. HMMT has 60 questions × 2 outputs = 120; MATH500 has 500 × 2 = 1000.", "",
        "Decode seconds measure one question batch producing two outputs concurrently. Step ms is total synchronized decode seconds / total batch decode forwards. Output tokens are per individual output.",
        "Shared Random and official random_pp use different mask sharing and eviction phases. `prompt + reasoning1024` gives prompt tokens extra budget; `total K=1024` includes protection inside 1024. These are separate algorithms.", "",
        "Scores marked probability and raw-QK remain separate because BF16 ties/underflow can change selected tokens. Compaction intervals 64/500/1000 and hybrid random4096 are also separate.", "",
        "HMMT settings: BF16, seed 42, representative seed 0, batch 2, temperature 0.6, generation Top-p 0.95, total prompt+output cap 32768. Different A100 allocations and freely generated lengths limit speed comparisons.", "",
        "## Superseded implementation", "",
        "Only same-algorithm implementations with120 exact completion pairs, identical input hashes and identical attention-pair counts replace an older result. Different scores, budgets, storage policies or compression intervals remain separate controls. Observed timing differences use different A100 allocations.", "",
        "| Configuration | Previous decode s/question | Current decode s/question | Correct | Observed reduction |",
        "|---|---:|---:|---:|---:|",
        "| W1 probability, full KV, K2048 | 1,790.8 | 1,270.4 | 52/120 | 29.06% |",
        "| W1 raw-QK, full KV, K1024 | 1,326.8 | 1,267.8 | 48/120 | 4.45% |",
        "| W1 raw-QK + random4096, interval1000, K1024 | 1,316.2 | 1,252.3 | 46/120 | 4.85% |", "",
        "Old result and proof: [implementation history](results/current/implementation_history.json). Raw run evidence remains recoverable.", "",
        "## HMMT: latest complete implementations and matched baselines", ""]
    hmmt = [r for r in current if r["dataset"] == "hmmt"]
    lines += table([r for r in hmmt if r["display_role"] == "primary"])
    lines += ["", "<details>", "<summary>Other complete HMMT controls: K=1024/2048/4096, Random/W1, protection and intervals</summary>", "",
              "These settings are independent controls; their algorithms differ from the latest implementations.", "",
              "### Protected total-K controls", ""]
    lines += table([r for r in hmmt if r["display_role"] == "historical_control" and r["protection"] == "protected" and r["budget"] != "prompt + reasoning1024"])
    lines += ["", "### No forced protection and older extra-prompt budget", ""]
    lines += table([r for r in hmmt if r["protection"] == "none" or r["budget"] == "prompt + reasoning1024"])
    lines += ["", "</details>"]
    lines += ["", "## MATH500", ""] + table([r for r in current if r["dataset"] == "math"])
    lines += ["", "Legacy Dense has no separated decode time. Historical batch-generation mean 284.8s is a different timing scope.", "",
        "Random full-KV reasoning1024: 854/1000 comes from 429 historical + 71 repaired inputs; its mixed mean 554.0s is not a fresh full500 repaired-speed benchmark. The fresh `repaired_full500` rerun failed on shard 4 and is excluded.", "",
        "## Other complete dataset splits", ""] + table([r for r in current if r["dataset"] not in ("math", "hmmt")])
    lines += ["", "## Validation-only optimizations and historical evidence", "",
        "The v3 fused raw-QK full/hybrid implementation has passed GPU correctness and same-device single-layer timing checks at K1024/K2048. It has no new full-HMMT result and does not replace a formal accuracy/timing row. See [v3 validation](W1_RAW_QK_FUSED_OPTIMIZATION_20261014.md). Run inventory classifies stored evidence; use the live scheduler for current job status.",
        "Pilot W8, probe, length40960, profiler and failed/partial outputs are outside the full-dataset tables. Older valid algorithms remain independent controls; a higher single-seed score alone is not a replacement criterion.",
        "[Run inventory](results/current/run_inventory.json) · [archive index](results/archive/README.md) · [CSV](results/current/results.csv) · [source-backed JSON](results/current/summary.json) · [validation](results/current/validation.json)", "",
        "Scope excludes the separate Llama prefill/KV-retrieval and J-lens studies; those files remain independent. Older dated PDF/TeX reports are historical snapshots; use this entry point for current reasoning results.", "",
        "Rebuild or audit from this experiment directory:", "", "```bash", "python scripts/consolidate_results.py", "python scripts/consolidate_results.py --check", "```", ""]
    (E / "RESULTS.md").write_text("\n".join(lines))
    (out / "README.md").write_text("# Current results\n\nUse [RESULTS.md](../../RESULTS.md) for the curated comparison, [results.csv](results.csv) for analysis, and [summary.json](summary.json) for full provenance. Rebuild with `python scripts/consolidate_results.py` from the experiment directory. Raw run outputs are never overwritten by this command.\n")
    print(json.dumps(dict(current=len(current), superseded=len(history), grades_checked=sum(r["answers"] for r in ROWS if r["verification"] == "grades+metrics+input_hashes"), run_roots=len(inv))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    complete_build()
    if args.check:
        print(json.dumps(dict(status="passed", complete_groups=len(ROWS), supersession_proofs=PROOFS)))
    else:
        write()


if __name__ == "__main__":
    main()
