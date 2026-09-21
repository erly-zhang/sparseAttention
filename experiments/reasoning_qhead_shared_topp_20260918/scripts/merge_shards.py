#!/usr/bin/env python3
"""Merge round-robin dataset-shard outputs into the official positional layout.

Reads <base>/shard_*/<task>/run_<r>/completions.jsonl (lines carry an "idx"
field) and <base>/shard_*/<task>/qhead_selections.jsonl (raw per-selection
telemetry). Produces <out>/run_<r>/completions_shard0.jsonl in original sample
order plus an aggregated qhead_stats.json.

Duplicate detection is per (run, idx). Missing / out-of-range / duplicate /
ragged layouts abort with a non-zero exit so a bad cell is never graded.
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def stat(vals):
    vals = sorted(vals)
    return dict(
        n=len(vals),
        min=vals[0],
        max=vals[-1],
        mean=round(statistics.mean(vals), 4),
        median=round(statistics.median(vals), 4),
        p25=vals[len(vals) // 4],
        p75=vals[3 * len(vals) // 4],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="method base dir containing shard_*/")
    ap.add_argument("--out", required=True, help="merged output dir")
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--expected_samples", type=int, default=500)
    ap.add_argument("--expected_runs", type=int, default=2)
    ap.add_argument("--num_shards_seen", type=int, default=None,
                    help="raise if fewer shard dirs are present (resume safety); defaults to num_shards")
    args = ap.parse_args()
    need_shards = args.num_shards_seen if args.num_shards_seen is not None else args.num_shards

    shard_dirs = sorted(glob.glob(os.path.join(args.base, "shard_*")))
    if len(shard_dirs) != need_shards:
        raise SystemExit(
            f"expected {need_shards} shard dirs under {args.base}, found {len(shard_dirs)}: {shard_dirs}"
        )

    comps = {}          # (run_id, idx) -> completion
    gen_lens = {}       # (run_id, idx) -> generate_len
    sel_rows = []       # raw qhead selection records

    for sd in shard_dirs:
        for cf in sorted(glob.glob(os.path.join(sd, "**", "completions.jsonl"), recursive=True)):
            run_dir = os.path.dirname(cf)
            run_id = int(os.path.basename(run_dir).split("_")[1])
            oi_path = os.path.join(run_dir, "other_info.json")
            glens = []
            if os.path.exists(oi_path):
                glens = json.load(open(oi_path)).get("generate_lens", [])
            for line_i, line in enumerate(open(cf)):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = int(rec["idx"])
                key = (run_id, idx)
                if key in comps:
                    raise SystemExit(f"DUPLICATE sample: run {run_id} idx {idx}")
                comps[key] = rec["completion"]
                if line_i < len(glens):
                    gen_lens[key] = int(glens[line_i])
        for sf in glob.glob(os.path.join(sd, "**", "qhead_selections.jsonl"), recursive=True):
            for line in open(sf):
                line = line.strip()
                if line:
                    sel_rows.append(json.loads(line))

    expected = set(range(args.expected_samples))
    for run_id in range(args.expected_runs):
        idxs = {idx for (r, idx) in comps if r == run_id}
        missing = expected - idxs
        extra = idxs - expected
        if missing:
            raise SystemExit(f"run {run_id}: MISSING {len(missing)} samples: {sorted(missing)[:20]} ...")
        if extra:
            raise SystemExit(f"run {run_id}: EXTRA/OUT-OF-RANGE {len(extra)} samples: {sorted(extra)[:20]} ...")
        if len(idxs) != args.expected_samples:
            raise SystemExit(f"run {run_id}: expected {args.expected_samples} unique samples, got {len(idxs)}")

    for run_id in range(args.expected_runs):
        out_run = os.path.join(args.out, f"run_{run_id}")
        os.makedirs(out_run, exist_ok=True)
        order = sorted(expected)
        with open(os.path.join(out_run, "completions_shard0.jsonl"), "w") as f:
            for idx in order:
                f.write(json.dumps({"completion": comps[(run_id, idx)]}) + "\n")
        gl = [gen_lens.get((run_id, idx), 0) for idx in order]
        with open(os.path.join(out_run, "other_info_shard0.json"), "w") as f:
            json.dump({"generate_lens": gl}, f)

    if sel_rows:
        summary = dict(
            selections=len(sel_rows),
            selected_reasoning_K=stat([r["selected_reasoning_K"] for r in sel_rows]),
            reasoning_keep_ratio=stat([r["reasoning_keep_ratio"] for r in sel_rows]),
            selected_reasoning_mass=stat([r["selected_reasoning_mass"] for r in sel_rows]),
            representative_q_heads=sorted({r["representative_q_head"] for r in sel_rows}),
            representative_kv_heads=sorted({r["representative_kv_head"] for r in sel_rows}),
        )
        with open(os.path.join(args.out, "qhead_stats.json"), "w") as f:
            json.dump(summary, f, indent=2)

    all_idx = {idx for (r, idx) in comps}
    print(f"shards={len(shard_dirs)} runs={args.expected_runs} completions={len(comps)} unique_samples={len(all_idx)}")
    print("num_unique_samples =", len(all_idx))
    print("missing_samples =", len(expected - all_idx))
    print("duplicate_samples =", "0 (checked per run/idx)")
    print("MERGE_OK")


if __name__ == "__main__":
    main()

