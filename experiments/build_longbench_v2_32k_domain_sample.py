#!/usr/bin/env python3
"""
Build a small LongBench-v2 jsonl for shared-mask experiments:
  - 32k token budget per assembled prompt (same truncation logic as build_longbench_v2_jsonl.py)
  - Sample from multiple domains (default: 3 domains x 1 sample each)

Output jsonl records keep all original fields plus:
  - prompt_token_len
  - context_token_len
  - scaffold_token_len
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse prompt/source helpers from datasets script (keep in sync).
_BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "datasets" / "longbench_v2" / "build_longbench_v2_jsonl.py"
import importlib.util

_spec = importlib.util.spec_from_file_location("build_longbench_v2_jsonl", _BUILD_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
build_longbench_v2_prompt = _mod.build_longbench_v2_prompt
resolve_source = _mod.resolve_source


ALL_LONGBENCH_DOMAINS = [
    "Long In-context Learning",
    "Single-Document QA",
    "Code Repository Understanding",
    "Multi-Document QA",
    "Long Structured Data Understanding",
    "Long-dialogue History Understanding",
]

DEFAULT_DOMAINS = ALL_LONGBENCH_DOMAINS[:3]

DEFAULT_HEAD_SELECTION_DOMAINS = [
    "Long In-context Learning",
    "Single-Document QA",
    "Code Repository Understanding",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build 32k LongBench-v2 jsonl with per-domain sampling"
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="LongBench-v2 data.json (default: HF cache auto-detect)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="/home/ubuntu/work/experiments/data/longbench_v2_32k_3domain.jsonl",
    )
    p.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=DEFAULT_DOMAINS,
        help="Domains to sample from (one block per domain)",
    )
    p.add_argument(
        "--samples_per_domain",
        type=int,
        default=1,
        help="Number of samples to pick per domain",
    )
    p.add_argument(
        "--total_samples",
        type=int,
        default=None,
        help="If set, sample this many records across --domains (round-robin).",
    )
    p.add_argument(
        "--head_selection_num_samples",
        type=int,
        default=None,
        help="Two-phase mode: N head-selection samples, each from a distinct domain.",
    )
    p.add_argument(
        "--eval_num_samples",
        type=int,
        default=None,
        help="Two-phase mode: M task-eval samples with broad domain coverage.",
    )
    p.add_argument(
        "--head_selection_domains",
        type=str,
        nargs="*",
        default=None,
        help="Domains for head selection (default: 3 distinct domains).",
    )
    p.add_argument(
        "--eval_domains",
        type=str,
        nargs="*",
        default=None,
        help="Domains for task eval round-robin (default: all LongBench-v2 domains).",
    )
    p.add_argument(
        "--max_total_tokens",
        type=int,
        default=32768,
        help="Whole prompt token budget (Qwen2.5-3B window)",
    )
    p.add_argument(
        "--margin",
        type=int,
        default=64,
        help="Reserved tokens below max_total_tokens",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="/home/ubuntu/work/model/Qwen2.5-3B",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for within-domain sampling",
    )
    p.add_argument(
        "--prefer_long",
        action="store_true",
        default=True,
        help="Prefer records whose context needs truncation (long examples)",
    )
    p.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=None,
        help="Optional lower bound on assembled prompt token length",
    )
    p.add_argument(
        "--use_all_records",
        action="store_true",
        help=(
            "Two-phase mode: include every LongBench-v2 record (after truncation). "
            "3 head-selection + all remaining as task_eval."
        ),
    )
    return p.parse_args()


def truncate_row_to_budget(
    row: Dict[str, Any],
    tokenizer,
    *,
    max_total_tokens: int,
    margin: int,
) -> Dict[str, Any]:
    context = str(row.get("context", ""))
    scaffold = build_longbench_v2_prompt({**row, "context": ""})
    scaffold_tokens = len(tokenizer(scaffold, add_special_tokens=False)["input_ids"])
    budget = max(0, max_total_tokens - scaffold_tokens - margin)

    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    truncated = len(ctx_ids) > budget
    if truncated:
        ctx_ids = ctx_ids[:budget]
        new_context = tokenizer.decode(ctx_ids, skip_special_tokens=True)
    else:
        new_context = context

    out = dict(row)
    out["context"] = new_context
    full_prompt = build_longbench_v2_prompt(out)
    prompt_token_len = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"])
    out["prompt_token_len"] = prompt_token_len
    out["context_token_len"] = len(ctx_ids)
    out["scaffold_token_len"] = scaffold_tokens
    out["context_truncated"] = truncated
    return out


def group_records_by_domain(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        domain = str(row.get("domain", "unknown"))
        grouped[domain].append(row)
    return grouped


def pick_domain_samples(
    candidates: List[Dict[str, Any]],
    tokenizer,
    *,
    samples_per_domain: int,
    max_total_tokens: int,
    margin: int,
    prefer_long: bool,
    min_prompt_tokens: Optional[int],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    processed = [
        truncate_row_to_budget(
            row,
            tokenizer,
            max_total_tokens=max_total_tokens,
            margin=margin,
        )
        for row in candidates
    ]

    if min_prompt_tokens is not None:
        processed = [r for r in processed if r["prompt_token_len"] >= min_prompt_tokens]

    if prefer_long:
        processed.sort(
            key=lambda r: (r["context_truncated"], r["prompt_token_len"]),
            reverse=True,
        )
    else:
        rng.shuffle(processed)

    return processed[:samples_per_domain]


def _take_unique_row(
    picks: List[Dict[str, Any]],
    start_idx: int,
    used_ids: set[str],
) -> tuple[Optional[Dict[str, Any]], int]:
    idx = start_idx
    while idx < len(picks):
        row = picks[idx]
        row_id = str(row.get("_id", ""))
        if row_id and row_id not in used_ids:
            return dict(row), idx + 1
        idx += 1
    return None, idx


def build_two_phase_samples(
    grouped: Dict[str, List[Dict[str, Any]]],
    tokenizer,
    *,
    head_selection_num_samples: int,
    eval_num_samples: int,
    head_selection_domains: List[str],
    eval_domains: List[str],
    max_total_tokens: int,
    margin: int,
    prefer_long: bool,
    min_prompt_tokens: Optional[int],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Build dataset for two-phase experiment:
      - Phase 1: exactly one sample per distinct head-selection domain.
      - Phase 2: round-robin across eval domains for broad coverage, no overlap.
    """
    if len(head_selection_domains) < head_selection_num_samples:
        raise ValueError(
            f"Need >= {head_selection_num_samples} head_selection_domains, "
            f"got {len(head_selection_domains)}"
        )
    if len(set(head_selection_domains[:head_selection_num_samples])) != head_selection_num_samples:
        raise ValueError(
            "head_selection_domains must be distinct for head-selection samples: "
            f"{head_selection_domains[:head_selection_num_samples]}"
        )

    pool_size = max(head_selection_num_samples + eval_num_samples, 16)
    per_domain_picks: Dict[str, List[Dict[str, Any]]] = {}
    all_domains = sorted(set(head_selection_domains) | set(eval_domains))
    for domain in all_domains:
        if domain not in grouped:
            raise ValueError(
                f"Domain not found: {domain}. Available: {sorted(grouped.keys())}"
            )
        per_domain_picks[domain] = pick_domain_samples(
            grouped[domain],
            tokenizer,
            samples_per_domain=pool_size,
            max_total_tokens=max_total_tokens,
            margin=margin,
            prefer_long=prefer_long,
            min_prompt_tokens=min_prompt_tokens,
            rng=rng,
        )

    used_ids: set[str] = set()
    selected: List[Dict[str, Any]] = []

    head_domains = head_selection_domains[:head_selection_num_samples]
    for domain in head_domains:
        picks = per_domain_picks[domain]
        row, _ = _take_unique_row(picks, 0, used_ids)
        if row is None:
            raise ValueError(f"No unique head-selection sample available for domain={domain}")
        row["_selected_domain"] = domain
        row["_sample_role"] = "head_selection"
        used_ids.add(str(row.get("_id", "")))
        selected.append(row)

    eval_domain_idx = {d: 0 for d in eval_domains}
    eval_count = 0
    while eval_count < eval_num_samples:
        progressed = False
        for domain in eval_domains:
            if eval_count >= eval_num_samples:
                break
            picks = per_domain_picks[domain]
            row, next_idx = _take_unique_row(picks, eval_domain_idx[domain], used_ids)
            eval_domain_idx[domain] = next_idx
            if row is None:
                continue
            row["_selected_domain"] = domain
            row["_sample_role"] = "task_eval"
            used_ids.add(str(row.get("_id", "")))
            selected.append(row)
            eval_count += 1
            progressed = True
        if not progressed:
            raise ValueError(
                f"Could not collect {eval_num_samples} unique eval samples "
                f"across domains={eval_domains}"
            )

    return selected


def build_all_records_two_phase(
    grouped: Dict[str, List[Dict[str, Any]]],
    tokenizer,
    *,
    head_selection_num_samples: int,
    head_selection_domains: List[str],
    max_total_tokens: int,
    margin: int,
) -> List[Dict[str, Any]]:
    """
    Process every record in the source dataset:
      - truncate context only when prompt would exceed max_total_tokens
      - Phase 1: one head-selection sample per head_selection domain
      - Phase 2: all other records as task_eval
    """
    if len(head_selection_domains) < head_selection_num_samples:
        raise ValueError(
            f"Need >= {head_selection_num_samples} head_selection_domains, "
            f"got {len(head_selection_domains)}"
        )
    if len(set(head_selection_domains[:head_selection_num_samples])) != head_selection_num_samples:
        raise ValueError(
            "head_selection_domains must be distinct: "
            f"{head_selection_domains[:head_selection_num_samples]}"
        )

    all_processed: List[Dict[str, Any]] = []
    for domain in sorted(grouped.keys()):
        for row in grouped[domain]:
            all_processed.append(
                truncate_row_to_budget(
                    row,
                    tokenizer,
                    max_total_tokens=max_total_tokens,
                    margin=margin,
                )
            )
    all_processed.sort(key=lambda r: str(r.get("_id", "")))

    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_processed:
        by_domain[str(row.get("domain", "unknown"))].append(row)

    used_ids: set[str] = set()
    selected: List[Dict[str, Any]] = []
    head_domains = head_selection_domains[:head_selection_num_samples]
    for domain in head_domains:
        if domain not in by_domain:
            raise ValueError(
                f"Head-selection domain not found: {domain}. "
                f"Available: {sorted(by_domain.keys())}"
            )
        row = None
        for candidate in by_domain[domain]:
            rid = str(candidate.get("_id", ""))
            if rid and rid not in used_ids:
                row = dict(candidate)
                break
        if row is None:
            raise ValueError(f"No unique head-selection sample for domain={domain}")
        row["_selected_domain"] = domain
        row["_sample_role"] = "head_selection"
        used_ids.add(str(row.get("_id", "")))
        selected.append(row)

    for row in all_processed:
        rid = str(row.get("_id", ""))
        if rid in used_ids:
            continue
        out = dict(row)
        out["_selected_domain"] = str(out.get("domain", "unknown"))
        out["_sample_role"] = "task_eval"
        selected.append(out)

    return selected


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    source = resolve_source(args.source)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    print(f"Loading source: {source}")
    records: List[Dict[str, Any]] = json.loads(source.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} records")

    grouped = group_records_by_domain(records)
    print("Available domains:", sorted(grouped.keys()))

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    selected: List[Dict[str, Any]] = []
    selection_report: List[Dict[str, Any]] = []

    two_phase = args.head_selection_num_samples is not None and (
        args.eval_num_samples is not None or args.use_all_records
    )
    if two_phase:
        head_domains = (
            list(args.head_selection_domains)
            if args.head_selection_domains
            else DEFAULT_HEAD_SELECTION_DOMAINS
        )
        eval_domains = (
            list(args.eval_domains)
            if args.eval_domains
            else list(ALL_LONGBENCH_DOMAINS)
        )
        if args.use_all_records:
            print(
                f"Two-phase ALL records: head={args.head_selection_num_samples} "
                f"(domains={head_domains[: args.head_selection_num_samples]}), "
                f"eval=ALL remaining | max_total_tokens={args.max_total_tokens}"
            )
            selected = build_all_records_two_phase(
                grouped,
                tokenizer,
                head_selection_num_samples=args.head_selection_num_samples,
                head_selection_domains=head_domains,
                max_total_tokens=args.max_total_tokens,
                margin=args.margin,
            )
            args.eval_num_samples = len(selected) - args.head_selection_num_samples
            print(f"  -> total={len(selected)} (eval={args.eval_num_samples})")
        else:
            print(
                f"Two-phase sampling: head={args.head_selection_num_samples} "
                f"(domains={head_domains[: args.head_selection_num_samples]}), "
                f"eval={args.eval_num_samples} (domains={eval_domains})"
            )
            selected = build_two_phase_samples(
                grouped,
                tokenizer,
                head_selection_num_samples=args.head_selection_num_samples,
                eval_num_samples=args.eval_num_samples,
                head_selection_domains=head_domains,
                eval_domains=eval_domains,
                max_total_tokens=args.max_total_tokens,
                margin=args.margin,
                prefer_long=args.prefer_long,
                min_prompt_tokens=args.min_prompt_tokens,
                rng=rng,
            )
    elif args.total_samples is not None:
        per_domain_picks: Dict[str, List[Dict[str, Any]]] = {}
        for domain in args.domains:
            if domain not in grouped:
                raise ValueError(
                    f"Domain not found: {domain}. Available: {sorted(grouped.keys())}"
                )
            per_domain_picks[domain] = pick_domain_samples(
                grouped[domain],
                tokenizer,
                samples_per_domain=max(args.total_samples, 1),
                max_total_tokens=args.max_total_tokens,
                margin=args.margin,
                prefer_long=args.prefer_long,
                min_prompt_tokens=args.min_prompt_tokens,
                rng=rng,
            )

        domain_idx = {d: 0 for d in args.domains}
        round_robin_order = list(args.domains)
        while len(selected) < args.total_samples:
            progressed = False
            for domain in round_robin_order:
                if len(selected) >= args.total_samples:
                    break
                picks = per_domain_picks[domain]
                idx = domain_idx[domain]
                if idx >= len(picks):
                    continue
                row = dict(picks[idx])
                domain_idx[domain] += 1
                row["_selected_domain"] = domain
                selected.append(row)
                progressed = True
            if not progressed:
                raise ValueError(
                    f"Could not collect {args.total_samples} samples across domains={args.domains}"
                )
    else:
        for domain in args.domains:
            if domain not in grouped:
                raise ValueError(
                    f"Domain not found: {domain}. Available: {sorted(grouped.keys())}"
                )
            picks = pick_domain_samples(
                grouped[domain],
                tokenizer,
                samples_per_domain=args.samples_per_domain,
                max_total_tokens=args.max_total_tokens,
                margin=args.margin,
                prefer_long=args.prefer_long,
                min_prompt_tokens=args.min_prompt_tokens,
                rng=rng,
            )
            if len(picks) < args.samples_per_domain:
                raise ValueError(
                    f"Domain {domain}: only found {len(picks)} samples, "
                    f"need {args.samples_per_domain}"
                )
            selected.extend(picks)

    for row in selected:
        domain = str(row.get("domain", row.get("_selected_domain", "unknown")))
        selection_report.append(
            {
                "domain": domain,
                "sample_role": row.get("_sample_role"),
                "sub_domain": row.get("sub_domain"),
                "_id": row.get("_id"),
                "difficulty": row.get("difficulty"),
                "length": row.get("length"),
                "prompt_token_len": row["prompt_token_len"],
                "context_token_len": row["context_token_len"],
                "context_truncated": row["context_truncated"],
                "answer": row.get("answer"),
            }
        )
        role = row.get("_sample_role", "unknown")
        print(
            f"  [{role}] {domain} | id={row.get('_id')} | "
            f"prompt_tokens={row['prompt_token_len']} | "
            f"context_tokens={row['context_token_len']} | "
            f"truncated={row['context_truncated']}"
        )

    with out_path.open("w", encoding="utf-8") as fout:
        for row in selected:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter

    role_counts = Counter(r.get("sample_role") for r in selection_report)
    domain_counts = Counter(r.get("domain") for r in selection_report)
    head_domain_counts = Counter(
        r.get("domain") for r in selection_report if r.get("sample_role") == "head_selection"
    )
    eval_domain_counts = Counter(
        r.get("domain") for r in selection_report if r.get("sample_role") == "task_eval"
    )

    report_path = out_path.with_suffix(".selection.json")
    report_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "out": str(out_path),
                "max_total_tokens": args.max_total_tokens,
                "margin": args.margin,
                "two_phase": two_phase,
                "use_all_records": args.use_all_records,
                "head_selection_num_samples": args.head_selection_num_samples,
                "eval_num_samples": args.eval_num_samples,
                "head_selection_domains": (
                    list(args.head_selection_domains)
                    if args.head_selection_domains
                    else DEFAULT_HEAD_SELECTION_DOMAINS
                ),
                "eval_domains": (
                    list(args.eval_domains)
                    if args.eval_domains
                    else list(ALL_LONGBENCH_DOMAINS)
                ),
                "domains": args.domains,
                "samples_per_domain": args.samples_per_domain,
                "role_counts": dict(role_counts),
                "domain_counts": dict(domain_counts),
                "head_selection_domain_counts": dict(head_domain_counts),
                "task_eval_domain_counts": dict(eval_domain_counts),
                "selections": selection_report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {len(selected)} records to {out_path}")
    print(f"Selection report: {report_path}")


if __name__ == "__main__":
    main()
