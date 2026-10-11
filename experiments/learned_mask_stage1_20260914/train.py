"""Serial offline comparison on audited teacher records, not generation accuracy."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .data import load_record, validate_manifest, validate_group_provenance
from .model import (Config, Predictor, VARIANTS, final_mask, mapped_k,
                    selection_metrics, shared_masks, teacher_loss, validate_groups)


def cases(model, record):
    layer, start = record["layer"], record["start"]
    end = start + len(record["q"])
    for group_id, group in enumerate(model.groups[str(layer)]):
        # Dense teacher records are group-independent. Rebind to this fold's
        # representative rather than carrying a historical shared_p into CV.
        p_rep = record["teacher_p"][group["representative"]]
        protect, core, full = shared_masks(p_rep, start, end, model.config)
        condition = full if model.variant == "full_mask" else core
        fixed = protect if model.variant == "full_mask" else core
        for head in group["members"]:
            if head == group["representative"]:
                continue
            q = record["q"][:, head]
            k = mapped_k(record["k"], model.num_heads, head)
            yield head, q, k, p_rep, protect, core, full, condition, fixed


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, entries, device):
    model.eval()
    totals, rows, n = {}, [], 0
    loss_total, seconds = 0., 0.
    for entry in entries:
        record = load_record(entry, device)
        start, end = record["start"], record["start"] + len(record["q"])
        for head, q, k, p_rep, protect, core, full, condition, fixed in cases(model, record):
            teacher = record["teacher_p"][head]
            synchronize(device)
            begin = time.perf_counter()
            scores = model(q, k, p_rep, condition, record["layer"], head)
            prediction = final_mask(scores, core, full, protect, end, model.config, model.variant)
            synchronize(device)
            seconds += time.perf_counter() - begin
            oracle = final_mask(teacher, core, full, protect, end, model.config, model.variant)
            metric = selection_metrics(prediction, oracle, teacher, fixed, start, end)
            # Common final-budget oracle permits cross-variant comparison.
            per_head = final_mask(teacher, core, full, protect, end, model.config, "full_mask")
            metric["per_head_overlap"] = int((prediction & per_head).sum())
            metric["per_head_count"] = int(per_head.sum())
            metric["shared10240_mass"] = float(teacher[full].sum())
            metric["core_removed"] = int((core & ~prediction).sum())
            loss_total += float(teacher_loss(scores, teacher, fixed, end))
            n += 1
            for name, value in metric.items():
                totals[name] = totals.get(name, 0) + value
            rows.append(dict(input_ids_sha256=record["input_ids_sha256"], layer=record["layer"],
                             head=head, start=start, **metric))
    if not n:
        raise ValueError("No member-head cases")
    return dict(cases=n, mean_cross_entropy=loss_total/n, raw_counts=totals,
                oracle_recall=totals["overlap"]/totals["target_count"] if totals["target_count"] else None,
                oracle_mass_fraction=totals["recovered_mass"]/totals["oracle_mass"] if totals["oracle_mass"] else None,
                mean_final_mass=totals["final_mass"]/n,
                common_per_head_recall=totals["per_head_overlap"]/totals["per_head_count"],
                member_pair_sparsity=1-totals["selected_pairs"]/totals["causal_pairs"],
                selector_seconds_total=seconds,
                timing_scope="serial prototype forward and stable sort; no LLM or prefill timing; not optimized",
                rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters")
    manifest = validate_manifest(args.manifest)
    groups = validate_group_provenance(manifest, args.groups)
    validate_groups(groups)
    if {e["layer"] for e in manifest["splits"]["train"]} != set(map(int, groups)):
        raise ValueError("Every configured layer needs teacher records")
    first = load_record(manifest["splits"]["train"][0])
    config = Config(head_dim=first["q"].shape[-1])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(args=vars(args), config=asdict(config),
                      manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                      groups_sha256=hashlib.sha256(Path(args.groups).read_bytes()).hexdigest(),
                      purpose="offline mask prediction; not formal KV Retrieval generation")
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2))
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        torch.manual_seed(args.seed)
        order_rng = random.Random(args.seed)
        model = Predictor(groups, variant, config).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        folder = output / variant
        folder.mkdir()
        best, history = float("inf"), []
        entries = list(manifest["splits"]["train"])
        for epoch in range(args.epochs):
            model.train()
            order_rng.shuffle(entries)
            train_loss, updates = 0., 0
            for entry in entries:
                record = load_record(entry, args.device)
                end = record["start"] + len(record["q"])
                for head, q, k, p_rep, _, _, _, condition, fixed in cases(model, record):
                    if end <= config.budget:
                        continue
                    scores = model(q, k, p_rep, condition, record["layer"], head)
                    loss = teacher_loss(scores, record["teacher_p"][head], fixed, end)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.detach())
                    updates += 1
            if not updates:
                raise ValueError("No training tile beyond final budget")
            validation = evaluate(model, manifest["splits"]["validation"], args.device)
            value = validation["mean_cross_entropy"]
            if value < best:
                best = value
                torch.save(model.state_dict(), folder/"best.pt")
            history.append(dict(epoch=epoch, train_loss=train_loss/updates, validation_loss=value,
                                updates=updates))
            print(variant, history[-1], flush=True)
        model.load_state_dict(torch.load(folder/"best.pt", weights_only=True, map_location=args.device))
        cv = manifest.get("protocol") == "leave_one_input_out"
        result = evaluate(model, manifest["splits"]["validation" if cv else "test"], args.device)
        result.update(variant=variant, trainable_parameters=sum(p.numel() for p in model.parameters()),
                      network_count=len(model.networks), history=history,
                      evaluation_role="checkpoint_selection_validation" if cv else "independent_test",
                      independent_test=not cv)
        filename = "validation.json" if cv else "heldout.json"
        (folder/filename).write_text(json.dumps(result, indent=2))
    print("OFFLINE_COMPARISON_FINISHED; no generation accuracy measured", flush=True)


if __name__ == "__main__":
    main()
