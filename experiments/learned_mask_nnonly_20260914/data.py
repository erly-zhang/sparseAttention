"""Explicit teacher-record schema and input-family-level split checks."""
import hashlib
import json
from pathlib import Path

import torch

from .model import exact_probability_mean, mapped_k, validate_groups


@torch.no_grad()
def export_record(path, *, q_tile, k, groups, layer, start, input_tokens,
                  input_ids_sha256, family_id):
    """Called on captured post-RoPE Q/K; this does not run/capture the LLM.

    q_tile: [real_queries, Q_heads, D]; k: [input_tokens, KV_heads, D].
    Full legal-key softmax is computed offline, never in the learned scorer.
    """
    heads = validate_groups(groups)
    if q_tile.ndim != 3 or q_tile.shape[1] != heads or len(k) != input_tokens:
        raise ValueError("Incorrect captured Q/K layout")
    if len(input_ids_sha256) != 64 or not family_id:
        raise ValueError("An audited input hash and source family are required")
    end = start + len(q_tile)
    if not 0 < len(q_tile) <= 128 or end > input_tokens:
        raise ValueError("Invalid real query tile")
    # Later keys have zero teacher probability and cannot be selected.
    visible_k = k[:end]
    teacher = torch.stack([exact_probability_mean(q_tile[:, h], mapped_k(visible_k, heads, h), start)
                           for h in range(heads)])
    representatives = [g["representative"] for g in groups[str(layer)]]
    record = dict(q=q_tile.detach().cpu(), k=visible_k.detach().cpu(),
                  teacher_p=teacher.cpu(), shared_p=teacher[representatives].cpu(),
                  layer=layer, start=start, input_tokens=input_tokens,
                  input_ids_sha256=input_ids_sha256, family_id=family_id,
                  teacher_state="dense_post_rope", padding_queries=False)
    with Path(path).open("xb") as stream:
        torch.save(record, stream)
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                layer=layer, start=start, input_tokens=input_tokens,
                input_ids_sha256=input_ids_sha256, family_id=family_id)


def validate_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "learned_mask_dense_teacher_v1":
        raise ValueError("Unknown manifest schema")
    protocol = manifest.get("protocol", "independent_test")
    cv = protocol in {"leave_one_input_out", "leave_one_input_out_fixed_groups"}
    expected = ({"train", "validation"} if cv
                else {"train", "validation", "test"})
    if protocol not in {"independent_test", "leave_one_input_out", "leave_one_input_out_fixed_groups"}:
        raise ValueError("Unknown validation protocol")
    if set(manifest["splits"]) != expected:
        raise ValueError("Split names do not match validation protocol")
    protected = Path(manifest["protected_reference"])
    if not protected.is_absolute():
        protected = path.parent / protected
    if hashlib.sha256(protected.read_bytes()).hexdigest() != manifest["protected_reference_sha256"]:
        raise ValueError("Protected reference checksum changed")
    reserved = {json.loads(l)["input_ids_sha256"] for l in protected.read_text().splitlines() if l.strip()}
    seen_identity, seen_family = {}, {}
    seen_records = set()
    layers = {}
    for split, entries in manifest["splits"].items():
        if not entries:
            raise ValueError(f"Empty split: {split}")
        layers[split] = {entry["layer"] for entry in entries}
        for entry in entries:
            identity, family = entry["input_ids_sha256"], entry["family_id"]
            if identity in reserved:
                raise ValueError("Formal reference input included in prototype data")
            if not family or seen_identity.get(identity, split) != split or seen_family.get(family, split) != split:
                raise ValueError("Input or document family leaks across splits")
            seen_identity[identity], seen_family[family] = split, split
            key = (identity, entry["layer"], entry["start"])
            if key in seen_records:
                raise ValueError("Duplicate input/layer/tile record")
            seen_records.add(key)
            source = Path(entry["path"])
            if not source.is_absolute():
                source = path.parent / source
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Teacher record checksum mismatch")
            entry["path"] = str(source)
    if any(value != layers["train"] for value in layers.values()):
        raise ValueError("Layer coverage differs across splits")
    if cv:
        families = {name: {e["family_id"] for e in entries}
                    for name, entries in manifest["splits"].items()}
        identities = {name: {e["input_ids_sha256"] for e in entries}
                      for name, entries in manifest["splits"].items()}
        if [len(families[s]) for s in ("train", "validation")] != [2, 1]:
            raise ValueError("A fold requires two training inputs and one validation input")
        if any(len(identities[s]) != len(families[s]) for s in families):
            raise ValueError("Expected exactly one tokenized identity per source input")
        if protocol == "leave_one_input_out" and set(manifest["group_fit_family_ids"]) != families["train"]:
            raise ValueError("Fold groups must be fitted on training families only")
        if protocol == "leave_one_input_out_fixed_groups":
            if manifest.get("grouping_scope") != "historical_fixed_all_calibration":
                raise ValueError("Fixed historical grouping must be explicitly declared")
            if manifest.get("grouping_independent_of_validation") is not False:
                raise ValueError("Cannot claim independent grouping for fixed historical AE")
    return manifest


def validate_group_provenance(manifest, groups_path):
    """Bind fold groups to the AE fit inputs, not just the NN split labels."""
    payload = json.loads(Path(groups_path).read_text())
    if manifest.get("protocol") == "leave_one_input_out":
        if hashlib.sha256(Path(groups_path).read_bytes()).hexdigest() != manifest["groups_sha256"]:
            raise ValueError("Fold group checksum mismatch")
        actual = payload.get("calibration_sample_ids", [])
        if len(actual) != 2 or set(actual) != set(manifest["group_fit_family_ids"]):
            raise ValueError("AE groups include a held-out input or lack fit provenance")
    elif manifest.get("protocol") == "leave_one_input_out_fixed_groups":
        if hashlib.sha256(Path(groups_path).read_bytes()).hexdigest() != manifest["groups_sha256"]:
            raise ValueError("Historical group checksum mismatch")
        if manifest.get("grouping_independent_of_validation") is not False:
            raise ValueError("Historical grouping validation limitation not recorded")
    return payload["layers"]


def load_record(entry, device="cpu"):
    record = torch.load(entry["path"], weights_only=True, map_location="cpu")
    for field in ("layer", "start", "input_tokens", "input_ids_sha256", "family_id"):
        if record[field] != entry[field]:
            raise ValueError(f"Record identity mismatch: {field}")
    if record["teacher_state"] != "dense_post_rope" or record["padding_queries"]:
        raise ValueError("Unexpected teacher convention")
    q, k, p = record["q"], record["k"], record["teacher_p"]
    end = record["start"] + len(q)
    if q.ndim != 3 or k.ndim != 3 or len(k) != end or end > record["input_tokens"]:
        raise ValueError("Bad record tensor shape or causal range")
    if not 0 < len(q) <= 128 or q.shape[-1] != k.shape[-1] or p.shape != (q.shape[1], end):
        raise ValueError("Bad query tile or teacher shape")
    if not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid teacher probabilities")
    torch.testing.assert_close(p.sum(-1), torch.ones(len(p)), atol=1e-5, rtol=1e-5)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in record.items()}
