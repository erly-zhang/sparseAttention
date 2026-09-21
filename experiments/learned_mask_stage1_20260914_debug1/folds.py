"""Build three audited folds from dense records and separately refitted AE groups."""
import argparse
import hashlib
import json
from pathlib import Path

from .data import validate_manifest, validate_group_provenance


def build_folds(records, group_files, reference, output):
    families = sorted({entry["family_id"] for entry in records})
    if len(families) != 3 or len(group_files) != 3:
        raise ValueError("Exactly three source inputs and three refitted group files required")
    identity = {}
    coverage = {}
    for family in families:
        subset = [entry for entry in records if entry["family_id"] == family]
        hashes = {entry["input_ids_sha256"] for entry in subset}
        if len(hashes) != 1:
            raise ValueError("Each source input must have one tokenized identity")
        identity[family] = next(iter(hashes))
        coverage[family] = {layer: sorted(e["start"] for e in subset if e["layer"] == layer)
                            for layer in {e["layer"] for e in subset}}
        if any(len(starts) != 8 or len(set(starts)) != 8 for starts in coverage[family].values()):
            raise ValueError("Each input/layer requires eight distinct tiles")
        if any(starts != next(iter(coverage[family].values())) for starts in coverage[family].values()):
            raise ValueError("Tile positions must agree across layers of an input")
    if len(set(identity.values())) != 3:
        raise ValueError("Duplicate tokenized input across families")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference = Path(reference).resolve()
    plans = []
    for family, groups in zip(families, group_files):
        train_ids = [value for value in families if value != family]
        groups = Path(groups).resolve()
        manifest = dict(schema="learned_mask_dense_teacher_v1", protocol="leave_one_input_out",
                        validation_family=family, group_fit_family_ids=train_ids,
                        groups_sha256=hashlib.sha256(groups.read_bytes()).hexdigest(),
                        protected_reference=str(reference),
                        protected_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                        splits={"train": [e for e in records if e["family_id"] in train_ids],
                                "validation": [e for e in records if e["family_id"] == family]})
        path = output / f"fold_{len(plans)}.json"
        path.write_text(json.dumps(manifest, indent=2))
        checked = validate_manifest(path)
        validate_group_provenance(checked, groups)
        plans.append(dict(manifest=str(path.resolve()), groups=str(groups), validation_family=family))
    (output / "folds.json").write_text(json.dumps(plans, indent=2))
    return plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="JSON list with absolute teacher paths")
    parser.add_argument("--groups", nargs=3, required=True,
                        help="Fold group files ordered by sorted held-out family ID")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_folds(json.loads(Path(args.records).read_text()), args.groups,
                                 args.reference, args.output), indent=2))


if __name__ == "__main__":
    main()
