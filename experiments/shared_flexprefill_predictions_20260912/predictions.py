"""CPU-only prediction export and fail-closed score verification."""
import json
import math
import ast
import hashlib
from pathlib import Path

KEY = "get_score_one_kv_retrieval"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def export(folder, data_path, scorer, count, filename="predictions.jsonl"):
    import numpy as np
    from transformers import AutoTokenizer
    folder = Path(folder)
    result = json.loads((folder / "lm_eval_results.json").read_text())
    samples = result.get("samples", {}).get("kv_retrieval", [])
    metrics = read_rows(folder / "online_metrics.jsonl")
    data = read_rows(data_path)
    assert len(samples) == len(metrics) == count
    assert [s["doc_id"] for s in samples] == list(range(count))
    identities = [(r["input_tokens"], r["input_ids_sha256"]) for r in metrics]
    assert len(set(identities)) == count
    by_identity = dict(zip(identities, metrics))
    tokenizer = AutoTokenizer.from_pretrained(
        "/mmfs1/scratch/yl258/jz648/Llama-3.1-8B-Instruct", local_files_only=True)
    records = []
    for sample, source in zip(samples, data):
        arguments = sample["arguments"][0]
        arguments = ast.literal_eval(arguments) if isinstance(arguments, str) else arguments
        ids = tokenizer(arguments[0], add_special_tokens=False)["input_ids"][-130944:]
        identity = (len(ids), hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest())
        metric = by_identity.pop(identity)
        doc = sample["doc"]
        assert all(doc[k] == source[k] for k in ("input", "context", "answer"))
        target = "===answerspliter===".join(doc["answer"])
        assert sample["target"] == target
        assert len(sample["filtered_resps"]) == len(sample["resps"]) == 1
        prediction = sample["filtered_resps"][0]
        assert isinstance(prediction, str) and len(sample["resps"][0]) == 1
        score = scorer((target, prediction))
        assert score in (0, 1) and score == sample[KEY]
        records.append(dict(doc_id=sample["doc_id"], target=target,
                            raw_prediction=sample["resps"][0][0],
                            filtered_prediction=prediction, score=score,
                            input_tokens=metric["input_tokens"],
                            input_ids_sha256=metric["input_ids_sha256"],
                            doc_hash=sample["doc_hash"], prompt_hash=sample["prompt_hash"],
                            generated_tokens=metric["generated_tokens"]))
    accuracy = sum(r["score"] for r in records) / count
    assert math.isclose(accuracy, result["results"]["kv_retrieval"][KEY + ",none"], abs_tol=1e-12)
    assert not by_identity
    order = {identity: i for i, identity in enumerate(identities)}
    records.sort(key=lambda r: order[(r["input_tokens"], r["input_ids_sha256"])])
    with (folder / filename).open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    assert read_rows(folder / filename) == records
    audit = dict(status="passed", count=count, correct=sum(r["score"] for r in records),
                 accuracy=accuracy, scoring="official KV scorer reapplied to saved filtered responses")
    audit["identity_join"] = "independent prompt tokenization and input SHA256, not positional zip"
    with (folder / filename.replace(".jsonl", "_audit.json")).open("x") as stream:
        json.dump(audit, stream, indent=2)
    return audit
