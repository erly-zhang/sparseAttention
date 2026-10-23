#!/usr/bin/env python3
"""vLLM accuracy-transfer check on the frozen multibench inputs (MATH500 by default).

Prompts are rebuilt with the frozen harness of the HF multibench root and must match its
input-id SHA-256 references exactly; sampling = temperature 0.6, top_p 0.95, no top_k,
max 32768 new tokens (the HF protocol). Grading = frozen Utils.parser.extract_answer +
Utils.grader.check_is_correct(timeout=False), 30 s process bound, 120 s retry.
The arm is selected by the launcher's environment only.
"""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

ROOT = Path("/mmfs1/project/yl258/jz648/sparseAttention/experiments/reasoning_qhead_shared_topp_20260918/results/"
            "multibench_v6_randompp_dense_20261016")
MATH_TASKS = ["aime24", "aime25", "aime26", "hmmt", "math", "olympiadbench"]


def rows(p):
    return [json.loads(l) for l in open(p)]


def build_prompts(task, model):
    sys.path[:0] = [str(ROOT/"source/ra_harness/kvcompress/harness"), str(ROOT/"source/ra_harness")]
    spec = importlib.util.spec_from_file_location("frozen_eval_hf", ROOT/"source/ra_harness/kvcompress/harness/eval_hf.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    refs = {r["idx"]: r for r in rows(ROOT/"inputs"/f"{task}.jsonl")}
    out = []
    for ex in m.load_data(task, "test", str(ROOT/"data")):
        q = m.parse_question(ex, task)
        _, _, fmt = m.get_three_prompt("qwen-instruct", task)
        prompt = fmt.format(question=q)
        if task in MATH_TASKS:
            prompt += "\nPlease reason step by step, and put your final answer within \\boxed{}."
        ids = m.get_conversation_prompt_by_messages(tokenizer=tok, messages=[{"role": "user", "content": prompt}])["input_ids"][0]
        r = refs[int(ex["idx"])]
        digest = hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()
        assert ids.shape[0] == r["input_tokens"] and digest == r["input_ids_sha256"], (task, ex["idx"])
        out.append((int(ex["idx"]), ids.tolist(), r["gold"]))
    return out


def _math_child(conn, text, gold):
    try:
        from Utils.parser import extract_answer
        from Utils.grader import check_is_correct
        pred = extract_answer(text)
        conn.send(dict(prediction=pred, correct=bool(check_is_correct(pred, gold, timeout=False))))
    except BaseException as e:
        conn.send(dict(error=repr(e)))
    finally:
        conn.close()


def grade_math(text, gold):
    for limit in (30, 120):
        ctx = multiprocessing.get_context("fork")
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_math_child, args=(child, text, gold))
        proc.start()
        child.close()
        try:
            if parent.poll(limit):
                result = parent.recv()
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result["prediction"], int(result["correct"]), False
        finally:
            parent.close()
            proc.join(1)
            if proc.is_alive():
                proc.kill()
                proc.join()
    return None, 0, True


def prep(task, model, out):
    with open(out, "w") as f:
        for idx, ids, gold in build_prompts(task, model):
            f.write(json.dumps(dict(idx=idx, input_ids=ids, gold=gold))+"\n")
    print("PREP", task, out, flush=True)


def grade(out_dir, runs):
    sys.path[:0] = [str(ROOT/"source/ra_harness/kvcompress/harness"), str(ROOT/"source/ra_harness")]
    out_dir = Path(out_dir)
    summary = json.load(open(out_dir/"generation.json"))
    for run in range(runs):
        res = rows(out_dir/f"run_{run}.jsonl")
        graded = []
        for r in res:
            pred, ok, flagged = grade_math(r["text"], r["gold"])
            graded.append(dict(r, prediction=pred, correct=ok, grader_flagged=flagged))
        with open(out_dir/f"graded_{run}.jsonl", "w") as f:
            for r in graded:
                f.write(json.dumps(r)+"\n")
        n = len(graded)
        summary["runs"][f"accuracy_{run}"] = sum(r["correct"] for r in graded)/n
        summary["runs"][f"mean_output_tokens_{run}"] = sum(r["output_tokens"] for r in graded)/n
        summary["runs"][f"length_limited_{run}"] = sum(r["finish"] == "length" for r in graded)
        summary["runs"][f"grader_flagged_{run}"] = sum(r["grader_flagged"] for r in graded)
    accs = [summary["runs"][f"accuracy_{r}"] for r in range(runs)]
    summary["accuracy"] = sum(accs)/len(accs)
    with open(out_dir/"summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prep", "generate", "grade"], default="generate")
    ap.add_argument("--ids", default="")
    ap.add_argument("--model", default="/scratch/yl258/jz648/Qwen3-4B")
    ap.add_argument("--task", default="math")
    ap.add_argument("--arm", default="")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "prep":
        return prep(args.task, args.model, args.out)
    if args.mode == "grade":
        return grade(args.out, args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = [(r["idx"], r["input_ids"], r["gold"]) for r in rows(args.ids)]
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    max_p = max(len(p[1]) for p in prompts)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=max_p+args.max_tokens+64, enforce_eager=True,
              enable_prefix_caching=False, max_num_seqs=args.max_num_seqs, disable_log_stats=False)
    summary = dict(arm=args.arm, task=args.task, max_num_seqs=args.max_num_seqs,
                   env={k: v for k, v in os.environ.items() if k.startswith(("TRIATTN", "HYBRID", "ENABLE_TRI"))},
                   runs={})
    for run in range(args.runs):
        fp = out/f"run_{run}.jsonl"
        if fp.exists() and len(rows(fp)) == len(prompts):
            continue
        sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=-1, max_tokens=args.max_tokens, seed=20261020+run)
        t0 = time.perf_counter()
        outs = llm.generate([TokensPrompt(prompt_token_ids=p[1]) for p in prompts], sp)
        summary["runs"][f"elapsed_s_{run}"] = time.perf_counter()-t0
        with open(fp, "w") as f:
            for (idx, _, gold), o in zip(prompts, outs):
                f.write(json.dumps(dict(idx=idx, gold=gold, output_tokens=len(o.outputs[0].token_ids),
                                        finish=o.outputs[0].finish_reason, text=o.outputs[0].text))+"\n")
        print("GENERATED", run, summary["runs"], flush=True)
    with open(out/"generation.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
