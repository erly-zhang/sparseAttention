# Head-slot x KV-support ablation

User approved: KV Retrieval 50 + InfiniteBench Code.Debug 50, each with
dense and four ablations (500 generations, including 30 smoke generations).
One independent PSC H100 80GB, 8 CPU, 128GB, maximum six hours.
User explicitly permits this experiment to be submitted from compute node gl002.
Never run the GPU evaluation on gl002; never resubmit or expand this budget automatically.

| KV support | Pair output slots | Original head slot, other slots zero |
| --- | --- | --- |
| Same position-modulo owner | group_pair | group_native |
| All legal causal history | all_pair | all_native |

These are O_h=A_h V_h output assemblies, not concatenated attention weights.
Owner = original position modulo 32. Pair uses the frozen sorted undirected
cycle edge slots, exactly two copies per source. Native uses only the owner's
original slot. No scaling, original full o_proj, original RoPE/GQA, full cache.
Dense layers: 2,5,8,11,14,17,20,23,26,29,31; remaining 21 layers routed.
All decode steps are dense. No output calibration MLP is trained or installed.

Model: Llama-3.1-8B-Instruct BF16, seed42, greedy128. Frozen token IDs, same
head-tail truncation at 130944, per-task official FlexPrefill-vendored scoring.
Code.Debug uses the official yarn_mistral task text with Llama's chat wrapper.
This is a matched local diagnostic, not the full official benchmark or its
task-specific generation caps. Source, dataset, tokenizer and input hashes freeze
before submission. Never inspect evaluation predictions to tune the design.

Sources:
- https://github.com/OpenBMB/InfiniteBench/blob/main/src/prompt.py
- https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/tree/90f0394333616266d9fe85824ceaf505093cbaa5
- Frozen prior routing.py, quality.py and official scoring provenance under PSC.

Limitations:
- Native versus pair changes both slot identity and duplication/amplitude.
- Full KV support increases attention pairs. q_kv runs FA2, q_only uses explicit
  masked SDPA. Record wall times but do not call them a matched-kernel speedup.
- Saved logits measure final-prompt vocabulary KL(dense || method), not layerwise KL.
- Three KV inputs have prior dense/pair generated-ID regression references.
  Code selection is deterministic, excluding indices 0..3; no tuning on outputs.
- Full caches do not imply that every cached position contributed to prefill attention.

Gates:
1. Isolated path, all frozen-source checks, six inherited plus seven new CPU tests.
2. Actual 100-input freeze; test-only of this exact sbatch script.
3. Submit once via submit.py, record actual ID. Intent blocks ambiguous retries.
4. 30 smoke records: all cells, both tasks, six historical token regressions,
   full caches, finite logits at every generation step, analytical attention counts.
5. Reserve 30 minutes and apply conservative 1.5x projected remaining runtime;
   stop with preserved smoke and failure if it cannot fit the six-hour allocation.
6. Require COMPLETED/0:0, 500 unique aligned outputs (50/task/method), official
   rescore and independent KL from saved logits before calling the run complete.

deploy.sh prepares/validates only; it does not submit. Preserve failures and all
earlier experiments. The local default Python currently lacks torch; perform
tensor tests with the existing pinned PSC environment, GPU visibility disabled.

Preparation revision 2: the first attempt stopped before any GPU submission
because raw Code.Debug labels have one function name, not an already expanded
function/letter pair. Normalize labels using the official get_answer semantics
(https://github.com/OpenBMB/InfiniteBench/blob/main/src/eval_utils.py), deriving
the letter from the options. Prompts, selection and methods are unchanged.
The first attempt's entire remote directory and archive are preserved.
