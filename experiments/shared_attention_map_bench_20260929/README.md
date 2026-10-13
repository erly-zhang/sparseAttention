# Shared Attention Map: Four-Task Evaluation

## Authorized scope

Four independent H100 80GB workers, each 8 CPU, 128 GB host memory,
at most 24 hours. One array submission, elements 0-3, concurrency 4.
50 samples each from KV Retrieval, En.MC, En.QA, Code.Debug. Dense and
shared-map generation for every sample: 400 generations. No training.
No automatic retry, resource increase, cancellation, or unrelated job changes.

## Exact algorithm

Llama 3.1 8B BF16, 32 query heads, 8 KV heads. At even layers 0..30,
query row i belongs to head i modulo 32. All heads compute all causal query
rows against the complete key sequence. The owner's normalized row is
selected into shared A. Every output head uses this A with its original
GQA value head, then the original concatenation and o_proj.
Neither averaging head maps nor copying one O across all heads is used.
All KV entries remain cached. Only prefill changes; decode is original dense.

Query-row blocks of 256 preserve the complete softmax key dimension and
the exact causal map semantics without allocating a complete long NxN map.
The explicit literal NxN implementation is retained for small parity tests.
Dense uses matched explicit FP32 attention arithmetic at the same 16 layers;
other layers and decode use original FA2. This is not a speed benchmark or
a claim of bitwise equivalence to an all-FA2 baseline.

## Fair input contract

Both methods read identical frozen input IDs, original positions, masks,
chat-rendered prompts, and answer identities. Official public dataset revision
90f0394333616266d9fe85824ceaf505093cbaa5 is pinned and checked against LFS hashes.
Official prompt/scorer/get_answer source is pinned to the downloaded commit
in inputs/manifest.json. Official task token caps: KV50, En.MC40, En.QA40,
Code.Debug5. Greedy seed42 and identical EOS handling per pair.

Selection is seeded, without correctness filtering. Only complete examples
that fit the model's context limit including generation are eligible; no
truncation. Report this length-eligible subset limitation. En.QA uses official
token F1, while the other selected task scorers return task-specific accuracy.
Pairs stay on the same GPU; 50 pairs per worker are balanced using input N^2.

## Gates and outputs

1. CPU preparation freezes 200 real examples exclusively, no overwrite.
2. Verified sources, 15 CPU tests and full input retokenization before submission.
3. Actual run.sbatch passes sbatch --test-only; one submit-intent and actual job ID.
4. Worker small GPU blocked/literal parity, then two longest paired samples.
5. Finite logits, full caches and analytical pair accounting must pass. Remaining
   runtime estimate uses observed cost with 1.5 multiplier plus 1800 seconds
   reserve, total below 84600 seconds. Projection checked after every later pair.
6. Four workers each write 100 results plus immutable logits and complete marker.
7. Slurm COMPLETED/0:0 for all elements and independent-audit.json PASS400 required.

Execution success is not quality recovery. Report scores, actual predictions,
repetition and caps. KL is final-prompt vocabulary KL, not layerwise KL.

Remote access only through the authenticated Safari VS Code Tunnel. User handles
authentication and trust. v020 is submission/development only, not evaluation.
