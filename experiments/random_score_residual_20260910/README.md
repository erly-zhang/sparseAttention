# Independent Uniform-score TopK residual control

Method: `probmean_shared6144_randomscore4096_allmembers_seed42`.
Only the existing 497 Llama-3.1-8B-Instruct KV Retrieval reference inputs are
eligible. No alternative task, calibration, budget sweep or full seed sweep.

## Selection

The historical `probability_mean`, `shared_indices`, `assemble` and compact
attention kernel are imported without editing their files. Every representative
retains Shared10240; all 29 other heads use Shared6144 plus 4096 complement keys.
The representative's original TopK and tie policy remain unchanged. Sink/local
protection is inside the core budget. Prefixes up to 10240 keys are fully dense.

For each eligible original position `j`, a stateless Philox4x32-10 counter
generates a 24-bit primary uniform integer `p`. Its conceptual score is
`u=(p+0.5)/2**24`, a finite-precision uniform in (0,1). The counter `j+2**20`
provides an independent 22-bit tie integer `t`. Candidates are ordered by
descending `(p,t)` and, only when both tie, ascending original position.
The code packs this order into a unique, positive int64 priority:

```text
((p << 22 | t) << 17) | (131071-j)
```

This supports the required context limit 131072. It has 46 random priority bits;
the remaining deterministic fallback means exact finite-precision sampling is
not mathematically identical to continuous real-valued uniform priorities.
Its residual positional bias is confined to complete primary+secondary ties.
No random-score softmax or attention weighting occurs. Core/future positions
are masked to -1 **before** TopK. Each TopK spans the entire candidate key axis.
Only independent query tiles are batched (default 8); there are no per-key-chunk
quotas, repeated sampling or deduplication.

SHA256 of `random-score-residual-v1|seed42|input_ids_sha256|layer|head|tile`
supplies a 64-bit Philox key. The position supplies the counter. No Python hash
or global Torch/NumPy/Python random stream is used. Device index, shard index,
tile-batch size and execution order are absent from the seed. The old sampler
and new sampler do **not** produce the same mask for seed42.

Final indices are sorted in original-position order. RoPE and GQA remain in the
canonical model path; the original compact kernel applies a causal mask for
each real query and normalizes core/residual jointly. There is no member QK
scoring, even in the new profiling implementation.

## Environment

The earlier `sparseattention` environment has Torch2.4/Triton3.0; GPU execution
revealed that the unchanged historical probability-mean selector cannot compile
its `math.sqrt` constexpr call on Triton3.0. The independent
`sparseattention-score26` environment uses Torch2.6/CUDA12.4, Triton3.2 and
FlashAttention2.7.4.post1. `requirements.txt` records its primary dependencies.
FlexPrefill is loaded from the existing source checkout via `--flexprefill_root`;
its package metadata's old exact Triton3.0 pin is not installed into this env.
The original conda environment is unchanged.

## Artifacts required before smoke or formal inference

Copy `paths.example.json` to your own path configuration and restore:

- Exact 497-record filtered data, not just the public 500-record source.
- Original input reference metrics with `input_tokens` and `input_ids_sha256`.
- Existing AE K=3 head groups; no regrouping or recalibration.
- Llama checkpoint and matching tokenizer/chat template.
- Original three smoke records, task YAML, Python metric helpers and dataset loader.
- Old Shared6144+random4096 and Shared10240 per-input metrics (pair counts and timing).
- Their `summary.json` files alongside metrics when available (scores/hardware provenance).

Missing artifacts cause a preflight exit with no inference. The script only
rewrites the dataset loader and data-directory paths in a copied task YAML;
it preserves templates, function tags, metrics, filters and generation settings.

## Validation and execution

Use a Slurm-allocated, otherwise idle GPU. This scheduler runs one GPU serially;
it never selects unallocated devices, stops other jobs or overwrites old outputs.
Set `PYTHONPATH` to the repository root and `PYTHONNOUSERSITE=1`.

```bash
conda activate sparseattention-score26
cd /mmfs1/project/yl258/jz648/sparseAttention
export PYTHONPATH="$PWD"
export PYTHONNOUSERSITE=1
python -m experiments.random_score_residual_20260910.test_correctness \
  --output /ABS/NEW_OUTPUT/correctness.json
```

On a different allocated physical GPU, run the same command with
`--fingerprint-only --output /ABS/NEW_OUTPUT/cross_gpu.json`.
The fingerprint checks fixed-core production-budget masks for multiple heads.
GPU UUIDs must differ, the Torch versions and code hashes must match, and the
fingerprints must be identical. Small-tensor tests also compare against an
independent NumPy Philox implementation and independent FP32 masked attention,
including contiguous/interleaved GQA, old-core equality, exact pairs, protected
positions, partial/padded tiles and global-RNG preservation. A 4096-draw small
candidate-set frequency check uses a predefined 6-sigma marginal bound; these
are unit-test draws, not additional full model seeds.

```bash
python -m experiments.random_score_residual_20260910.schedule \
  --config /ABS/paths.json --output-root /ABS/NEW_OUTPUT --preflight-only
python -m experiments.random_score_residual_20260910.schedule \
  --config /ABS/paths.json --output-root /ABS/NEW_OUTPUT \
  --correctness /ABS/NEW_OUTPUT/correctness.json \
  --cross-gpu /ABS/NEW_OUTPUT/cross_gpu.json
```

The order is correctness gate -> three full smoke generations -> four disjoint
formal shards covering 497 -> separate three-input profiling. All use BF16,
batch1, greedy, seed42, chat, context131072 and max_new_tokens128 via the canonical
runner. Input identity is checked against the expected shard **before** model
generation; final validation checks uniqueness/full coverage, official scoring
and per-input/total pair equality against both historical methods. Smoke is
also subject to finite-logit, protected-index and exact-budget checks.
Source hashes, immutable source snapshots and a pip freeze are saved. Completed
attempts are reused only after validation; failed attempts are retained.

## Timing and reporting

Formal prefill is synchronized wall-clock around the first model forward,
including representative scoring, stream derivation, random priorities, TopK,
index operations, required pair-count kernels and attention. Model loading,
synthetic warmup and additional profiling serialization are excluded. The
historical CUDA-event measurement is retained under a separate field.
Decode/total latency, peak allocated GPU memory, selected/causal raw pair counts,
pooled pair sparsity and full generation scores are retained per input.

Profiling saves raw per-layer/head/tile core/residual/final/novel counts, value
and KV evidence slots/pairs and separate phase GPU-event durations. These stream
durations can include host submission gaps. Novelty is versus Shared10240 on
the new trajectory's Q/K, not a claim of identical hidden states across methods.
Coverage is pooled retained/eligible counts, not whole-record retention or
causal usefulness. No profiling-only scores replace the formal 497 accuracy.

`comparison.json` and `RESULTS.md` are produced only after successful execution.
The main historical comparison is 8/497, 36.38s prefill; Shared10240 is secondary.
The report records historical score provenance and does not automatically
claim speedup across different hardware or timing instrumentation. A one-seed
accuracy difference between two uniform samplers is a different-mask result,
not proof that the sampler has improved retrieval quality.
