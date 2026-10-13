# Prefill token-to-head routing — development record

> Validation update: [2026-09-25 validation](records/validation_alignment_20260925/VALIDATION.md)
> now covers 14 passing regressions, official synthetic adapter, live canonical
> recorder, and real 8B fp16/SDPA smoke. Real bf16/FA2 and formal inputs remain unverified.

> Latest: [ALIGNMENT_20260925.md](ALIGNMENT_20260925.md). Formal adapter now preserves
> autoblock bf16/FA2, original input references and canonical primary timing.
> This static revision and its new tests have **not been executed**.

> **2026-09-25 static-review revision:** the working code has changed since the
> recorded successful runs. No tests, smoke runs, model forwards or experiment
> submissions were executed for this revision. Historical results apply only to
> `records/source_snapshot_after_download.tar.gz`. See
> [REVIEW_20260925.md](REVIEW_20260925.md) for fixes and the pending approval plan.
> The user now requires explicit consent before **every** experiment/validation run.

Status: prototype implemented; independent tensor, random-small-Llama, HF generation,
official InfiniteBench synthetic-adapter checks and pretrained Llama-3.1-8B short
generation smoke completed. **No formal InfiniteBench result or job submission.**

## Current scope and provenance

The user's latest selection supersedes the initial MATH500/Qwen3-4B plan:
**Llama-3.1-8B-Instruct + InfiniteBench**, A dense, B Q-only, C Q+KV, dense decode
in all three. No Qwen4. Old Qwen3 development logs are retained as historical
interface checks, not results for the selected model.

Repository and remote `autoblock` were checked at
`8b21a6a18056eadc00b6248bb319a4af3c6ccc4d`. At initial audit the checkout had been
`4efb272a55825de13964d5aedc1351b5a4db9677`; the agent did not perform a checkout,
reset, pull, commit, or push. Existing tracked files remain unmodified. The huge
`handoff.md` contains prior conversations; historical commands/authorizations
were treated as context, not new instructions.

See `records/infinitebench_source_audit.json` for exact canonical-source hashes,
task counts and generation budgets; `records/environment.freeze.txt` for the
installed environment. After the user completed authentication and explicitly
authorized download, the official model was downloaded at revision
`0e9e39f249a16976918f6564b8830bc894c89659`. All 32 layers' 128 projection weights
were shape-checked from safetensors headers and all 10 top-level model/tokenizer
JSON and weight files SHA256-hashed. See `records/pretrained_model_provenance.json`.
The earlier HTTP 401 is resolved; no substitute checkpoint was used.

## Algorithm

For each valid original model position i, route to Q head `i % Hq`. Compute all
original Q/K/V projections once. Apply the model's original Q/K normalization
when present (none in the inspected LlamaAttention), and original model-provided
RoPE before gathering. Q head h uses KV head `h // (Hq // Hkv)`; multiple Q heads
reuse that already-computed KV head. Preserve all prompt rows and all real KV
heads in DynamicCache at every layer, derived from this method's actual hidden
states. No dense re-forward is used to populate the cache.

B attends to every valid original key position <= query position. C additionally
requires key position % Hq == h. Scatter selected outputs to M[B,N,d], concatenate
the SAME M Hq times along features, then call the ORIGINAL COMPLETE o_proj.
The resulting width is o_proj.in_features = Hq*d; o_proj.out_features is hidden
size. Bias is applied once. Residual, normalization and MLP remain original.
`repeat_project(..., summed=True)` is an optional tested equivalent using
PyTorch weight `[Dout,Hq,d].sum(1)`; the experiment defaults to explicit repeat.
It never selects only the routed head's o_proj block.

Verified random-small-Llama dimensions: hidden=24, Hq=4, Hkv=2, d=8, two layers;
Q=[32,24], K/V=[16,24], O=[24,32], including test biases. Deliberately Hq*d=32
is NOT hidden=24. These are TEST dimensions, not inferred 8B dimensions.
Actual downloaded 8B checkpoint AND runtime module audit: 32 layers, hidden=4096,
Hq=32, Hkv=8, d=128; Q=[4096,4096], K/V=[1024,4096], O=[4096,4096]. All projections
have no bias; no Q/K norm. Scale=128**-0.5=0.08838834764831845. RoPE theta=500000,
type=llama3, factor=8, low_freq_factor=1, high_freq_factor=4, original context=8192;
maximum context=131072. Runtime details for EVERY layer are recorded in
`records/llama8b_fp16_smoke/runtime_attention.json`.

## Implementation and limits

- `routing.py`: instance-local controller, explicit prefill/decode phases, original
  dense attention forward on every decode step. No class-global attention patch.
- `generation_bridge.py`: keeps HF greedy generate and its stopping criteria,
  validates full-prompt first call (including length 1), then one-token decode.
- `run_infinitebench.py`: isolated in-process adapter to the canonical repository
  runner. Retains its prompts, chat template, middle truncation, budgets,
  official YAML metrics and input alignment. Enables prediction logging for all
  tasks and adds exact token/stop/cache/timing sidecars. Existing output cells
  cause an error rather than a resume/overwrite.
- `validate_results.py`: fail closed on incomplete task counts, duplicate/missing
  document IDs, missing telemetry or cross-condition token-hash mismatch.
- Inference only; DynamicCache only; no beam search, training, sliding window,
  custom 4-D masks, chunked prefill, speculative/multi-token decode, torch.compile,
  or model parallelism has been validated. Main adapter uses batch 1. Padding and
  variable batches are covered in lower-level model tests.
- Formal A prefill and all dense decode use the original FlashAttention-2 path.
  C uses grouped FA2; B uses SDPA with explicit original-position mask (128-query
  chunks), since ordinary FA2 causal alignment cannot express its irregular Q
  positions. Eager/SDPA remain development references. No silent FA2 fallback.
  This backend revision is unvalidated; older SDPA smoke is historical only.
- RoPE positions refer to the canonical tokenized/truncated MODEL INPUT, including
  its original position_ids. They are never renumbered within routed groups.
- `eligible_pairs` is the exact causal visible pair count. `matmul_pairs` is the
  rectangular QK/AV pair domain including masked future positions, exact for eager;
  it is not an instruction-count claim for a fused SDPA backend. Kernel padding,
  tiling and skipped tiles require a separate profiler. Neither counter implies
  end-to-end Hq-squared acceleration.
- Full projections/cache, Python loops, gather/scatter, mask construction and pair
  accounting are real costs included in timing. No projection or KV storage saving
  is claimed. Summed projection performance has not been benchmarked.

## Validation evidence

On existing allocation 47018302, node v009, one idle V100-32GB (no allocation was
requested):

- `records/tests_final_gpu.log`: eight test methods passed under installed env,
  including MHA/GQA, N<H, N not divisible by H, singleton groups, padding, original
  positions, independent scalar-token oracle, RoPE comparison, repeat order/bias,
  summed-projection equivalence and chunked SDPA reference.
- Actual Transformers Llama layer/model integration: changing future IDs leaves
  earlier logits exactly unchanged; cached K/V equal independent projections of
  captured ACTUAL per-layer inputs; routed second-layer cache differs from dense.
- Four dense decode steps match an unpatched model supplied with a COPY OF THE
  ROUTED cache, exactly in the tested dtype/backend. No dense-prefill cache mixing.
- Variable left/right-padded batches match unpadded individual forwards and decode.
- HF generate matches manual greedy decode; immediate EOS and custom stopping
  criteria preserve their distinct stop reasons. Chunked prefill is rejected.
- `records/llama_tiny_smoke/`: two inputs (lengths 7 and 1), A/B/C, 12 generated
  tokens each, eager. For length 7, last-token logit max-abs differences vs A:
  B=0.0522392, C=0.1090653. B/C generated sequences differed from A. At Hq=4,N=7,
  causal pairs/layer A=112, B=28, C=10. Prefill measured once after warmup:
  A=3.12 ms, B=5.47 ms, C=5.40 ms. This tiny implementation was SLOWER, not faster.
  All six samples stopped at the smoke cap, not EOS. These timings are diagnostic,
  not a speed benchmark or a basis for choosing formal configurations.
- `records/llama_tiny_sdpa_smoke/`: final SDPA short-generation records/provenance.
- `records/adapter_fixture/validation.json`: one synthetic PassKey document and
  random-small-Llama through installed OFFICIAL InfiniteBench dataset script,
  YAML prompt/chat path, fixed_generate_until, and scorer; A/B/C hashes agree.
  This is an installation/integration check, NOT InfiniteBench accuracy. lm_eval
  logged harmless inability to query an HF revision for an in-memory random model.

## Environment installed independently

Workspace environment: `/ocean/projects/cis260309p/jzhang114/prefill-routing-env`.
Read-only shared Torch=2.11.0+cu126 / Transformers=5.7.0 are exposed via .pth;
all new packages live in the workspace venv. `pip check` passed. lm_eval was
initially 0.4.13, then pinned to the exact commit in the official FlexPrefill
requirements: `a4987bba6e9e9b3f22bd3a6c1ecf0abd04fd5622` (version 0.4.3).
Datasets=3.6.0 retains trusted local dataset-script support. Metrics dependencies
are installed. No shared package was uninstalled or overwritten.

Official FlexPrefill adapter checkout:
`/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/FlexPrefill`, commit
`a24b27bf117fd6ad4ba34b0db82c598e7786fd59`, from
https://github.com/ByteDance-Seed/FlexPrefill . Only its benchmark adapter is used;
its old Triton/Transformers sparse-kernel stack was NOT installed or substituted
for this new routing implementation. Imports and synthetic end-to-end task passed.

User authentication and official checkpoint download are now complete. For future reauthentication only (never paste the token into chat):

```bash
HF_HOME=/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/hf-cache \
/ocean/projects/cis260309p/jzhang114/prefill-routing-env/bin/hf auth login
```

Downloaded model directory:
`/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/models/Llama-3.1-8B-Instruct`.
Real dataset and original input-reference manifests have not been fetched/aligned,
per the user's earlier deferral. Native bf16 is unavailable on the present V100;
the completed real 8B smoke explicitly used fp16, while formal bf16 parity needs
other hardware.

## Formal proposal — NOT ready for approval/submission yet

Candidate scope (requires confirmation after real-asset audit): three conditions,
all layers, fixed original-position modulo routing, all ten InfiniteBench tasks,
one greedy generation/document, seed 42, batch 1, chat=True, context 131072.
Task budgets and expected counts from actual checked source:

| task | expected samples | max new tokens |
|---|---:|---:|
| longbook_sum_eng | 103 | 1200 |
| longbook_qa_eng | 351 | 40 |
| longbook_choice_eng | 229 | 40 |
| longdialogue_qa_eng | 200 | 40 |
| longbook_qa_chn | 189 | 40 |
| code_debug | 394 | 128 |
| math_find | 350 | 10 |
| passkey | 590 | 15 |
| number_string | 590 | 128 |
| kv_retrieval | 497 | 128 |

3493 documents/condition, 10479 total generations IF actual assets match these
source counts. Never enforce these by silently dropping extra documents: confirm
original filtered dataset/task configs and ordered input identities first.

Use the official per-task score, macro-task score and sample-weighted score.
MATH500 `flag_acc`/`acc_strict` and its 31768-token termination heuristic are NOT
InfiniteBench metrics and are not carried over. Separately report exact EOS rate,
custom-stop rate and length-limit rate. All predictions/IDs/lengths/stop reasons,
input hashes, full config and source/model/environment provenance must be saved.
Timing excludes loading/warmup; prefill includes the full model forward (including
attention, projections, MLP and logits), decode-forward time is distinct from total
wall time (sampling/stopping/dispatch included). Peak allocated/reserved VRAM and
actual generated lengths accompany timing.

`formal.sbatch.draft` proposes a ceiling of 1 H100-80GB per worker, 24h/condition,
three sequential conditions: at most 72 GPU-hours, NOT an estimated runtime or
monetary quote. H100-80 is listed on Bridges-2; no resources were requested.
A single V100 cannot establish full 128k memory feasibility. Verified checkpoint memory lower bound:
32 layers x 8 KV heads x 128 head_dim x 131072 tokens x 2(K,V) x 2 bytes already
requires 16 GiB of KV; this is now confirmed against the downloaded checkpoint.
The earlier draft's 64 GiB figure was an arithmetic error and is corrected here.
Long-context runtime memory is not yet validated; the draft deliberately exits
with a blocking message. It must not be submitted as a runnable experiment.
Actual GPU count, runtime, and allocation-unit/currency cost remain unresolved
until memory profiling and user-approved resource validation.

Intended outputs: this directory's `records/formal_<approved_id>/<condition>/<task>`.
Before submission: immutable source snapshot and config hashes, explicit user
approval, atomic submission-lock directory, queue audit, save returned job ID;
never automatically retry an uncertain sbatch. A creates fresh dense reference
inputs, B/C must exactly match it. Old decode/prefill jobs and snapshots are untouched.

Deferred: real InfiniteBench prompt/input alignment and full dataset membership;
full-precision/backend equivalence beyond fp16 development smoke; long-context
runtime memory feasibility; formal measurements/grading. No accuracy-preservation
or speedup conclusion is made.

## Pretrained 8B smoke update (after authentication)

`records/llama8b_fp16_smoke/results.json` contains six completed records: two
synthetic examples using the OFFICIAL PassKey YAML prompt and REAL Llama tokenizer
chat template, each 93 tokens, A/B/C, greedy, max_new_tokens=15, SDPA, fp16,
existing V100 allocation. These are not downloaded InfiniteBench documents.
`records/pretrained_smoke_inputs.json` saves messages, input IDs and hashes.

| condition | prefill ms (two inputs) | total ms | new tokens | stopping | prediction observation |
|---|---|---|---|---|---|
| A dense | 79.24 / 74.70 | 342.61 / 341.25 | 9 / 9 | EOS / EOS | correct synthetic keys 12345 / 73921 |
| B Q-only | 419.20 / 418.82 | 884.90 / 878.50 | 15 / 15 | smoke cap / cap | repeated backslashes |
| C Q+KV | 418.10 / 416.77 | 877.17 / 879.15 | 15 / 15 | smoke cap / cap | repeated U+200D |

All 32 layers retained full prompt KV shape [1,8,93,128]; decode used the original
dense forward. Per-layer eligible causal pairs: A=139872, B=4371, C=183. Peak
allocated bytes: A=16095210496, B/C=16092331520 (~15.0 GiB, full model included).
Max-absolute first-next-token logit difference vs A: B=23.59375 / 23.3125,
C=21.734375 / 21.625. B/C outputs differ strongly from dense in these two cases.
These single measurements include implementation overhead and are not speedup
estimates. They must NOT select/tune the formal routing configuration or be
reported as InfiniteBench accuracy. Longer generated outputs also affect total
latency, so prefill and generation length remain separate in the report.
