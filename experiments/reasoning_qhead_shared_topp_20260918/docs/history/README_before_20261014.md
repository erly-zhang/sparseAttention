# Shared-mask reasoning attention

This directory contains Qwen3-4B decode-time attention experiments on MATH500,
HMMT, GPQA-Diamond, AIME, and LiveCodeBench, plus matched official `random_pp`
controls. Prefill uses dense attention. Historical methods and tests remain
available where they are needed to reproduce reference comparisons.

## Results

Use [RESULTS.md](../../RESULTS.md) as the fixed current comparison entry point.
The [CSV](../../results/current/results.csv) and [JSON](../../results/current/summary.json)
contain one primary row per verified complete method/settings, with source
hashes and measurement definitions. Old code implementations are replaced in
this table only after complete equivalence checks; raw-QK versus probability,
budgets, mask protection and compaction intervals remain independent controls.
Standalone smokes, failed runs and duplicate snapshots are in the
[archive](../../results/archive/README.md), with compatibility links for old paths.
Dated reports are historical snapshots. Rebuild the current index with
`python scripts/consolidate_results.py`; use `--check` for a read-only audit.

## Current W1 implementation

Each layer selects one representative Q head using a private deterministic
generator keyed by representative seed, canonical input identity, layer and
absolute decode step. For GQA, Q head h reads KV head h divided by the number
of Q heads sharing one KV head. All heads in a layer share selected token
positions; each KV head retains its own K/V values. Layers and rollouts can
select different positions.

Raw-QK W1 ranks the current post-RoPE query/key dot products directly, without
a selector softmax. Equal BF16 scores prefer earlier chronological positions.
Actual attention still uses its ordinary scaling and softmax. Raw-QK masks need
not be bitwise identical to the historical probability scores because scaling,
rounding and softmax underflow can produce different ties.

The fixed total attention budget K includes all prompt positions and the latest
64 reasoning positions. Protected positions are excluded before ranking the
remaining candidates. This total-K convention differs from older
`reasoning1024` methods, which give the prompt an additional budget.

| Method | Decode attention | Physical cache |
|---|---|---|
| Full-history W1 | Select K every forward | Retain all historical KV |
| Periodic W1 compression | Attend all surviving KV plus new tokens | After attention every 64, 500 or 1000 steps, compress to K |
| W1 plus random4096 candidates | Select K every forward | After attention every 1000 steps, retain that core plus 4096 random positions from its complement |

The hybrid method skips physical eviction when all candidates fit in K+4096.
New KV accumulates between updates, so live capacity is K+4096+1000; the 4096
extra candidates do not participate in attention unless W1 selects them on a
later step. Uniform float32 priorities and independent secondary random bits
select distinct residual positions. The core is masked before random TopK.
Private mask RNG does not change generation sampling RNG. Original RoPE
positions remain attached to keys after chronological compaction.

## Implementation and launch entry points

| Purpose | Files |
|---|---|
| Historical full/compact selector oracle | `ra_harness/kvcompress/engine/selector_grid_cache.py` |
| Optimized full-history W1 | `ra_harness/kvcompress/engine/full_w1_optimized_cache.py` |
| Compact1000 and hybrid random4096 | `ra_harness/kvcompress/engine/w1_interval1000_cache.py` |
| Opt-in full raw-QK runner | `scripts/run_full_w1_qk.py` |
| Optimized full W1, default raw-QK since 2026-10-12 | `scripts/run_full_w1_optimized.py`; set `RA_FULL_W1_SCORE_MODE=probability` for the historical control |
| Raw-QK full/hybrid W1 implementation optimization v2 | `scripts/run_w1_qk_fast.py`; exact packed BF16 TopK, fused index assembly and the original hybrid Uniform RNG stream |
| Opt-in full/compact500 raw-QK runner | `scripts/run_w1_qk.py` |
| Opt-in compact1000/hybrid1000 runner | `scripts/run_w1_interval1000.py` |
| Freeze and submit latest HMMT runs | `scripts/prepare_hmmt_w1_interval1000.py`, `scripts/submit_hmmt_w1_interval1000.py` |
| Matched official random_pp | `scripts/hmmt_official_random_pp.py`, `scripts/run_official_random_pp.py` |
| Shared random64-step compression across datasets | `scripts/prepare_compact_multibench.py`, `scripts/run_compact_multibench.py` |
| Correct-answer output length analysis | `scripts/analyze_hmmt_correct_output_lengths_20261011.py` |

Implementations are enabled by process-local factory installation. Existing
launches and frozen reference sources are not silently switched to new methods.
The vendored official random-attention implementation retains its license,
third-party notices, and provenance.

## HMMT settings and validation

The latest configurations use Qwen3-4B, BF16, batch2, seed42, representative
seed0, generation temperature 0.6, generation Top-p 0.95, and total prompt plus
output limit 32768. Each method runs the same 60 inputs with two sampled outputs
each. Accuracy is correct outputs divided by 120, not pass@2 or 120 distinct
questions. K=1024 runs precede K=2048 runs.

Each launch freezes source/configuration/data and records SHA-256 provenance.
GPU correctness must pass before complete generation on the same three smoke
inputs. Both method smokes must pass input and analytic attention-pair checks
before formal shards start. Collection requires 60 unique input records and 120
unique `(idx,run)` grades, exact input token counts and input-ID hashes, and
`input_alignment=passed/full`. Incomplete runs do not receive formal accuracy.

GPU-synchronized decode wall time includes selection, random priorities,
sorting, gathering, physical compaction, attention, sampling and bookkeeping.
It excludes model loading, warmup, and metric serialization. Decode seconds
per question measure a batch generating two outputs together. Output length
is measured per individual output and excludes the prompt. Consequently,
output mean length and batch forward count are different metrics.

Tests include independent mask and FP32 attention references, GQA, protection,
short histories, stable ties, absolute positions, two physical compression
events, private RNG, residual sampling sanity and active-batch shrink.

From this directory, in the reasoning environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=ra_harness OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_w1_interval1000.py -v
```

CUDA checks are skipped on a CPU-only node. Formal GPU checks require an
independent scheduler allocation and an A100 80GB GPU.

## Environment and local reproduction inputs

Reasoning runs use the `random-attention` conda environment: Python 3.10,
PyTorch 2.4.0+cu121, Transformers 5.0.0, FlashAttention 2.7.3, and NumPy 2.2.6.
This differs from the SharePrefill environment in the repository's
`ENVIRONMENT.md`; do not substitute that environment for these runs.

Cluster launchers retain their recorded scheduler account and installation
paths. Adapt these to another cluster before submitting. Model weights,
authorized dataset files, input references, previous results and frozen run
templates are local artifacts and are excluded from Git. The preparation
scripts deliberately require these reference artifacts to preserve input and
baseline alignment; a fresh Git clone alone is not enough for a formal rerun.
The latest preparation script requires the completed raw-QK K1024
full/compact500 reference directory; K2048 comparisons additionally reference
the existing probability-scored full and compact500 experiments. Those K2048
comparisons therefore also change score arithmetic and do not isolate only
the compression interval.

After restoring the required local artifacts, the latest launch workflow is:

```bash
python scripts/prepare_hmmt_w1_interval1000.py --budget 1024
python scripts/submit_hmmt_w1_interval1000.py --budget 1024
python scripts/prepare_hmmt_w1_interval1000.py --budget 2048
python scripts/submit_hmmt_w1_interval1000.py --budget 2048 --after K1024_COLLECT_JOB_ID
```

Preparation and submission refuse existing sealed runs or duplicate launches.
Do not run these commands on an already submitted directory. Results retain
immutable sources even when the maintained checkout is reorganized.

## Repository cleanup

The approved 2026-10-11 cleanup removed duplicate debug source, a redundant
Protocol B copy, two early one-off Top-p launchers, edit backups and Python
cache files. Unique baseline implementations, tests, raw outputs, retained
Protocol B sources and running experiment snapshots were preserved. A local
rollback archive and exact hashed deletion manifest were recorded outside
the Git repository.
