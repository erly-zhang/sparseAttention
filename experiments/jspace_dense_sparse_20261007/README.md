# Fixed-prefix HMMT dense/sparse J-space pilot

Authorized scope: Qwen3-4B, HMMT, total KV budget K=2048, full prompt +
recent64 protection; Dense, full Random/W1, compact Random/W1. Compact uses
the frozen every-64-step selection after attention; full selects before every
decode attention. Actual attention is causal; original RoPE positions remain.
All four sparse arms import the frozen implementation from the existing HMMT
experiment without modifying it. Two numerically matched dense executions
use the append kernel (compact control) and gather/no-append kernel (full
control), respectively. Early smoke found significant BF16 cross-kernel drift
before pruning; therefore sparse effects are always measured against their
matching dense kernel, and cross-kernel drift is recorded separately.

The measurement lens is the public Neuronpedia Qwen3-4B lens at the commit
recorded in assets/lens_revision.txt. Its checkpoint contains 35 matrices of
2560x2560 and n_prompts=479. All arms use this SAME lens. No lens is fitted to
sparse outputs. This controls the measurement coordinate system.

First validate original-HF vs shared-kernel dense prefill, bitwise agreement
before sparse becomes active, actual causal/selected pair accounting, and all
four sparse paths including periodic compaction. Only then capture a pilot.

Pilot: fixed HMMT sample IDs 0,1,2, one dense-derived diagnostic rollout per
input, maximum 4096 new tokens, temperature .6, top-p .95, private seed42/input
identity RNG. This reproduces the frozen runner's sampling LAW (its custom
runner applies no top-k despite generation_config top_k=20), not its batch2
random-number consumption. It is not a formal accuracy evaluation. Each sparse
arm replays precisely the same dense-generated token IDs, starting with the
canonical prompt whose token SHA-256 is checked. Thus differing sampled words
cannot confound the comparison. Sparse prefixes after a divergence are
teacher-forced counterfactuals, not the sparse model's own generated reasoning.

Sample residuals at all layers every128 steps, plus positions around the first
full selection/compact events. Record whether each position is inside the
think span. Short/EOS traces that never exceed K are controls and must not be
presented as tests of active sparsity. Do not silently extend a think span or
ignore EOS to obtain a positive comparison.

Readout metrics: raw residual cosine and relative difference; fixed J-lens
top25 recall, full-vocab JS divergence, conceptual token ranks, and
temporal persistence. The final layer is the actual next-token distribution.
This pilot does not assume that Claude's workspace band applies to Qwen3.

J-space approximation: nonnegative gradient pursuit, at most25 unit-normalized
token directions, full vocabulary candidate search, projected line-search
updates. Our variant is explicit and numerically validated; the exact paper
implementation was not released. Primary directions are rows of W_U J_l;
RMSNorm-scale-aware directions can be separately labeled sensitivity checks.
Same-size unit-normalized Gaussian random dictionaries supply reconstruction
controls. Raw uncentered reconstruction gain and control excess are labeled
as such, not as the paper's corpus-centered fraction of variance. They do not
prove that sparse errors are caused by workspace degradation; causal claims
would require subsequent controlled interventions.

No existing experiments, environments, manifests, or historical results are
modified. Runs use independently scheduled idle L40 GPUs in the account-authorized
low QoS; standard QoS explicitly disallows L40. Diagnostic capture timings
include serialization and cannot be compared with formal A100 decode timings.

Sample2 can be captured on the second L40 into a staging directory. Only a
completed sample directory is published into the common capture tree, and an
existing sample is never overwritten. The original worker resumes/skips
already completed arms. Analysis waits for both workers. The launch manifest
records all job IDs. Scientific plots and the PDF companion retain matched
dense controls and separately report cross-kernel calibration.

## Completed pilot (2026-10-07)

All three inputs and all six paths completed. Input hashes, exact paired
prefixes, pre-budget equality and actual pair counts passed. Both dense
controls have identical captured residuals at K=2048. Sample1 compact arms
also passed bitwise pre-budget alignment against a dense control captured on
the other L40. No historical experiment was stopped or modified.

Active reasoning only; source blocks0-34; equal weight per sample:

| Sparse path | Unfiltered J-lens recall@25 | Full-vocabulary JS (nats) |
|---|---:|---:|
| Random, full KV | 0.93012 | 0.01820 |
| W1, full KV | 0.95579 | 0.00847 |
| Random, compact KV | 0.94071 | 0.01591 |
| W1, compact KV | 0.94852 | 0.01262 |

W1 has higher readout recall in each of these three cases; the mean advantage
is2.57 percentage points with full KV and0.78 with compact KV. This is a
fixed-prefix fidelity observation, not an accuracy result or population claim.
The standalone layer plot uses the separately labeled wordlike display filter,
which retains letters/numbers and excludes punctuation/special tokens.

The reconstruction control is negative even for Dense: raw uncentered gain
is4.77%, versus14.08% for the same-cardinality, full-vocabulary Gaussian
dictionary. An explicitly separate RMSNorm-scale-aware sensitivity using
`W_U diag(gamma) J_l` gives1.92%, still below the same random control. Primary
results are retained. Component cosine is frame-sensitive; the bounded
occupancy proxy must not be interpreted as established workspace capacity.
This does not establish capacity loss, causal workspace damage, or an explanation
of the formal accuracy difference. The short-WikiText lens, uncentered metric,
random control and approximate solver remain measurement limitations.

Each diagnostic trajectory is capped at4096 new tokens. Samples0 and2 are
still reasoning at that cap; sample1 closes its think span. All three have
active reasoning checkpoints. This measures sparsity onset, not the complete
32768-token reasoning regime.

Final files: `analysis/jspace_pilot_report.pdf` (eight-page Times New Roman
companion), `analysis/readout_by_layer.{png,pdf,svg}`,
`analysis/readout_by_step.{png,pdf,svg}`, `analysis/summary.json`,
`analysis/validation.json`, `analysis/interpretation.json`, and explicit
`analysis/gamma_sensitivity_summary.json`. Raw residuals, token IDs, readouts,
coding coefficients, gain curves and all source/lens hashes remain available.

The Figure 5 style offline viewer is `figure5_html/index.html`, packaged as
`/mmfs1/project/yl258/jz648/jspace_figure5_html.zip`. Extract the complete
folder and open `index.html` in a recent Chrome, Edge or Firefox browser.
It provides all three cases, Dense and four sparse paths, checkpoint/layer
selection, full-vocabulary token rank heatmaps and paired rank trajectories.
Columns are real captured checkpoints, not a fabricated continuous token grid.
Every displayed Top 8 token is pinnable; rank files load locally on demand.
Finite-precision ties retain exact lower/upper bounds; plots use their geometric
midpoint on a logarithmic rank axis. All underlying prefix and lens caveats above
also apply to this viewer.

`export_figure5.py` computes readouts from saved residuals without new generation.
Slurm job 1388662 completed on an idle L40 in 1m51s, exit 0; other experiments
were not stopped. `pack_figure5.py` stores exact rank intervals with lossless
delta/varint/gzip encoding. `validate_figure5_data.py` independently decoded
85,065,840 intervals and checked all 227,520 displayed Top 8 entries.
`figure5_html/data_validation.json` records exact counts. The offline Playwright
check passed all 15 sample/path combinations, pinning/search, matrix selection,
Dense overlay, phase filtering and a 390px viewport, with no page errors or
external requests (`figure5_html/browser_validation.json`).

The English report is `figure5_html_en/index.html`, packaged as
`/mmfs1/project/yl258/jz648/jspace_figure5_html_en.zip`. All interface text,
legends, method explanations, and the reading guide are in English; original
model token text remains unchanged. `localize_figure5.py` creates this version.
Its offline browser check also passed all sample/method combinations and the
390px viewport. `figure5_html_en/localization_validation.json` records unchanged
data checksums. The original Chinese archive is retained.

The corrected English-character token view is `figure5_html_en_v2/index.html`,
packaged as `/mmfs1/project/yl258/jz648/jspace_figure5_html_en_v2.zip`.
Earlier English UI reports still showed multilingual raw readout tokens. V2
recomputes Top8 over ASCII letter/number-containing tokens from the full
vocabulary, using the saved residuals, and labels this display scope prominently.
This is a character display filter, not token translation or language inference.
All token ranks/probabilities remain full-vocabulary quantities; inputs,
captured residuals, selector algorithms, Recall@25 and JS are unchanged.
`english_display_validation.json` verifies original prefix/metric parity and
137,479 shared-token entries; all 227,520 new Top8 entries passed independent
rank decoding. The browser checked all 15 sample/method combinations, with
ASCII-only exported token text and no Han characters in rendered page text.
The readout-only Slurm export job is 1388893; existing experiments continued.
Previous archives are preserved for the original multilingual readout view.

The numeric-rank viewer is `figure5_html_en_v3/index.html`, packaged as
`/mmfs1/project/yl258/jz648/jspace_figure5_html_en_v3.zip`. Every valid rank
heatmap cell now prints its exact full-vocabulary rank or tie interval.
The rank view tracks one fixed pinned token, with decode checkpoints along
columns and blocks along rows. When no token is pinned, a visible empty-state
overlay and one-click pin action replace the ambiguous blank view. Rank-grid
hover no longer falls back to an unrelated per-cell top token; pending tooltips
are invalidated on pointer exit, scroll or render. All V2 readout/rank data are
unchanged. The offline browser test validated two complete 1,872-cell canvas
renders against decoded rank arrays, checked label fit, token/method switching,
empty-state recovery and all existing sample/path interactions.
