# Own-prefix HMMT generation and fixed J-lens readouts

This follows the 2026-10-07 fixed-prefix pilot on the same canonical HMMT
inputs (IDs 0,1,2). Dense and four sparse methods each sample their own next
token; no method is forced to consume the Dense continuation. The frozen
selector code is imported without changes. Total K=2048 includes the full
prompt and recent64 protection; compact selection happens after attention
every64 steps. Full KV methods select before each decode attention and retain
unselected KV. Positions are shared within a layer across KV heads.

One batch1 diagnostic rollout per input/method, BF16 Qwen3-4B,
temperature0.6, top-p0.95, seed42, no top-k sampling, maximum4096 new tokens.
Private per-input CUDA generation RNG is reset for each method and independent
of selector RNGs. Each method genuinely forms its own KV and residual states.
This cap may truncate reasoning; it is not the full32768-token formal test and
three questions cannot establish an accuracy advantage.

Before the study, a64-token generation smoke checks all five methods agree
before pruning, actual pair counts, private RNG use, and EOS/cap behavior.
The prior pilot's six-path compaction/kernel smoke is also required to pass.
Each input's methods run on the same allocated A100; Slurm allocates only idle
resources. Existing jobs and historical output directories are untouched.
Each recorded trajectory includes the GPU name, node and job ID.

All36 block residuals are captured at stride128 and around sparsity boundaries.
A final unsampled forward captures the state after the last emitted token;
its time is excluded from the generation timing, and pair totals explicitly
include it. All timings still include diagnostic residual transfers and
sampling and are not formal efficiency benchmarks.

The same public fixed Qwen3-4B J-lens is used for all own-prefix residuals.
Readouts include English-character Top8 with full-vocabulary probabilities and
exact rank tie intervals, plus full-vocabulary Recall@25 and JS against Dense
at the same step. Records explicitly identify whether prefixes still match.
AFTER TEXT DIVERGENCE these metrics combine attention changes with changes in
text history; they are not isolated sparse effects or causal workspace evidence.
No new J-space capacity reconstruction is claimed.

Outputs: `captures/sample_<id>/<method>.json`, `.txt`, `.pt`, `readouts.jsonl`,
`validation.json`, and `complete.json`. Collection produces raw CSV/JSON and
English evidence notes, including first divergence, EOS/cap, think closure,
boxed answer (when complete), and exact repetition measures.

## English offline HTML explorer

`figure5_html_en/index.html` displays the genuine own-generated text of all
five methods. It includes pairwise first-divergence snippets, side-by-side
complete text, per-method context and reasoning labels, a same-prefix filter,
Figure5-style block/position readouts, and exact numeric pinned-token rank
heatmaps. Dense overlays after divergence are explicitly labeled as different
text histories. All15 traces remain capped diagnostics without final boxes.

Slurm readout export1389685 completed on an idle A100 in4m00s, exit0; no new
continuations were generated. All227,520 Top8 entries match the original
A100 IDs and rank tie bounds. Independent decoding checked113,926,860 rank
intervals. All15 methods/cases passed offline browser interaction checks,
including exact own text, per-method reasoning filtering, common-prefix
filtering, comparison switching, canvas rank labels and a390px viewport.
No external requests or browser errors occurred. Independent context checks
verified790 snippets directly against the original token IDs.

The complete offline package is
`/mmfs1/project/yl258/jz648/hmmt_free_generation_jlens_html_en_20261007.zip`.
Extract the whole folder and open index.html; retain every local rank folder.
The earlier fixed-Dense-prefix HTML and the original text/data ZIP are retained.
`html_archive_validation.json` records ZIP integrity, exact file counts and
archive SHA-256. The folder also includes provenance and validation reports,
raw generated text downloads, English notes and source scripts.
