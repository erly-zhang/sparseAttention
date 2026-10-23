# Shared KV selection and compression

Updated 14 October 2026. This is the maintained entry point for the Qwen3-4B
reasoning experiments. Begin with the links below rather than dated trial files.

| Purpose | Entry |
|---|---|
| Current English report, Times New Roman | [PDF](../../../report_build/current/hmmt_results_en.pdf) · [HTML](../../../report_build/current/index.html) |
| All complete results and exact configurations | [RESULTS.md](RESULTS.md) |
| Current code, versions and environment | [CODE_INDEX.md](CODE_INDEX.md) |
| Six current runner/maintenance aliases | [scripts/current/](scripts/current/README.md) |
| Current versus historical reports | [REPORTS.md](REPORTS.md) |
| Machine-readable complete result catalog | [CSV](results/current/results.csv) · [JSON](results/current/summary.json) |

## Current result status

The main PDF is a3-page comparison of8 latest protected raw-QK W1 / Dense /
official random_pp configurations. Earlier controls and other datasets are in
`historical_results_en.pdf`; v3 single-layer evidence is in `w1_v3_validation_en.pdf`.
There are no lookup IDs in the reader-facing report. The complete catalog below
is an evidence archive, not the main report's selection of current methods.

The catalog contains60 current complete groups:46 HMMT,10 MATH500 and four
other completed dataset splits. Three older, exactly equivalent implementations
are retained in implementation history. Different algorithms remain controls.
There are no incomplete or profiling-only accuracy rows in the current table.

HMMT uses Qwen3-4B, BF16, batch2, seed42, representative seed0, temperature0.6,
generation Top-p0.95 and a total prompt+output cap32768. The same60 questions
produce two sampled outputs each; accuracy is correct/120, not pass@2.

- Dense:48/120 (40.00%).
- Full-KV raw-QK W1, K1024, optimized v2:48/120 (40.00%).
- Raw-QK W1 + random4096, K1024, optimized v2:46/120 (38.33%).
- Raw-QK W1 compact1000, K2048:40/120 (33.33%).
- Raw-QK W1 + random4096, K2048, interval1000:55/120 (45.83%).

The fused v3 implementation passed GPU correctness and single-layer timing
checks at K1024/K2048. It has no new full-Qwen3-4B/HMMT result and is not the
default. See [v3 evidence](W1_RAW_QK_FUSED_OPTIMIZATION_20261014.md).

## Method flow

All methods use dense prefill. Raw-QK W1 uses the current post-RoPE representative
query to rank live keys, with correct GQA mapping and earlier-position ties.
All heads in one layer share selected positions, preserving their own KV values.
Prompt+recent64 protection counts inside the total attention budget K.

| Method | Attention during decode | Physical KV |
|---|---|---|
| Dense | All historical keys | Full history |
| Full-KV W1 | Select K every step | Full history |
| Compact W1 / Shared Random | All surviving keys plus newly appended KV | Compress to K after attention every64/500/1000 steps |
| W1 + random4096 | Select K every step | Every1000 steps retain that W1 core plus4096 random positions outside it |

The hybrid pool grows between updates to at most K+4096+1000. Random4096
positions are candidates for future W1 selection, not extra attention keys.
Historical probability scores and older extra-prompt `reasoning1024` budgets
are distinct configurations; they remain separately labeled.

## Maintenance

```bash
/usr/bin/python3 scripts/current/refresh_results.py --check
/usr/bin/python3 scripts/current/refresh_results.py
/usr/bin/python3 scripts/current/build_report.py
```

These commands perform CPU-only validation/reporting and submit no experiments.
They replace only derived current tables and reports. Raw runs, frozen sources,
outputs and grading evidence remain unchanged. See the code guide for inference
assets, configuration and scheduler preparation.

The [previous detailed guide](docs/history/README_before_20261014.md) is retained
for historical reference; its dated status and launch examples are not the
current experiment state. Archived runs remain indexed under results/archive/.
J-lens and Llama prefill/KV-retrieval studies are separate projects.
