# Current comparison and separate references

The [main English PDF](../../../report_build/current/hmmt_results_en.pdf) is now
3 pages, with8 latest protected HMMT configurations: Dense, official random_pp,
raw-QK full-KV W1, compressed1000 W1 and W1 + random4096. It has no lookup IDs.
All W1 rows in its result tables use raw-QK scoring.

| Purpose | File |
|---|---|
| Main current HMMT comparison | [PDF](../../../report_build/current/hmmt_results_en.pdf) · [HTML](../../../report_build/current/index.html) |
| Current8 configurations | [CSV](../../../report_build/current/results.csv) |
| Earlier controls and other datasets | [Historical PDF](../../../report_build/current/historical_results_en.pdf) · [Historical CSV](../../../report_build/current/historical_results.csv) |
| Fused v3: correctness and single-layer timings only | [Validation PDF](../../../report_build/current/w1_v3_validation_en.pdf) |
| All60 complete reference results | [CSV](../../../report_build/current/all_results.csv) · [Catalog JSON](results/current/summary.json) |
| Exact report sources and presentation scope | [Report JSON](../../../report_build/current/report_data.json) |
| Current code and implementation status | [CODE_INDEX.md](CODE_INDEX.md) |

The historical report contains probability-scored W1, Shared Random controls,
the500-step predecessor, K4096 controls, unprotected runs, the older extra-prompt
budget, MATH500 and the other datasets. These remain scientifically useful
references but do not appear in the main current-method comparison.

The complete source catalog still contains60 verified groups. Only3 equivalent
older implementations are superseded in that catalog. A different algorithm is
never erased because another configuration scored higher.

Raw output texts, grades, metrics and frozen inference sources are unchanged.
The previous8-page report is preserved under
`.codex-persist/report_simplification_20261014/before/report/`. Older dated PDF/TeX
reports and J-lens studies remain separate historical snapshots.
