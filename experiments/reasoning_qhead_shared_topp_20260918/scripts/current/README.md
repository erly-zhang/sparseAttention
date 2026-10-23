# Current entry points

These six aliases resolve to the original maintained scripts. See
[CODE_INDEX.md](../../CODE_INDEX.md) for configuration, environment and version status.

| Alias | Purpose |
|---|---|
| `w1_raw_qk_v2.py` | Completed formal full/hybrid raw-QK implementation |
| `w1_raw_qk_v3.py` | Validated fused prototype; no new full-HMMT result |
| `w1_interval1000.py` | Compact1000/hybrid reference |
| `official_random_pp.py` | Matched official per-KV-head random_pp |
| `refresh_results.py` | CPU-only raw-evidence validation and catalog regeneration |
| `build_report.py` | English PDF/HTML report construction |

Selecting a runner alone does not determine K, KV storage or generation settings.
Use the intended formal row's frozen configuration. Existing inference and
preparation scripts remain under `scripts/` for compatibility and historical controls.
