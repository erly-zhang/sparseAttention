# Per-head Vertical / Slash ablation

The base Token support is unchanged: one frozen representative per layer supplies
four-probe weighted raw QK scores (weights 0.2, 0.3, 0.4, 0.1), with up to 8192 key
tokens per 128-query tile. Calibration and random representatives are reused.

Each query head now independently supplies Vertical, or Vertical and Slash,
using its own last 128 queries and corresponding GQA key head. The existing
causal FP32 attention probabilities are aggregated over key columns for Vertical
and query-key distances for Slash. Both use gamma 0.95, with no count cap. Each
head's final mask is shared Token union own Vertical union optional own Slash.
The probabilities used for selection are not shared. Main attention uses the
head's original Q/K/V and renormalizes over its own causal retained mask.

All 32 layers, the identical original 200 InfiniteBench inputs (50 each of KV
Retrieval, En.MC, En.QA, Code.Debug), chat templates, original positions, full KV
cache, greedy seed 42 and task-specific original output caps are retained. No
text hints, J-lens interventions, fitting, training or tokenizer changes occur.
There are four new methods and 800 sample-method evaluations. Previous Dense,
shared-pattern variants, FlexPrefill and MInference scores provide paired
references. Their source and input hashes are checked; one full-context KV
sentinel per historical method must reproduce saved generated token IDs before
formal jobs may run. The comparison measures quality at the same gamma and
base budget, not equal final sparsity. More or fewer tokens may be selected by
independent patterns; record actual logical and scheduling densities separately.

The implementation reuses the existing source-bound selection helpers, exact
per-query slash kernel, output projection, MLP implementation and Dense decode.
One head's augmented CSR is built and consumed at a time to avoid materializing
32 full-context CSR copies. The original KV slice is explicit, preserving GQA.
CPU tests compare causal masks with an independent literal oracle, count density,
check own-head sources, compare the representative head to its old full mask and
validate probability budgets. Approved GPU tests compare all 32 heads with an
independent masked FP32 reference and constant KV values at a partial query tile.

GPU stages after explicit user approval: (1) one synthetic kernel proof; (2) seven
parallel historical full-context sentinels after the proof; (3) 160 formal workers,
five samples each, after all sentinels pass. Each worker uses one H100-80GB,
eight CPUs, 128 GB host memory, account-allowed gpu QoS, requested eight hours
with a two-hour minimum backfill window. There is no artificial GPU concurrency
or GPU-hour cap. Scheduler constraints still apply. Progress snapshots every
five minutes, decode prefixes every eight steps, atomic results after each sample;
restarts skip hash-verified completed samples but rerun an incomplete prefill.
CPU monitoring and successor jobs cover prolonged queues. Frozen historical
files are read only; all new artifacts reside in this experiment directory.

Execution has not been approved for the concrete configuration in this document.
Preparation and CPU mask tests do not load an inference model or use a GPU.
