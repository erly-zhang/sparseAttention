# Shared attention-map row stitching pilot

New experiment, not output repetition and not MLP training. Existing experiments
and checkpoints are untouched. User approved the single-H100 smoke and one
submission from the authenticated v020 terminal.

For every query head h, compute all N*N scores Q_h K_g(h)^T / sqrt(d), apply
the original-position causal mask, then softmax over the full key dimension.
Assign token i to head original_position(i) % H. Set shared_A[i,:] to precisely
that head's normalized row, retaining original query order. Each output head
then computes O_h = shared_A @ V_g(h). Concatenate outputs and use the original
full o_proj once, with no scaling, followed by the unchanged model computation.
Full maps are computed sequentially per head solely to bound memory; no query
row is skipped in score computation. Assignment is round-robin, balanced to
within one token, not a contiguous-block assignment or sorted-row concatenation.

Default inherited configuration: Llama3.1-8B,32 query heads,8 KV heads,d=128;
16 alternating modified layers0,2,...30. Other layers unchanged. Prefill only,
full original KV cache, unchanged dense decode. No learned correctors. GQA
implies identical outputs within each four-query-head KV group after sharing A;
different KV groups keep their own V. This is not32 independent value heads.

Both experimental shared_map and dense_full control use explicit FP32 scores,
softmax and AV (FP64 for numerical unit tests); outputs return to model dtype.
The control computes each head's own A_h @ V_g(h) with the same implementation.
It is a mathematically dense control, not promised bitwise BF16 FA2 parity.
Do not claim latency gains: all H full maps are evaluated deliberately.

Approved smoke (not yet submitted):3 fresh synthetic UUID retrieval
prompts near4K tokens, dense_full vs shared_map,6 greedy generations max128tokens,
seed42,1H10080GB8CPU128GBmax30min. No training or evaluation-based tuning.
The user's N/16 example is interpreted using the model's actual H=32; head count
and16 modified layers are different quantities. Any all-layer experiment is
separate from this inherited16-layer configuration.

CPU validation:
    python test_attention.py

Required remote gates before a job: frozen sources and tokenized inputs, all CPU
tests no skips, actual sbatch test-only, explicit budget and submission-route
authorization. GPU completion must include SlurmCOMPLETED0:0,6 aligned records,
full caches/finite logits, row-normalization/causality/ownership diagnostics,
independent exact-match/tokenization/repetition/cap scoring and actual outputs.
Passing implementation tests is not evidence of retrieval quality recovery.
