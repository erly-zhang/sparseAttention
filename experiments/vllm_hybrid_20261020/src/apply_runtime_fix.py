"""Row-index fix + diagnostics for a COPY of the TriAttention vLLM 0.19 runtime.

Defect (present in the paper's shims at checkout 9c26d14): the effective-length
overrides (per-row seq_len base and KV slot delta) are keyed by input_batch row and built
from input_batch.req_id_to_index BEFORE base_runner.execute_model, whose _update_states
removes finished requests, adds new ones and condense()s the persistent batch. Whenever
the batch composition changes, the overrides land on the wrong rows: a compressed request
can attend with its logical (uncompressed) length and write its new KV through a wrong
slot delta, while another row inherits its override.

Fix (TRIATTN_FIX_ROW_REMAP=1, default): remember row -> req_id when the overrides are built
and, inside the patched _prepare_inputs (after _update_states), remap every override to the
request's current row. Selection, compaction and kernels are untouched.

Diagnostics (always on, counters only): stale rows (override row now holds a different
request), remapped rows, and a capacity check that every row's seq_len fits in the KV
blocks it owns (violated only by a missing/wrong override). TRIATTN_ROW_CHECK_RAISE=1
makes a capacity violation fatal.

Also: TRIATTN_STABLE_REQ_SEED=1 (default) derives random_pp's seed from the request id
without vLLM's random 8-character suffix, so a rerun reproduces the same masks.
Idempotent; loud failure on any missing anchor.
"""
import sys
from pathlib import Path

rt = Path(sys.argv[1])


def edit(name, pairs, marker):
    p = rt/name
    s = p.read_text()
    if marker in s:
        print(f"{name}: already applied")
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"FATAL: anchor not unique/found in {name}: {old[:80]!r}")
        s = s.replace(old, new)
    compile(s, str(p), "exec")
    p.write_text(s)
    print(f"{name}: patched")


# 1. state: row -> req_id map, remap helper, counters
edit("input_patch_state.py", [
    ("ACTIVE_EFFECTIVE_MAPPING_VALIDATED: bool = False\n",
     "ACTIVE_EFFECTIVE_MAPPING_VALIDATED: bool = False\n"
     "# ROW-REMAP FIX: request id of every override row at build time (pre-_update_states).\n"
     "ACTIVE_REQ_ID_BY_ROW: dict[int, str] | None = None\n"
     "ROW_STATS: dict[str, int] = {\"steps\": 0, \"override_steps\": 0, \"stale_rows\": 0, \"remapped_rows\": 0,\n"
     "                             \"dropped_rows\": 0, \"capacity_violations\": 0, \"violation_steps\": 0}\n"),
    ("    expected_query_lens: tuple[int, ...] | None = None,\n) -> None:\n"
     "    global ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX, ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX\n",
     "    expected_query_lens: tuple[int, ...] | None = None,\n    req_id_by_row: dict[int, str] | None = None,\n) -> None:\n"
     "    global ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX, ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX\n"
     "    global ACTIVE_REQ_ID_BY_ROW\n"
     "    ACTIVE_REQ_ID_BY_ROW = req_id_by_row\n"),
], "ACTIVE_REQ_ID_BY_ROW")
p = rt/"input_patch_state.py"
s = p.read_text()
if "def remap_active_rows_by_req_id" not in s:
    s += '''

def remap_active_rows_by_req_id(req_ids: list, req_id_to_index: dict, fix: bool) -> None:
    """ROW-REMAP FIX: move every override to the row its request occupies after _update_states."""
    by_row = ACTIVE_REQ_ID_BY_ROW
    if not by_row:
        return
    ROW_STATS["override_steps"] += 1
    for old, rid in by_row.items():
        now = req_ids[old] if 0 <= old < len(req_ids) else None
        if now != rid:
            ROW_STATS["stale_rows"] += 1
    if not fix:
        return
    new_of = {}
    for old, rid in by_row.items():
        new = req_id_to_index.get(rid)
        if new is None:
            ROW_STATS["dropped_rows"] += 1
        else:
            new_of[old] = int(new)
            if int(new) != old:
                ROW_STATS["remapped_rows"] += 1

    def remap(m):
        if not m:
            return m
        out = {new_of[int(k)]: v for k, v in m.items() if int(k) in new_of}
        return out or None
    expected = ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU
    if expected is not None:
        expected = tuple(new_of[int(v)] for v in expected.tolist() if int(v) in new_of) or None
    queries = ACTIVE_EXPECTED_QUERY_LENS_CPU
    set_active_effective_sparse_overrides(
        effective_base_by_req_idx=remap(ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX),
        effective_pos_delta_by_req_idx=remap(ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX),
        single_effective_seq_base=ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE,
        single_effective_pos_delta=ACTIVE_SINGLE_EFFECTIVE_POS_DELTA,
        expected_req_row_indices=expected,
        expected_query_lens=tuple(queries.tolist()) if queries is not None else None,
        req_id_by_row={new_of[o]: r for o, r in by_row.items() if o in new_of},
    )
'''
    compile(s, str(p), "exec")
    p.write_text(s)

# 2. backend activation passes the map through
edit("input_patch_backend.py", [
    ("    expected_query_lens: tuple[int, ...] | None = None,\n) -> None:\n    set_active_effective_overrides_enabled(True)\n",
     "    expected_query_lens: tuple[int, ...] | None = None,\n    req_id_by_row: dict[int, str] | None = None,\n) -> None:\n"
     "    set_active_effective_overrides_enabled(True)\n"),
    ("        expected_query_lens=expected_query_lens,\n    )\n\n\ndef clear_effective_overrides",
     "        expected_query_lens=expected_query_lens,\n        req_id_by_row=req_id_by_row,\n    )\n\n\ndef clear_effective_overrides"),
], "req_id_by_row=req_id_by_row")

# 3. adapter records row -> req_id when the overrides are built
edit("input_adapter.py", [
    ("    expected_query_lens: tuple[int, ...] | None = None\n\n    def activate(self) -> None:",
     "    expected_query_lens: tuple[int, ...] | None = None\n    req_id_by_row: dict[int, str] | None = None\n\n    def activate(self) -> None:"),
    ("            expected_query_lens=self.expected_query_lens,\n        )\n\n    @staticmethod",
     "            expected_query_lens=self.expected_query_lens,\n            req_id_by_row=self.req_id_by_row,\n        )\n\n    @staticmethod"),
    ("    return EffectiveInputOverrides(\n        seq_base_map=seq_base_map,",
     "    # ROW-REMAP FIX: rows here are pre-_update_states; remember whose they are.\n"
     "    req_id_by_row = None\n"
     "    if isinstance(req_id_to_index, dict) and (seq_base_map or pos_delta_map):\n"
     "        rows = set(seq_base_map or {}) | set(pos_delta_map or {})\n"
     "        req_id_by_row = {int(i): r for r, i in req_id_to_index.items() if int(i) in rows}\n"
     "    return EffectiveInputOverrides(\n        req_id_by_row=req_id_by_row,\n        seq_base_map=seq_base_map,"),
], "req_id_by_row: dict[int, str] | None = None")

# 4. patched _prepare_inputs: remap after _update_states, then check capacity
edit("input_patch_vllm_v1_backend.py", [
    ("        if not (hasattr(self.positions, \"np\") and hasattr(self.seq_lens, \"np\")):\n"
     "            _apply_overrides_gpu_resident(\n"
     "                self,\n"
     "                num_scheduled_tokens,\n"
     "                total_num_scheduled_tokens,\n"
     "                num_reqs,\n"
     "            )\n"
     "            return out\n",
     "        if not (hasattr(self.positions, \"np\") and hasattr(self.seq_lens, \"np\")):\n"
     "            # ROW-REMAP FIX: overrides were keyed by pre-_update_states rows.\n"
     "            _patch_state.remap_active_rows_by_req_id(\n"
     "                list(self.input_batch.req_ids[:num_reqs]),\n"
     "                self.input_batch.req_id_to_index,\n"
     "                os.environ.get(\"TRIATTN_FIX_ROW_REMAP\", \"1\") == \"1\",\n"
     "            )\n"
     "            _apply_overrides_gpu_resident(\n"
     "                self,\n"
     "                num_scheduled_tokens,\n"
     "                total_num_scheduled_tokens,\n"
     "                num_reqs,\n"
     "            )\n"
     "            _row_capacity_check(self, num_reqs)\n"
     "            return out\n"),
    ("def make_patched_v1_prepare_inputs(",
     "def _row_capacity_check(self: Any, num_reqs: int) -> None:\n"
     "    \"\"\"Every row's attended length must fit in the KV blocks it owns.\"\"\"\n"
     "    stats = _patch_state.ROW_STATS\n"
     "    stats[\"steps\"] += 1\n"
     "    bt = self.input_batch.block_table\n"
     "    tb = bt.block_tables[0] if hasattr(bt, \"block_tables\") else bt\n"
     "    seq = self.optimistic_seq_lens_cpu[:num_reqs].numpy()\n"
     "    cap = np.asarray(tb.num_blocks_per_row[:num_reqs], dtype=np.int64)*int(tb.block_size)\n"
     "    bad = np.nonzero(seq > cap)[0]\n"
     "    if bad.size:\n"
     "        stats[\"capacity_violations\"] += int(bad.size)\n"
     "        stats[\"violation_steps\"] += 1\n"
     "        if stats[\"violation_steps\"] <= 20:\n"
     "            r = int(bad[0])\n"
     "            print(f\"[TRIATTN-ROWCHECK] capacity violation row={r} req={self.input_batch.req_ids[r]} \"\n"
     "                  f\"seq_len={int(seq[r])} owned={int(cap[r])} num_reqs={num_reqs} rows={bad.size}\", flush=True)\n"
     "        if os.environ.get(\"TRIATTN_ROW_CHECK_RAISE\") == \"1\":\n"
     "            raise RuntimeError(\"TRIATTN_ROW_CAPACITY_VIOLATION\")\n"
     "    if stats[\"steps\"] % 4096 == 0:\n"
     "        print(f\"[TRIATTN-ROWCHECK] stats {stats}\", flush=True)\n"
     "\n\n"
     "def make_patched_v1_prepare_inputs("),
], "_row_capacity_check")

# 5. random_pp seed without vLLM's random request-id suffix
edit("selector_random.py", [
    ("    def _active_req_id() -> str:\n        if base_runner is None:\n            return \"?\"\n"
     "        return str(getattr(base_runner, \"_triattention_active_req_id\", \"?\"))\n",
     "    def _active_req_id() -> str:\n        if base_runner is None:\n            return \"?\"\n"
     "        rid = str(getattr(base_runner, \"_triattention_active_req_id\", \"?\"))\n"
     "        # STABLE SEED: vLLM 0.19 appends '-<8 random chars>' to every request id.\n"
     "        head, _, tail = rid.rpartition(\"-\")\n"
     "        if os.environ.get(\"TRIATTN_STABLE_REQ_SEED\", \"1\") == \"1\" and head and len(tail) == 8:\n"
     "            return head\n"
     "        return rid\n"),
], "STABLE SEED")
