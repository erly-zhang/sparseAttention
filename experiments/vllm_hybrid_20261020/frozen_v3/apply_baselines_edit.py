"""Add the opt-in SnapKV / R-KV / VaSE selectors (our port) to a copy of the fixed runtime.

Active only with TRIATTN_RUNTIME_SELECTOR in {snapkv, rkv, vase}; other selectors unchanged.
"""
import sys
from pathlib import Path

rt = Path(sys.argv[1])
hook = rt/"hook_impl.py"
src = hook.read_text()
if "build_baseline_selector" not in src:
    a = "from .selector_random import build_random_pp_selector\n"
    b = '    if selector_kind in {"random", "random_pp"}:\n'
    if src.count(a) != 1 or src.count(b) != 1:
        sys.exit("FATAL: hook_impl anchors not found")
    src = src.replace(a, a+"from .baselines_vllm import SELECTORS as _BASELINE_SELECTORS, build_baseline_selector\n")
    src = src.replace(b, '    if selector_kind in _BASELINE_SELECTORS:\n'
                         '        (\n            select_keep_indices,\n            select_keep_indices_for_group,\n'
                         '            selector_status,\n        ) = build_baseline_selector(config, base_runner=base_runner)\n'
                         + b.replace("    if ", "    elif "))
    compile(src, str(hook), "exec")
    hook.write_text(src)
    print("hook_impl: baseline dispatch added")
mp = rt/"integration_monkeypatch.py"
src = mp.read_text()
if "baselines_vllm" not in src:
    a = "    _PATCHED_SCHEDULER_ACTIVE = bool(patch_scheduler)\n    _PATCHED_WORKER_ACTIVE = bool(patch_worker)\n    _PATCHED = True\n"
    if src.count(a) != 1:
        sys.exit("FATAL: monkeypatch anchor not found")
    src = src.replace(a, '    if os.environ.get("TRIATTN_RUNTIME_SELECTOR", "").strip().lower() in ("snapkv", "rkv", "vase"):\n'
                         '        from .baselines_vllm import install as _install_baselines\n'
                         '        _install_baselines()\n' + a)
    compile(src, str(mp), "exec")
    mp.write_text(src)
    print("integration_monkeypatch: baseline query capture install added")
