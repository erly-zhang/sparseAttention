"""Add the opt-in hybrid selector to a copy of the patched TriAttention runtime.

Only active with TRIATTN_RUNTIME_SELECTOR=hybrid; the random_pp / triattention paths are
unchanged. Loud failure on missing anchors; idempotent.
"""
import sys
from pathlib import Path

rt = Path(sys.argv[1])
hook = rt/"hook_impl.py"
src = hook.read_text()
if "build_hybrid_selector" not in src:
    a = "from .selector_random import build_random_pp_selector\n"
    b = '    if selector_kind in {"random", "random_pp"}:\n'
    if src.count(a) != 1 or src.count(b) != 1:
        sys.exit("FATAL: hook_impl anchors not found")
    src = src.replace(a, a+"from .selector_hybrid import build_hybrid_selector\n")
    src = src.replace(b, '    if selector_kind == "hybrid":\n'
                         '        (\n            select_keep_indices,\n            select_keep_indices_for_group,\n'
                         '            selector_status,\n        ) = build_hybrid_selector(config, base_runner=base_runner)\n'
                         + b.replace("    if ", "    elif "))
    compile(src, str(hook), "exec")
    hook.write_text(src)
    print("hook_impl: hybrid dispatch added")
mp = rt/"integration_monkeypatch.py"
src = mp.read_text()
if "hybrid_attention" not in src:
    a = "    _PATCHED_SCHEDULER_ACTIVE = bool(patch_scheduler)\n    _PATCHED_WORKER_ACTIVE = bool(patch_worker)\n    _PATCHED = True\n"
    if src.count(a) != 1:
        sys.exit("FATAL: monkeypatch anchor not found")
    src = src.replace(a, '    if os.environ.get("TRIATTN_RUNTIME_SELECTOR", "").strip().lower() == "hybrid":\n'
                         '        from .hybrid_attention import install as _install_hybrid\n'
                         '        _install_hybrid()\n' + a)
    compile(src, str(mp), "exec")
    mp.write_text(src)
    print("integration_monkeypatch: hybrid attention install added")
