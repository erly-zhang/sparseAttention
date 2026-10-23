"""Verify (or with --write, create) the SHA-256 manifest of the v2 code trees and scripts."""
import hashlib, json, sys
from pathlib import Path
ROOTS = ["/scratch/yl258/jz648/triattn_vllm/tri_baselines_fix/triattention",
         "/mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/frozen_v3"]
def scan():
    out = {}
    for r in ROOTS:
        for p in sorted(Path(r).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".sh") and "__pycache__" not in p.parts:
                if p.is_symlink():
                    sys.exit(f"symlink in frozen tree: {p}")
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
man = Path(sys.argv[1])
if "--write" in sys.argv:
    man.write_text(json.dumps(scan(), indent=1)+"\n"); print("written", len(json.loads(man.read_text())))
else:
    want, got = json.loads(man.read_text()), scan()
    bad = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    print("frozen check:", "OK" if not bad else "MISMATCH", len(want), "files", hashlib.sha256(man.read_bytes()).hexdigest())
    for b in bad[:20]: print("  ", b)
    sys.exit(1 if bad else 0)
