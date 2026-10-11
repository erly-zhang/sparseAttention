"""Run the isolated W1 compact1000 / hybrid random4096 interval1000 methods."""
import runpy
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(base/'ra_harness'), str(base/'ra_harness/kvcompress/harness')]
from kvcompress.engine.w1_interval1000_cache import install
install()
print('[W1 interval1000] raw QK; compact K or attention K plus random4096 candidate cache', flush=True)
runpy.run_path(str(base/'ra_harness/kvcompress/harness/eval_hf.py'), run_name='__main__')
