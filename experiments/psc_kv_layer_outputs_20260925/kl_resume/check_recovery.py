"""Small CPU-only check: recovered mean is bitwise identical to original averaging."""
import tempfile
from pathlib import Path
import numpy as np
from resume import mean_heads
with tempfile.TemporaryDirectory() as td:
    folder=Path(td); rng=np.random.default_rng(42)
    arrays=[rng.normal(size=(4,7,7)).astype(np.float16) for _ in range(3)]
    files=[]
    for i,a in enumerate(arrays):
        path=folder/('%d.npy'%i);np.save(path,a);files.append(path)
    expected=np.zeros((4,7,7),dtype=np.float32)
    for a in arrays:expected+=a.astype(np.float32)
    expected/=3
    mean_heads(files,folder/'out.npy',expected.shape)
    np.testing.assert_array_equal(np.load(folder/'out.npy'),expected.astype(np.float16))
    arrays[1][0,0,0]=np.nan;np.save(files[1],arrays[1])
    try:mean_heads(files,folder/'bad.npy',expected.shape)
    except ValueError:pass
    else:raise AssertionError('nonfinite input was not rejected')
    assert not (folder/'bad.npy').exists()
print('PASS: headwise mean bitwise matches original; nonfinite maps rejected before publishing output')
