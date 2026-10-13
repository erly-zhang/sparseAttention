import sys
import triton
from triton.backends.compiler import GPUTarget
kernel=triton.compile(sys.argv[1], target=GPUTarget('cuda',90,32),options={'num_warps':4,'num_stages':2})
print('COMPILED',kernel.name,flush=True)
