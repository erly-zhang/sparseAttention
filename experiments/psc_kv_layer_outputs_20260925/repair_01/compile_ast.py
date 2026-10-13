import importlib,sys
import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
mod=importlib.import_module('kernel_'+sys.argv[1]);fn=mod._triton_mixed_sparse_attn_fwd_kernel
const={'BLOCK_M':64,'BLOCK_N':64,'BLOCK_DMODEL':128,'dtype':tl.bfloat16}
sig={name:('*bf16' if name in ('Q','K','V','Out') else '*i32' if name in ('seqlens','block_count','block_offset','column_count','column_index') else 'fp32' if name=='sm_scale' else 'i32') for name in fn.arg_names if name not in const}
k=triton.compile(ASTSource(fn,sig,const),target=GPUTarget('cuda',90,32),options={'num_warps':4,'num_stages':2})
print('COMPILED',k.name,flush=True)
