import sys,importlib.util
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
for file,names in [('probmean_selector.py',['_lse','_prob']),('member_score.py',['_sample_lse','_sample_prob'])]:
    spec=importlib.util.spec_from_file_location('audit_'+file[:-3],HERE/file);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    for name in names:
        fn=getattr(m,name);const={a:128 for a in fn.arg_names if a in ('D','M','B')}
        sig={a:('*bf16' if a in ('Q','K') else '*fp32' if a in ('L','P') else 'i32') for a in fn.arg_names if a not in const}
        obj=triton.compile(ASTSource(fn,sig,const),target=GPUTarget('cuda',90,32),options={'num_warps':8 if file.startswith('prob') else 4,'num_stages':2})
        print('COMPILED',obj.name,flush=True)
