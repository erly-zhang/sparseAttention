import random
import torch
from counts import flex_pairs, minference_pairs


def test():
    rng=random.Random(20261007)
    checked=0
    for n in [1,7,63,64,65,127,128,129,193,257]:
        for bs in [8,64,128]:
            count=(n+bs-1)//bs
            for _ in range(10):
                selected=[i for i in range(count*count) if rng.random()<.35]
                dense=torch.zeros(n,n,dtype=torch.bool)
                for idx in selected:
                    row,col=divmod(idx,count)
                    dense[row*bs:min(n,(row+1)*bs),col*bs:min(n,(col+1)*bs)]=True
                got,errors=flex_pairs(torch.tensor(selected,dtype=torch.long),n,bs)
                assert int(errors)==0 and int(got)==int(dense.tril().sum())
                checked+=1
        # Build sorted disjoint, unaligned intervals and complement columns.
        rows=(n+63)//64
        starts=[];columns=[]
        dense=torch.zeros(n,n,dtype=torch.bool)
        for r in range(rows):
            s=r*64;e=min(s+64,n)
            choices=[];j=rng.randrange(0,8)
            while j<s+64:
                if rng.random()<.5: choices.append(j);j+=64
                else:j+=rng.randrange(1,24)
            cols=[j for j in range(s) if all(not a<=j<a+64 for a in choices) and rng.random()<.1]
            starts.append(choices);columns.append(cols)
            for a in choices:dense[s:e,a:min(a+64,n)]=True
            for j in cols:dense[s:e,j]=True
        width=max(1,max(map(len,starts)));cw=max(1,max(map(len,columns)))
        bo=torch.zeros(1,1,rows,width,dtype=torch.int32);ci=torch.zeros(1,1,rows,cw,dtype=torch.int32)
        bc=torch.tensor([[[len(a) for a in starts]]],dtype=torch.int32)
        cc=torch.tensor([[[len(a) for a in columns]]],dtype=torch.int32)
        for r,a in enumerate(starts):bo[0,0,r,:len(a)]=torch.tensor(a,dtype=torch.int32)
        for r,a in enumerate(columns):ci[0,0,r,:len(a)]=torch.tensor(a,dtype=torch.int32)
        got,errors,_,_=minference_pairs((bc,bo,cc,ci),n)
        assert int(errors)==0 and int(got)==int(dense.tril().sum()),(n,int(got),int(errors))
        checked+=1
    # Invalid overlap, duplicate and future columns must fail closed.
    bc=torch.tensor([[[2]]]);bo=torch.tensor([[[[0,32]]]])
    cc=torch.tensor([[[1]]]);ci=torch.tensor([[[[0]]]])
    assert int(minference_pairs((bc,bo,cc,ci),64)[1])>0
    return {'status':'PASS','independent_dense_reference_cases':checked,'invalid_support_rejected':True}

if __name__=='__main__':
    import json
    print(json.dumps(test()))
