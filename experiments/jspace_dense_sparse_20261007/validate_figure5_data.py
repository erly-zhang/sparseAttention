"""Independently decode every packaged rank stream and check every Top8 cell."""
import base64,gzip,json,hashlib,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'figure5_html'

def decode(body):
    b=np.frombuffer(gzip.decompress(base64.b64decode(body)),dtype=np.uint8)
    starts=np.r_[0,np.flatnonzero(b[:-1]<128)+1]
    lengths=np.diff(np.r_[starts,len(b)])
    assert lengths.max()<=3 and b[-1]<128
    shifts=(np.arange(len(b))-np.repeat(starts,lengths))*7
    return np.add.reduceat((b.astype(np.uint32)&127)<<shifts.astype(np.uint32),starts).astype(np.int32)

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=OUT)
    OUT=parser.parse_args().directory
    result=dict(status='passed',samples=[])
    for sample in json.loads((OUT/'manifest.json').read_text())['samples']:
        path=OUT/sample['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==sample['sha256']
        text=path.read_text();d=json.loads(text[text.index(']=')+2:].removesuffix(';\n'))
        ids=np.array(d['top_ids']);lo=np.array(d['top_low']);hi=np.array(d['top_high'])
        order=np.argsort(ids,kind='stable');sorted_ids=ids[order]
        unique,starts,counts=np.unique(sorted_ids,return_index=True,return_counts=True)
        positions={str(t):order[start:start+count] for t,start,count in zip(unique,starts,counts)}
        checked=0;decoded=0
        for file in sorted((OUT/f"ranks_{d['idx']}").glob('*.js')):
            raw=file.read_text();items=json.loads(raw.removeprefix('Object.assign(window.FIGURE5_RANKS,').removesuffix(');\n'))
            for key,body in items.items():
                vals=decode(body).reshape(5,d['P']*36,2)
                delta=(vals[1:]>>1)^-(vals[1:]&1)
                vals[1:]=delta+vals[0]
                assert (vals[...,0]>=1).all() and (vals[...,1]>=0).all()
                vals[...,1]+=vals[...,0]
                assert (vals[...,1]<=d['vocab_size']).all()
                selected=positions[key.split('/')[1]];bounds=vals.reshape(-1,2)[selected//d['N']]
                assert np.array_equal(bounds[:,0],lo[selected]) and np.array_equal(bounds[:,1],hi[selected])
                checked+=len(selected);decoded+=vals.shape[0]*vals.shape[1]
        assert checked==len(ids)
        item=dict(idx=d['idx'],top8_cells_validated=checked,rank_intervals_decoded=decoded,all_full_vocabulary_bounds_valid=True)
        result['samples'].append(item);print(json.dumps(item),flush=True)
    (OUT/'data_validation.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
