"""Pack exact rank bounds as compressed deltas; load small local JS chunks on pin."""
import base64,gzip,json,hashlib,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'figure5_html'

def varints(values):
    values=values.astype(np.uint32).ravel()
    columns=[];valid=[]
    for i in range(3):
        columns.append(((values>>(7*i))&127)|np.where(values>>(7*(i+1)),128,0))
        valid.append(np.ones_like(values,dtype=bool) if i==0 else values>=(1<<(7*i)))
    assert (values<2**21).all()
    return np.stack(columns,axis=-1).astype(np.uint8)[np.stack(valid,axis=-1)].tobytes()

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=OUT)
    OUT=parser.parse_args().directory
    manifest=json.loads((OUT/'manifest.json').read_text())
    for sample in manifest['samples']:
        idx=sample['idx'];path=OUT/sample['file']
        text=path.read_text(); data=json.loads(text[text.index(']=')+2:].removesuffix(';\n'))
        if 'ranks' not in data:
            continue
        rankdir=OUT/f'ranks_{idx}';rankdir.mkdir(exist_ok=True)
        chunks={};mapping={};total=0
        for number,(tid,encoded) in enumerate(data['ranks'].items()):
            array=np.frombuffer(gzip.decompress(base64.b64decode(encoded)),dtype='<u4').reshape(5,data['P'],36,2).astype(np.int32)
            transformed=array.copy();transformed[...,1]-=transformed[...,0]
            deltas=transformed[1:]-transformed[0]
            zigzag=(deltas<<1)^(deltas>>31)
            payload=varints(transformed[0])+varints(zigzag)
            body=base64.b64encode(gzip.compress(payload,compresslevel=9,mtime=0)).decode()
            chunk=number//32
            chunks.setdefault(chunk,{})[f'{idx}/{tid}']=body
            mapping[tid]=chunk
        for chunk,items in chunks.items():
            file=rankdir/f'{chunk}.js'
            file.write_text('Object.assign(window.FIGURE5_RANKS,'+json.dumps(items,separators=(',',':'))+');\n')
            total+=file.stat().st_size
        del data['ranks'];data['rank_chunks']=mapping;data['rank_codec']='gzip-varint-delta-v1'
        path.write_text('window.FIGURE5_CASES=window.FIGURE5_CASES||{};window.FIGURE5_CASES['+str(idx)+']='+json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')+';\n')
        sample.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rank_bytes=total,rank_chunks=len(chunks),rank_codec=data['rank_codec'])
        print(json.dumps(dict(idx=idx,metadata_bytes=path.stat().st_size,rank_bytes=total)),flush=True)
    manifest['rank_storage']='Dense lower rank and tie width + signed sparse deltas; exact lossless varint/gzip, loaded locally on demand'
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
