"""CPU-only immutable preparation for the new PSC experiment; no model forward."""
from pathlib import Path
import hashlib, json, struct
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
MODEL=Path('/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/models/Llama-3.1-8B-Instruct')
REV='90f0394333616266d9fe85824ceaf505093cbaa5'

def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(r): return json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def main():
    path=ROOT/'data/kv_retrieval.jsonl'
    if not path.exists(): (path.with_suffix('.jsonl.part')).rename(path)
    filehash=hashlib.sha256(); selected=[]; count=0
    with path.open('rb') as stream:
        for index,line in enumerate(stream):
            filehash.update(line)
            if not line.strip(): raise ValueError('blank JSONL record')
            row=json.loads(line)
            if not isinstance(row.get('context'),str) or not isinstance(row.get('input'),str) or not isinstance(row.get('answer'),list):
                raise ValueError('unexpected record schema')
            if index<4: selected.append((index,row,sha(line),sha(canonical(row))))
            count+=1
    if count != 500: raise ValueError(f'expected original 500 rows, got {count}')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    out=ROOT/'inputs';out.mkdir(exist_ok=False)
    records=[]; calibration=[]
    for index,row,rawhash,recordhash in selected:
        prompt='Extract the value corresponding to the specified key in the JSON object below.\n\n'+row['context']+'\n\n'+row['input']
        rendered=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        ids=tok.encode(rendered,add_special_tokens=False)
        before=len(ids); limit=131072-128
        if len(ids)>limit: ids=ids[:limit//2]+ids[-(limit-limit//2):]
        payload=struct.pack('<'+'q'*len(ids),*ids)
        (out/f'{index}.ids.i64le').write_bytes(payload)
        item={'source_index':index,'role':'calibration' if index<3 else 'test','raw_line_sha256_including_newline':rawhash,'canonical_record_sha256':recordhash,'input_tokens_before_truncation':before,'input_tokens':len(ids),'input_ids_sha256':sha(payload),'token_file':f'{index}.ids.i64le'}
        if index<3:
            # Original AE extractor deliberately uses non-chat prefix length1936.
            ae=tok(prompt,truncation=True,max_length=1936,padding=False)['input_ids']
            if len(ae)!=1936: raise ValueError('calibration map length mismatch')
            ap=struct.pack('<'+'q'*len(ae),*ae)
            (out/f'{index}.ae_ids.i64le').write_bytes(ap)
            item.update(ae_input_tokens=len(ae),ae_input_ids_sha256=sha(ap))
            calibration.append({'_id':f'kv_retrieval-{index}','prompt':prompt,'source_index':index,'benchmark':'infinitebench','task':'kv_retrieval'})
        records.append(item)
    (out/'calibration.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in calibration))
    (out/'selected_records.jsonl').write_text(''.join(json.dumps({'source_index':i,'record':r},ensure_ascii=False)+'\n' for i,r,_,_ in selected))
    manifest={'dataset_revision':REV,'data_sha256':filehash.hexdigest(),'records_validated':count,'records_tokenized':4,'selection_fixed_before_model_outputs':True,'calibration_indices':[0,1,2],'test_indices':[3],'input_hash_rule':'SHA256 contiguous little-endian signed int64 token IDs, no header','record_hash_rule':'UTF-8 JSON sort_keys=True ensure_ascii=False separators=(comma,colon)','maxcontext':131072,'reserved_generation_tokens':128,'truncation':'first floor(130944/2) plus last remainder','batch_size':1,'seed':42,'ae_rule':'original non-chat prompt prefix1936, tokenizer default special tokens; independent from full chat IDs','records':records,'tokenizer_files':{p.name:sha(p.read_bytes()) for p in MODEL.iterdir() if p.suffix in ('.json','.jinja') and p.name!='model.safetensors.index.json'}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'count':count,'records':records},indent=2))
if __name__=='__main__':main()
