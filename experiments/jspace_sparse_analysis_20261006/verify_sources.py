from common import HERE,save,sha,verify
import time

def main():
    m=verify();start=time.time()
    for i,row in enumerate(m['records']):
        if sha(row['artifact'])!=row['artifact_sha256']:raise ValueError('Changed activation: '+row['artifact'])
        if (i+1)%100==0:
            save(HERE/'source-verification-progress.json',dict(verified=i+1,total=1400,seconds=time.time()-start))
            print('VERIFIED',i+1,flush=True)
    for p,expected in __import__('json').load(open(__import__('pathlib').Path(m['source'])/'input-manifest.json'))['model_files'].items():
        if sha(p)!=expected['sha256']:raise ValueError('Model content changed: '+p)
    if sha(m['lens'])!=m['lens_sha256']:raise ValueError('Lens content changed')
    save(HERE/'source-verification.json',dict(status='PASS',artifacts=1400,lens_sha256=m['lens_sha256'],model_files_sha256_verified=True))
if __name__=='__main__':main()
