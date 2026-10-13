"""Submission is enabled only after approval matches the frozen concrete plan."""
import subprocess
from support import HERE,gate,read,save,sha


def main():
    p=gate(allocation=False)
    assert read(HERE/'cpu-validation.json')['status']=='passed'
    assert read(HERE/'cpu-validation.json')['lock_sha256']==sha(HERE/'lock.json')
    assert not (HERE/'submitted.json').exists()
    with (HERE/'submit-intent.json').open('x') as stream:stream.write('{}\n')
    ledger={}
    def submit(name,args):
        job=subprocess.check_output(['sbatch','--parsable']+args,universal_newlines=True).strip().split(';')[0]
        ledger[name]=job;save(HERE/'submitted.json',dict(ledger));return job
    proof=submit('proof',[str(HERE/'gpu.sbatch'),'proof'])
    history=submit('history',['--array=0-6','--dependency=afterok:'+proof,'--kill-on-invalid-dep=yes',str(HERE/'gpu.sbatch'),'history'])
    submit('formal',['--array=0-159','--dependency=afterok:'+history,'--kill-on-invalid-dep=yes',str(HERE/'gpu.sbatch'),'formal'])
    parent=submit('monitor',[str(HERE/'monitor.sbatch')])
    for i in range(3):parent=submit('monitor_successor_%d'%i,['--dependency=afterany:'+parent,str(HERE/'monitor.sbatch')])
    print(ledger)

if __name__=='__main__':main()
