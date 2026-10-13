"""CPU supervision, periodic summaries and stage failure reporting."""
import json
import subprocess
import time
from support import HERE,read,save
from summarize import summarize


def main():
    ledger=read(HERE/'submitted.json');started=time.time()
    while time.time()-started < 46*3600:
        try:
            status=summarize()
            status['updated_unix_seconds']=time.time()
            status['jobs']=ledger
            status['failure_files']=[str(p) for p in HERE.glob('failure-*.json')]
            scheduler=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ledger[k] for k in ['proof','history','formal']),
                '--format=JobID,State,ExitCode'],universal_newlines=True,timeout=60)
            status['scheduler_failed']=[line for line in scheduler.splitlines() if any(x in line for x in ['FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','CANCELLED','DEPENDENCY'])]
            save(HERE/'latest-status.json',status)
            print(json.dumps({k:status[k] for k in ['status','samples_completed','workers_completed','failure_files','scheduler_failed']}),flush=True)
            if status['status']=='complete':return
            if status['failure_files'] or status['scheduler_failed']:
                status['status']='needs_inspection';save(HERE/'latest-status.json',status);return
        except Exception as error:
            save(HERE/'monitor-error.json',dict(error=repr(error),updated_unix_seconds=time.time()))
        time.sleep(300)

if __name__=='__main__':main()
