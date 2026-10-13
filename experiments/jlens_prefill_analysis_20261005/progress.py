"""Periodic progress saves and rolling immutable checkpoint hardlinks."""
import contextlib
import datetime
import os
from pathlib import Path
import threading

from common import config,read,save,sha


class PeriodicSaver:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self.loop,daemon=True)

    def snapshot(self):
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        snapshot_dir=self.directory/'periodic-saves'
        snapshot_dir.mkdir(parents=True,exist_ok=True)
        files={}
        for p in self.directory.glob('*.json'):
            if '.tmp.' not in p.name: files[p.name]=sha(p)
        checkpoint=self.directory/'checkpoint.pt'
        if checkpoint.exists():
            link=snapshot_dir/('checkpoint-'+stamp+'.pt')
            os.link(checkpoint,link)  # fit writes by atomic replace; the link stays immutable
            files['checkpoint']=dict(path=link.name,sha256=sha(link),bytes=link.stat().st_size)
            for old in sorted(snapshot_dir.glob('checkpoint-*.pt'))[:-2]: old.unlink()
        save(snapshot_dir/'latest.json',dict(saved_at_utc=stamp,files=files,
            checkpoint_retention='latest two hardlinks; JSON result records remain immutable'))

    def loop(self):
        while not self.stop.wait(config()['snapshot_seconds']):
            try: self.snapshot()
            except Exception as exc: save(self.directory/'snapshot-error.json',dict(error=repr(exc)))

    def __enter__(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self,*exc):
        self.stop.set()
        self.thread.join()
        self.snapshot()
