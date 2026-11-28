"""Bounded local driver for an EXISTING registered X369 GPU campaign.

No training owner changes. Each child is the exact registered scheduler,
one checkpointed chunk, hard subprocess deadline <=90 seconds. This
driver never edits registration or recovers orphan partial files by
deleting them; stop on any failed/incomplete child invocation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def require(ok,msg):
    if not ok:raise ValueError(msg)


def command(a):
    p=json.loads((a.directory/'registration.json').read_text())
    require(p['runner_device']=='gfx803' and p['heldout_selection'] is False and len(p['arms'])==18,
            'existing native fixed registration required')
    c=p['arms'][0]['contract']
    keys=('width','experts','dtype','horizon','report_every','lr','weight_decay')
    require(all(all(e['contract'][k]==c[k] for k in keys) for e in p['arms']),
            'registered arm scope differs')
    args=[sys.executable,'-m','scripts.run_x369_registered_learning_campaign',
          '--directory',str(a.directory.resolve()),'--device','gfx803',
          '--steps',str(c['horizon']),'--width',str(c['width']),
          '--experts',str(c['experts']),'--dtype',c['dtype'],
          '--report-every',str(c['report_every']),'--lr',str(c['lr']),
          '--weight-decay',str(c['weight_decay']),
          '--chunk-steps',str(a.chunk_steps),'--seconds',str(a.seconds)]
    return args


def run(a):
    require(1<=a.chunks<=360 and a.chunk_steps>0 and 0<a.seconds<a.hard_timeout<=90,
            'bounded local chunk scope')
    cmd=command(a)
    for i in range(a.chunks):
        try:
            r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=a.hard_timeout)
        except subprocess.TimeoutExpired as exc:
            # Immutable checkpoint protocol is owned by the scheduler.
            # Never silently delete a .partial or resume a half receipt.
            raise RuntimeError('chunk hard deadline; inspect existing immutable checkpoint/receipt files before resuming') from exc
        if r.returncode:
            raise RuntimeError(f'chunk stopped (exit {r.returncode}): {r.stderr[-8000:]}')
        v=json.loads(r.stdout)
        require(v['status'] in ('CAMPAIGN_IN_PROGRESS','FIXED_CAMPAIGN_COMPLETE') and
                v['arms_required']==18 and v['heldout_selection'] is False,'unexpected campaign receipt')
        print(json.dumps(dict(invocation=i+1,status=v['status'],arms_complete=v['arms_complete'],
            completed_steps=sum(x['completed_steps'] for x in v['arms']),
            registered_total_steps=18*int(cmd[cmd.index('--steps')+1])),
            flush=True)
        if v['status']=='FIXED_CAMPAIGN_COMPLETE':return


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--chunks',type=int,default=18);p.add_argument('--chunk-steps',type=int,default=100)
    p.add_argument('--seconds',type=float,default=80);p.add_argument('--hard-timeout',type=float,default=90)
    run(p.parse_args())

if __name__=='__main__':main()
