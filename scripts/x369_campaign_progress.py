"""Telemetry only: no changes to training draws, checkpoints or acceptance receipts."""
import json,time,os
from datetime import datetime,timedelta

class CampaignProgress:
    def __init__(self,plan,steps,snapshots,output,interval=10.,clock=time.monotonic):
        self.plan=tuple(plan);self.steps=steps;self.snapshots=snapshots
        self.output=output;self.interval=interval;self.clock=clock
        self.started=clock();self.last=float('-inf')
        self.train={k:0 for k in self.plan};self.audit={k:0 for k in self.plan}
        self.finished=set();self.rates={'train':{},'audit':{}}
        self.observed={};self.stage_started=self.started
    def seed(self,key,step,complete=False):
        self.train[key]=step
        if complete:self.audit[key]=self.snapshots+1;self.finished.add(key)
    def begin(self,key,phase):
        self.stage_started=self.clock()
        self.observed[(key,phase)]=(self.train[key] if phase=='train' else self.audit[key],self.stage_started)
        self.emit(key,phase,force=True)
    def update(self,key,phase,count,force=False,**extra):
        table=self.train if phase=='train' else self.audit
        if count<table[key]:raise ValueError('progress regressed')
        table[key]=count
        prev,t=self.observed.get((key,phase),(count,self.clock()))
        now=self.clock()
        if count>prev and now>t:
            rate=(now-t)/(count-prev)
            old=self.rates[phase].get(key[0])
            self.rates[phase][key[0]]=rate if old is None else .25*rate+.75*old
            self.observed[(key,phase)]=(count,now)
        self.emit(key,phase,force=force,**extra)
    def estimate(self,key,phase,remaining):
        if not remaining:return 0.
        rates=self.rates[phase]
        rate=rates.get(key[0])
        if rate is None and rates:rate=sum(rates.values())/len(rates)
        return None if rate is None else rate*remaining
    def emit(self,key,phase,force=False,**extra):
        now=self.clock()
        if not force and now-self.last<self.interval:return
        self.last=now
        tr=sum(self.train.values());au=sum(self.audit.values())
        train_total=len(self.plan)*self.steps;audit_total=len(self.plan)*(self.snapshots+1)
        def total_eta(phase,table,total):
            values=[self.estimate(k,phase,total-table[k]) for k in self.plan]
            return None if any(v is None for v in values) else sum(values)
        train_eta=total_eta('train',self.train,self.steps)
        audit_eta=total_eta('audit',self.audit,self.snapshots+1)
        eta=None if train_eta is None or audit_eta is None else train_eta+audit_eta
        current=self.estimate(key,'train',self.steps-self.train[key])
        record=dict(kind='campaign_progress',arm=key[0],seed=key[1],phase=phase,
                    arm_index=self.plan.index(key)+1,arms_total=len(self.plan),
                    arms_completed=len(self.finished),step=self.train[key],steps=self.steps,
                    training_steps_completed=tr,training_steps_total=train_total,
                    training_percent=round(100*tr/train_total,2),
                    audit_units_completed=au,audit_units_total=audit_total,
                    arm_audit_units=self.audit[key],arm_audit_total=self.snapshots+1,
                    elapsed_seconds=round(now-self.started,2),
                    arm_training_eta_seconds=current,training_eta_seconds=train_eta,
                    audit_eta_seconds=audit_eta,campaign_eta_seconds=eta,
                    estimated_completion_local=(datetime.now().astimezone()+timedelta(seconds=eta)).isoformat(timespec='seconds') if eta is not None else None,
                    eta_scope='observed phase rates; setup/checkpoint overhead and interruptions may change estimate',
                    **extra)
        print(json.dumps(record,allow_nan=False),flush=True)
        # Latest status is replaceable telemetry, never an authenticated receipt.
        path=self.output/'progress.json';tmp=self.output/f'.progress.{os.getpid()}.tmp'
        tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        os.replace(tmp,path)
    def complete(self,key):
        self.finished.add(key)
        self.emit(key,'arm_complete',force=True)
