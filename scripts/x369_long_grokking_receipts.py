"""X369 source-identified full optimizer/data/RNG replay and cheap step ledger.

Opt-in instrument for the upstream Leech train loop, not its replacement.
Caller supplies real data cursor and sampler RNG; without those, complete
same-next-batch replay is not established. Full Gram/SVD mechanism probes
are scheduled at powers of two, not materialized at each optimizer step.
"""
from __future__ import annotations
import hashlib,json,math,os,random
from pathlib import Path
import numpy as np
import torch

def source_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def valid_sha(h):
    return isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h)

def checked_identity(identity,cursor):
    required=('model_sha256','tokenizer_sha256','train_data_sha256','code_revision')
    if any(k not in identity for k in required) or any(
        not valid_sha(identity[k]) for k in required[:3]) or not identity['code_revision']:
        raise ValueError('original model/tokenizer/data/code receipts required')
    if not isinstance(cursor,dict) or not cursor or not all(
        isinstance(k,str) for k in cursor):
        raise ValueError('nonempty original data cursor receipt required')
    try:
        clean=json.loads(json.dumps(cursor,sort_keys=True,allow_nan=False))
    except (TypeError,ValueError) as e:
        raise ValueError('original cursor must have JSON-safe data') from e
    return dict(identity),clean

def capture_rng(*,require_gpu=False):
    py=random.getstate()
    name,arr,pos,has_gauss,cached=np.random.get_state()
    snap={'python':py,
          'numpy':[name,arr.tolist(),int(pos),int(has_gauss),float(cached)],
          'torch_cpu':torch.get_rng_state()}
    if require_gpu:
        if not torch.cuda.is_available():
            raise ValueError('GPU RNG requested without available native accelerator')
        snap['torch_cuda']=torch.cuda.get_rng_state_all()
    return snap

def restore_rng(snap):
    random.setstate(tuple(snap['python']))
    name,arr,pos,gauss,cached=snap['numpy']
    np.random.set_state((name,np.asarray(arr,dtype=np.uint32),pos,gauss,cached))
    torch.set_rng_state(snap['torch_cpu'])
    if 'torch_cuda' in snap:
        if not torch.cuda.is_available():
            raise ValueError('cannot restore GPU RNG without accelerator')
        torch.cuda.set_rng_state_all(snap['torch_cuda'])

def is_full_mechanism_step(step,first=128):
    if type(step)!=int or step<1 or type(first)!=int or first<1 or first & (first-1):
        raise ValueError('positive step and power-of-two initial probe required')
    return step>=first and (step & (step-1))==0

class GrokkingStepLedger:
    def __init__(self,path,run_identity):
        self.path=Path(path)
        if not isinstance(run_identity,dict) or any(
            not valid_sha(run_identity.get(k)) for k in
            ('model_sha256','tokenizer_sha256','train_data_sha256')):
            raise ValueError('step ledger needs complete source identity')
        self.run_identity=dict(run_identity)
        self.last_step=-1
        self.last_token_count=-1
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():continue
                old=json.loads(line)
                if (old['source_identity']!=self.run_identity or
                    old['step']<=self.last_step or
                    old['token_count']<=self.last_token_count):
                    raise ValueError('existing step log source or monotonicity mismatch')
                self.last_step=old['step']
                self.last_token_count=old['token_count']

    def record(self,step,*,token_count,train_ce,learning_rate,grad_norm,
               router_entropy=None,signed_history_energy=None):
        if (type(step)!=int or step<=self.last_step or
            type(token_count)!=int or token_count<=self.last_token_count):
            raise ValueError('optimizer step and token count must strictly increase')
        data=dict(step=step,token_count=token_count,train_ce=train_ce,
                  learning_rate=learning_rate,grad_norm=grad_norm,
                  router_entropy=router_entropy,
                  signed_history_energy=signed_history_energy)
        if any(x is not None and (isinstance(x,bool) or
                                 not isinstance(x,(int,float)) or
                                 not math.isfinite(x)) for x in data.values()):
            raise ValueError('nonfinite or unsupported per-step measurement')
        data['source_identity']=self.run_identity
        data['full_mechanism_probe_due']=is_full_mechanism_step(step)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(data,sort_keys=True,allow_nan=False)+'\n')
        self.last_step=step
        self.last_token_count=token_count
        return data

def save_exact_replay(path,*,step,model,optimizer,scheduler=None,scaler=None,
                      data_cursor,source_identity,require_gpu_rng=False):
    if type(step)!=int or step<0:
        raise ValueError('invalid source optimizer step')
    ident,cursor=checked_identity(source_identity,data_cursor)
    if any(p.is_cuda for p in model.parameters()) and not require_gpu_rng:
        raise ValueError('original GPU model must capture GPU RNG')
    path=Path(path)
    if path.exists():
        raise FileExistsError('refuse overwrite replay checkpoint')
    state=dict(schema=1,kind='x369_original_training_full_replay',
               optimizer_step=step,source_identity=ident,data_cursor=cursor,
               model_state_dict=model.state_dict(),
               optimizer_state_dict=optimizer.state_dict(),
               scheduler_state_dict=scheduler.state_dict() if scheduler else None,
               amp_scaler_state_dict=scaler.state_dict() if scaler else None,
               rng=capture_rng(require_gpu=require_gpu_rng),
               F5_reserved_source_opened=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    if temp.exists():
        raise FileExistsError('stale temporary source checkpoint')
    try:
        torch.save(state,temp)
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()
    return dict(path=str(path),sha256=source_sha(path),step=step)

def load_exact_replay(path,*,model,optimizer,scheduler=None,scaler=None,
                      expected_source_identity,expected_sha256=None):
    if expected_sha256 is not None and source_sha(path)!=expected_sha256:
        raise ValueError('original full replay file hash mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('kind')!='x369_original_training_full_replay':
        raise ValueError('not an original full-replay checkpoint')
    if state.get('source_identity')!=expected_source_identity:
        raise ValueError('original checkpoint source identity mismatch')
    checked_identity(state['source_identity'],state['data_cursor'])
    if any(p.is_cuda for p in model.parameters()) and 'torch_cuda' not in state['rng']:
        raise ValueError('GPU original replay missing GPU RNG')
    if (scheduler is None)!=(state.get('scheduler_state_dict') is None):
        raise ValueError('scheduler provenance missing or mismatched')
    if (scaler is None)!=(state.get('amp_scaler_state_dict') is None):
        raise ValueError('AMP grad-scaler provenance missing or mismatched')
    model.load_state_dict(state['model_state_dict'],strict=True)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler:scheduler.load_state_dict(state['scheduler_state_dict'])
    if scaler:scaler.load_state_dict(state['amp_scaler_state_dict'])
    restore_rng(state['rng'])
    return state['optimizer_step'],state['data_cursor']
