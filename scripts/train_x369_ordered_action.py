"""Companion to the existing typed mod3 learner; original Leech-LILA: Anatolii Kornienko.
Exact learned linear control plus GPU-first ordered source learner. No F5 data.
"""
from collections import defaultdict
from itertools import product
import argparse,json,random,time
from fractions import Fraction as F
import torch
from torch import nn
from torch.nn import functional as NF
from scripts.audit_x369_ordered_signed_task import records,TRIADS,gram,endpoint

MODES=('mean','support','gram','signed')
GRAMS=tuple(sorted(set(map(gram,TRIADS))))
LABELS=(TRIADS,GRAMS,TRIADS)
HEADS=tuple(map(len,LABELS))

def generator_fit(rows):
    """Exact least squares from elementary training labels only. Normal equations."""
    sums=defaultdict(lambda:[[0]*3 for _ in range(3)]); cov=defaultdict(lambda:[[0]*3 for _ in range(3)])
    for r in rows:
        if r['split']!='train' or len(r['word'])!=1: continue
        token=r['word']; key=(token,r['guards']['abjn'.index(token)])
        for i,j in product(range(3),repeat=2):
            sums[key][i][j]+=r['signed_label'][i]*r['residual'][j]
            cov[key][i][j]+=r['residual'][i]*r['residual'][j]
    result={}
    for k,c in cov.items():
        n=c[0][0]; assert n>0 and c==[[n*int(i==j) for j in range(3)] for i in range(3)]
        result[k]=tuple(tuple(F(v,n) for v in row) for row in sums[k])
    assert len(result)==10
    return result

def matvec(A,v): return tuple(sum(a*b for a,b in zip(row,v)) for row in A)

def generator_predict(A,r):
    m,d=r['mean'],r['residual']
    for token in r['word']:
        v=r['guards']['abjn'.index(token)]; W=A[token,v]
        d=matvec(W,d); m=matvec(W,m)
        # Declared task branch-swap character, not learned from mean labels.
        if token=='j' and v: m=tuple(-z for z in m)
    return m,gram(d),d

def encode_inputs(r,mode):
    """Only initial observer packet and chronological literal reads; no endpoint or raw residual bypass."""
    d=r['residual']; p={'mean':(), 'support':tuple(v*v for v in d), 'gram':gram(d), 'signed':d}[mode]
    seed=tuple(r['mean'])+tuple(p)+(0,)*(6-len(p))
    ops=[]
    for ch in r['word']:
        i='abjn'.index(ch);v=r['guards'][i]
        ops.append(tuple(int(i==j) for j in range(4))+tuple(int(v==j) for j in range(3)))
    return seed,tuple(ops)

def batch(rows,mode,device,dtype):
    s,o=zip(*(encode_inputs(r,mode) for r in rows)); assert len(set(map(len,o)))==1
    y=[[L.index(r[k]) for L,k in zip(LABELS,('mean_label','gram_label','signed_label'))] for r in rows]
    return torch.tensor(s,device=device,dtype=dtype),torch.tensor(o,device=device,dtype=dtype),torch.tensor(y,device=device,dtype=torch.long)

class OrderedLearner(nn.Module):
    """Explicit initial-state task adapter to original step or paired step. No skip of input into readout."""
    def __init__(self,width=12,experts=4,paired=False,geometry='dense',source=None):
        super().__init__();self.paired=paired
        if source is None:
            from models.history_orbit_moe import HistoryOrbitMoE
            source=HistoryOrbitMoE(d_model=width,num_experts=experts,top_k=2,expert_width=width,route_geometry='c2_c3_residual')
        self.width=width;self.experts=experts
        if geometry!='dense':
            from models.geometric_cross_triads import CrossTriadBitLinear
            for name,layer in list(source.named_modules()):
                if isinstance(layer,nn.Linear) and name.split('.')[0] in ('experts','transitions'):
                    parent,key=name.rsplit('.',1)
                    setattr(source.get_submodule(parent),key,CrossTriadBitLinear(layer.in_features,layer.out_features,mode=geometry,scale=.25,bias=layer.bias is not None))
        if paired:
            from models.geometric_paired_history import PairedHistoryMoE
            self.history=PairedHistoryMoE(source)
        else:self.history=source
        self.mean_seed=nn.Linear(3,experts*width,bias=False);self.residual_seed=nn.Linear(6,experts*width,bias=False)
        self.embed=nn.Linear(7,width)
        self.readout=nn.Linear((2 if paired else 1)*(experts+1)*width,sum(HEADS))
    def forward(self,seed,ops):
        b=len(seed);m=self.mean_seed(seed[:,:3]).reshape(b,self.experts,self.width);d=self.residual_seed(seed[:,3:]).reshape_as(m)
        h=torch.stack((m+d,m-d),dim=2) if self.paired else m+d
        for t in range(ops.shape[1]):
            x=self.embed(ops[:,t]);
            if self.paired:
                out=self.history.step(x,h);h=out.history;y=out.branches
            else:y,h,*_=self.history.step(x,h,train_sign=self.training)
        return self.readout(torch.cat((y.flatten(1),h.flatten(1)),dim=1)).split(HEADS,dim=-1)

def score(model,groups,mode,device,dtype):
    counts=[0]*3;total=0
    with torch.no_grad():
        for rs in groups.values():
            for start in range(0,len(rs),256):
                s,o,y=batch(rs[start:start+256],mode,device,dtype); pred=model(s,o)
                for i,p in enumerate(pred):counts[i]+=int((p.argmax(-1)==y[:,i]).sum())
                total+=len(y)
    return dict(correct=counts,total=total)

def final_action_audit(model,rows,mode,device,dtype):
    from scripts.audit_x369_ordered_action_predictions import audit_predictions
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[len(r['word'])].append(i)
    logits=[torch.empty(len(rows),n,dtype=torch.float32) for n in HEADS]
    with torch.no_grad():
        for indices in groups.values():
            for start in range(0,len(indices),256):
                ix=indices[start:start+256];s,o,_=batch([rows[i] for i in ix],mode,device,dtype)
                for dst,p in zip(logits,model(s,o)):dst[ix]=p.cpu().float()
    return audit_predictions(rows,logits)

def main():
    p=argparse.ArgumentParser();p.add_argument('--exact-control',action='store_true');p.add_argument('--require-gfx803',action='store_true');p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=41);p.add_argument('--width',type=int,default=12);p.add_argument('--mode',choices=MODES,default='signed');p.add_argument('--paired',action='store_true');p.add_argument('--geometry',choices=('dense','c3','s3','full'),default='dense');p.add_argument('--batch-size',type=int,default=128);a=p.parse_args()
    rows=list(records());A=generator_fit(rows);counts=defaultdict(int)
    for r in rows:
        assert generator_predict(A,r)==(r['mean_label'],r['gram_label'],r['signed_label']);counts[r['split']]+=1
    print(json.dumps(dict(kind='exact_learned_linear_control',generators=len(A),correct=dict(counts),mean_twist_given=True)),flush=True)
    if a.exact_control:return
    if not torch.cuda.is_available():raise RuntimeError('GPU required for neural training; no CPU fallback')
    if a.require_gfx803 and (not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803')):raise RuntimeError('gfx803 HIP required')
    if min(a.steps,a.batch_size)<1 or a.width%3 or a.width<3:raise ValueError('invalid dimensions/horizon')
    torch.manual_seed(a.seed);rng=random.Random(a.seed);dev=torch.device('cuda');dt=torch.float32
    model=OrderedLearner(a.width,4,a.paired,a.geometry).to(dev,dt);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
    groups={k:defaultdict(list) for k in ('train','composition')}
    for r in rows:groups[r['split']][len(r['word'])].append(r)
    print(json.dumps(dict(kind='neural_meta',args=vars(a),parameters=sum(v.numel() for v in model.parameters()),torch=torch.__version__,hip=torch.version.hip,matched_cost_claim=False)),flush=True)
    for t in range(1,a.steps+1):
        model.train();rs=rng.choice(list(groups['train'].values()));rs=rng.choices(rs,k=a.batch_size);s,o,y=batch(rs,a.mode,dev,dt);opt.zero_grad(set_to_none=True);pred=model(s,o);loss=sum(NF.cross_entropy(v,y[:,i]) for i,v in enumerate(pred));loss.backward();opt.step()
        if t==1 or t%100==0 or t==a.steps:
            model.eval();print(json.dumps(dict(step=t,train=score(model,groups['train'],a.mode,dev,dt),composition=score(model,groups['composition'],a.mode,dev,dt),grokking_established=False)),flush=True)
    model.eval();print(json.dumps(dict(kind='final_action_audit',**final_action_audit(model,rows,a.mode,dev,dt))),flush=True)
if __name__=='__main__':main()
