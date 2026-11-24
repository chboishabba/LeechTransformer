"""Independent public task output audit: exact class action, distribution and order controls."""
import torch
from scripts.audit_x369_ordered_signed_task import endpoint, TRIADS, gram

GRAMS=tuple(sorted(set(map(gram,TRIADS))))
LABELS=(TRIADS,GRAMS,TRIADS)
KEYS=('mean_label','gram_label','signed_label')

def _key(r,word=None,d=None):
    return (r['word'] if word is None else word,r['guards'],r['mean'],r['residual'] if d is None else d)

def permutations(token,g):
    z=(0,0,0); rep={gram(d):d for d in TRIADS}
    return ([TRIADS.index(endpoint(token,g,m,z)[0]) for m in TRIADS],
            [GRAMS.index(endpoint(token,g,z,rep[q])[2]) for q in GRAMS],
            [TRIADS.index(endpoint(token,g,z,d)[1]) for d in TRIADS])

def audit_predictions(rows,logits):
    if len(logits)!=3 or any(p.shape!=(len(rows),len(L)) or not torch.isfinite(p).all() for p,L in zip(logits,LABELS)):
        raise ValueError('finite joint-class logits for all ordered task rows required')
    probs=[p.detach().double().cpu().softmax(-1) for p in logits]; pred=[p.argmax(-1) for p in probs]
    lookup={_key(r):i for i,r in enumerate(rows)}
    if len(lookup)!=len(rows):raise ValueError('duplicate task row')
    total=0; defects=[0]*3; l1=[0.]*3; separated=[0]*3; correct=[0]*3
    # Compare one-token extensions with same original guard assignment and initial registers.
    # Compute only independent task action, never inspect learner internals.
    cache={}
    for j,r in enumerate(rows):
        if len(r['word'])<2:continue
        i=lookup.get(_key(r,word=r['word'][:-1]))
        if i is None:continue
        k=(r['word'][-1],r['guards']); ps=cache.setdefault(k,permutations(*k)) if k not in cache else cache[k]
        total+=1
        for n,p in enumerate(ps):
            defects[n]+=int(p[int(pred[n][i])]!=int(pred[n][j]))
            l1[n]+=float((probs[n][i]-probs[n][j][p]).abs().sum())
    for a,b in (('aj','ja'),('ajaj','aajj'),('ajb','bja')):
        for i,r in enumerate(rows):
            if r['word']!=a:continue
            j=lookup.get(_key(r,word=b))
            if j is None:continue
            for n,(L,k) in enumerate(zip(LABELS,KEYS)):
                x,y=L.index(r[k]),L.index(rows[j][k]);
                if x!=y:
                    separated[n]+=1;correct[n]+=int(int(pred[n][i])==x and int(pred[n][j])==y)
    return dict(prefix_extension_pairs=total,hard_action_defects=defects,distribution_l1_mean=[v/total if total else None for v in l1],ordered_separation_pairs=separated,both_order_endpoints_correct=correct,learned_hidden_action_established=False)
