"""Original public task consumer quotient: geometry-first, no source bypass.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
This module only proves the declared task's algebraic consumer boundary.
It does NOT claim original Torch recurrence has learned this semantics.
"""
import json
from collections import defaultdict
from itertools import product

from scripts.audit_x369_ordered_signed_task import GUARDS,TRIADS,endpoint,gram
from scripts.audit_x369_joint_action import MEANS
from scripts.x369_consumer_automata import compile_observers,execute


def squared_margin(v,labels,metric):
    """Exact unnormalized nearest-prototype score, no float/GPU."""
    scores=[2*sum(w*x*y for w,x,y in zip(metric,v,p))
            -sum(w*y*y for w,y in zip(metric,p)) for p in labels]
    best=max(scores)
    if scores.count(best)!=1:raise ValueError('ambiguous exact prototype')
    return scores.index(best),best-max(s for s in scores if s!=best)


def compile_first_memory():
    controls=('a','b','j','n')
    def latch(r,u):
        if r is not None:return r
        return u if u in ('a','b') else None
    receipt,stats=compile_observers(
        (None,),controls,lambda r,u:(r,latch(r,u)),lambda r:r)
    if stats['classes']!=3:raise AssertionError('three-state latch quotient')
    return receipt,stats


def certificate(suffixes=('', 'a','j','n','jn','abjn')):
    receipt,stats=compile_first_memory()
    observed=defaultdict(lambda:[0,0])
    checked=0
    for guards,mean,residual,suffix in product(GUARDS,MEANS,TRIADS,suffixes):
        w1='ab'+suffix;w2='ba'+suffix
        m1,d1,g1=endpoint(w1,guards,mean,residual)
        m2,d2,g2=endpoint(w2,guards,mean,residual)
        if (m1,d1,g1)!=(m2,d2,g2):
            raise AssertionError('commuting semantic endpoint witness failed')
        # Labels differ even when guards, initial state and endpoint agree.
        key=(guards,mean,residual,suffix,m1,d1,g1)
        observed[key][0]+=1
        observed[key][1]+=1
        a,_=execute(receipt,0,tuple(w1))
        b,_=execute(receipt,0,tuple(w2))
        if a==b:raise AssertionError('minimal latch erased order')
        checked+=1
    correct=sum(max(counts) for counts in observed.values())
    total=sum(map(sum,observed.values()))
    assert total==2*checked and correct*2==total

    mean_labels=TRIADS
    gram_labels=tuple(sorted(set(map(gram,TRIADS))))
    min_m=min_d=min_g=None
    for v in TRIADS:
        i,margin=squared_margin(v,mean_labels,(1,1,1))
        assert mean_labels[i]==v
        _,signed=squared_margin(v,mean_labels,(1,1,1))
        g=gram(v)
        j,quadratic=squared_margin(g,gram_labels,(1,1,1,2,2,2))
        assert gram_labels[j]==g
        min_m=margin if min_m is None else min(min_m,margin)
        min_d=signed if min_d is None else min(min_d,signed)
        min_g=quadratic if min_g is None else min(min_g,quadratic)
    return dict(scope='public signed finite task, not native LILA recurrence',
       exact_history_pairs=checked,balanced_rows=total,
       semantic_only_best=correct,semantic_only_ceiling='1/2',
       exact_latch_classes=stats['classes'],exact_latch_class_bits=stats['state_id_bits'],
       minimum_prototype_margin=dict(mean=min_m,gram=min_g,signed=min_d),
       all_prototype_labels_exact=True,
       learned_residual_required_for_this_consumer=False,
       original_sparse_consumer_sufficient=False,
       source_checkpoint_authenticated=False,
       GPU_acceptance=False,Agda_kernel_acceptance=False)


if __name__=='__main__':print(json.dumps(certificate(),sort_keys=True))
