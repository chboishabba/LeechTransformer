"""Structural C3 task input/action checker on ORIGINAL token ID arrays.

A verified structural token action is NOT independent evidence that those
labels encode a meaningful language task, that the model's logits transform,
or that an autonomous C3 circuit was learned.
"""
from __future__ import annotations
import numpy as np

RULE='cycle_three_token_positions_v1'

def cyclic_token_action(tokens,positions):
    """Apply right cyclic shift at three explicit token positions."""
    x=np.asarray(tokens)
    if x.ndim!=1 or not np.issubdtype(x.dtype,np.integer):
        raise ValueError('single integer original-token input required')
    if not isinstance(positions,(list,tuple)) or len(positions)!=3 or (
        not all(type(p) is int for p in positions) or len(set(positions))!=3
        or any(p<0 or p>=len(x) for p in positions)
    ):
        raise ValueError('three distinct in-range exact token positions required')
    a,b,c=positions
    rotated=x.copy()
    rotated[a],rotated[b],rotated[c]=x[c],x[a],x[b]
    return rotated

def certify_orbit_action(fit,evaluation,manifest):
    spec=manifest.get('machine_action_certificate')
    if spec is None:
        return {'machine_action_structurally_verified':False,
                'semantic_label_correctness_independently_verified':False,
                'reason':'no executable C3 action/label certificate'}
    if not isinstance(spec,dict) or spec.get('rule')!=RULE or (
        spec.get('output_rule')!='plus_one_mod_three_v1'
    ):
        raise ValueError('unsupported or untyped C3 input/output action')
    pos=spec.get('token_positions')
    all_groups=[('fit',np.asarray(fit),spec.get('fit_phase_labels')),
                ('evaluation',np.asarray(evaluation),spec.get('eval_phase_labels'))]
    evaluated=0
    for name,groups,labels in all_groups:
        if groups.ndim!=3 or groups.shape[1]!=3 or (
            not np.issubdtype(groups.dtype,np.integer)
        ):
            raise ValueError('typed three-action original-token groups required')
        if not isinstance(labels,list) or len(labels)!=len(groups):
            raise ValueError('source phase labels missing for '+name)
        for j,group in enumerate(groups):
            base=group[0]
            if not (np.array_equal(cyclic_token_action(base,pos),group[1])
                    and np.array_equal(cyclic_token_action(group[1],pos),group[2])
                    and np.array_equal(cyclic_token_action(group[2],pos),base)):
                raise ValueError('original-token C3 action does not close: '+name)
            if np.array_equal(base,group[1]):
                raise ValueError('trivial token orbit cannot establish nontrivial action')
            y=labels[j]
            if not isinstance(y,list) or len(y)!=3 or (
                not all(type(v) is int and 0<=v<3 for v in y)
            ):
                raise ValueError('three typed output phase labels required')
            if y[1]!=(y[0]+1)%3 or y[2]!=(y[1]+1)%3:
                raise ValueError('C3 output phase action violates group law')
            evaluated+=1
    return {'machine_action_structurally_verified':True,
            'rule':RULE,'input_action_order':3,
            'output_action_rule':'plus_one_mod_three_v1',
            'original_token_slots':list(pos),
            'source_orbit_groups_checked':evaluated,
            'semantic_label_correctness_independently_verified':False,
            'original_model_output_consumer_equivariance_verified':False,
            'learned_original_model_c3_circuit_proved':False}
