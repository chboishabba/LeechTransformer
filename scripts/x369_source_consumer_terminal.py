"""Terminal X369 source-owned future-consumer admission.

Uses original native source callbacks and the existing simultaneous-weld owner.
Does not infer source-sufficiency from public labels or a supplied action register.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from dataclasses import dataclass
from scripts.x369_simultaneous_weld import native_simultaneous,verify_joint_realization


def adjudicate(result):
    """Conservative finite-source verdict. Closed means finite *declared graph*.
    Only an original-source replay can authorize a positive result.
    """
    f=result['future']
    if result.get('semantic_collision') is not None:
        return dict(status='negative',reason='same fine state, different semantic labels',
                    witness=result['semantic_collision'],universal_source_safe=False)
    if result.get('constructed',{}).get('edge_defects',0) or result.get('constructed',{}).get('seed_mismatches',0):
        return dict(status='negative',reason='actual source semantic edge or seed defect',
                    witness=result.get('constructed',{}).get('first_defect'),
                    universal_source_safe=False)
    if f['status']!='closed':
        # An observed direct separator may refute the proposed chart, but
        # a finite prefix never certifies a positive quotient.
        sep=f.get('observed_direct_separator')
        return dict(status='negative' if sep else 'incomplete',
                    reason='bounded source future search',
                    witness=sep,universal_source_safe=False,
                    budget_exhausted=f['status'] in ('state_budget','time_budget'))
    if not result.get('semantic_table_certified'):
        return dict(status='incomplete',reason='source semantic table not closed',
                    universal_source_safe=False)
    if not result.get('linear_decoder_certified'):
        return dict(status='negative',reason='source semantic labels lack admitted linear decoder',
                    witness=result.get('decoder_fit',{}).get('conflict'),
                    universal_source_safe=False)
    if not result.get('constructed_exact_on_closed_graph'):
        return dict(status='incomplete',reason='constructed source decoder not proved on closed graph',
                    universal_source_safe=False)
    if not result.get('readout_unique_correct_on_closed_graph'):
        return dict(status='negative',reason='original source readout fails declared consumer on closed graph',
                    witness=result.get('readout_path_counts'),
                    universal_source_safe=False)
    if 'joint_realization' not in result:
        return dict(status='incomplete',reason='source-bound quotient not exported for independent replay',
                    universal_source_safe=False)
    return dict(status='pending_replay',reason='all recorded finite producers pass; independent original source replay required',
                universal_source_safe=False,closed_graph_only=True)


def admit_native(model,controls,*,budget,seconds=90):
    """Needs the caller's authenticated original-source checkpoint and device.
    No reconstruction of the original neural model from a public finite task.
    """
    from scripts.x369_source_future_acceptance import native_future_contract
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.x369_simultaneous_weld import literal_matrix,apply_matrix
    from scripts.audit_x369_joint_action import MEANS
    from scripts.audit_x369_ordered_signed_task import TRIADS
    from itertools import product
    from scripts.x369_capped_semantic_geometry import exact_repair,cap_profile
    from fractions import Fraction as Q
    import torch
    if model.training or model.typed_consumer:
        raise ValueError('raw eval-only source required for native simultaneous weld')
    controls=tuple(controls)
    if not controls or len(set(controls))!=len(controls):raise ValueError('unique controls required')
    for g,t in controls:literal_matrix((g,t))
    source_seeds,step,chart=native_future_contract(model,'raw')
    r=native_simultaneous(model,controls,budget=budget,seconds=seconds,export=True)
    decision=adjudicate(r)
    if decision['status']!='pending_replay':
        return dict(decision=decision,raw=r)
    seed,_,native_action=native_source(model)
    seeds=tuple((seed(m,d),m+d,i) for i,(m,d) in enumerate(product(MEANS,TRIADS)))
    assert tuple(dict.fromkeys(h for h,_,_ in seeds))==source_seeds
    matrices={u:literal_matrix(u) for u in controls}
    independent_action=lambda u,z:apply_matrix(matrices[u],z)
    v=verify_joint_realization(r['joint_realization'],seeds,controls,chart,step,independent_action)
    return dict(decision=dict(status='finite_source_admitted',
             reason='source callbacks independently replayed on full declared reachable finite graph',
             declared_finite_graph=True,unrestricted_neural_source=False,
             native_floating_error_certified=False,
             original_checkpoint_authentication_external=True,
             source_future_minimal=True,source_readout_correct=True,
             edges=v['edges'],states=v['states'],classes=v['classes']),
         raw=r)


def certificate():
    """Synthetic verdict contract only; source replay must use admit_native."""
    base=dict(future=dict(status='closed',observed_direct_separator=None),
              semantic_collision=None,
              constructed=dict(edge_defects=0,seed_mismatches=0,first_defect=None),
              semantic_table_certified=True,linear_decoder_certified=True,
              constructed_exact_on_closed_graph=True,
              readout_unique_correct_on_closed_graph=True,
              joint_realization=dict())
    assert adjudicate(base)['status']=='pending_replay'
    x=dict(base,semantic_collision={'fine':'collision'})
    assert adjudicate(x)['status']=='negative'
    x=dict(base,constructed=dict(edge_defects=1,seed_mismatches=0,first_defect={'edge':1}))
    assert adjudicate(x)['status']=='negative'
    x=dict(base,future=dict(status='state_budget',observed_direct_separator=None))
    assert adjudicate(x)['status']=='incomplete'
    x=dict(base,future=dict(status='state_budget',observed_direct_separator={'left':0,'right':1}))
    assert adjudicate(x)['status']=='negative'
    x=dict(base,linear_decoder_certified=False)
    assert adjudicate(x)['status']=='negative'
    x=dict(base,readout_unique_correct_on_closed_graph=False)
    assert adjudicate(x)['status']=='negative'
    x=dict(base);x.pop('joint_realization')
    assert adjudicate(x)['status']=='incomplete'
    return dict(verdict_controls=8,synthetic_only=True,native_admission_executed=False,
                optimizer_steps=0,agda_kernel_checked=False)
if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True))
