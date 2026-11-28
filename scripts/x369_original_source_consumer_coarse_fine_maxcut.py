"""Coarse→fine SOURCE consumer max-cut for original marked 369 fibre.

Reuses x369_indexed_source_fibre_audit's ORIGINAL source graph,
per-field one-step congruence tests, independent collision replay and
existing minimal finite automaton compiler. No replacement learner.

Coarse candidate: only the original 189 signed-source initial seed
mark. Unmarked successor histories deliberately receive one 'unknown'
value, which is a falsifiable lossy-chart hypothesis, never a claim
that unknown learned source states share semantic meaning.

Fine candidate: original byte history, which keeps all information.
Intermediate Stage13 / typed pants projections are only installed when
the original source supplies a callable typed interpreter.

Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from scripts.x369_indexed_source_fibre_audit import fibre_audit,replay_witness
from scripts.x369_original_marked_seed_fabric import mark_original_source_history

def marked_source_chart(marks):
    if not marks['original_seed_marking_injective']:
        raise ValueError('source original marked seed aliasing')
    def chart(h):
        r=mark_original_source_history(h,marks)
        if r['status']=='unmarked_source_history':
            return ('unmarked_original_recurrent_history',)
        return ('original_signed_mark',r['mean'],r['signed_residual'])
    return chart

def source_consumer_maxcut(seeds,controls,step,marks,consumers,*,budget,
                           seconds=90,stage13_chart=None,pants_chart=None,
                           export_realization=False):
    seeds=tuple(seeds)
    if not seeds or any(type(h) is not bytes for h in seeds):
        raise ValueError('original source byte seeds required')
    charts={
        'original_189_seed_mark_only':marked_source_chart(marks),
        'entire_original_recurrent_bytes':lambda h:h}
    supplied={}
    for name,fn in (('source_owned_stage13',stage13_chart),
                    ('source_owned_five_field_pants',pants_chart)):
        if fn is None:
            supplied[name]=False
        elif callable(fn):
            supplied[name]=True
            charts[name]=fn
        else:
            raise ValueError('source typed chart must be callable or absent')
    evidence=fibre_audit(seeds,controls,step,charts,consumers,
                         budget=budget,seconds=seconds,
                         export_realization=export_realization)
    negative={}
    for name,report in evidence['charts'].items():
        neg=report['witnessed_failures']
        negative[name]={key:replay_witness(w,seeds,step,charts)
                        for key,w in neg.items()}
        if any(not w['replayed'] for w in negative[name].values()):
            raise AssertionError('source-owned negative failed independent replay')
    return dict(
        source_original_seed_count=len(seeds),
        original_marked_fabric_cardinality=len(marks['records']),
        native_typed_optional_source_charts_present=supplied,
        source_graph=evidence,
        independently_replayed_negative_evidence=negative,
        original_neural_recurrence_unchanged=True,
        full_original_history_bytes_retained=True,
        original_source_model_checkpoint_authenticated_by_caller_only=True,
        original_all_float_states_covered=False,
        original_stage13_semantics_invented=False,
        original_pants_five_field_semantics_invented=False)

def native_original_consumer_maxcut(model,controls,*,budget=1000,
                                    seconds=90,export_realization=False):
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.x369_source_future_acceptance import native_future_contract
    from scripts.audit_x369_joint_action import MEANS
    from scripts.audit_x369_ordered_signed_task import TRIADS
    from scripts.x369_original_marked_seed_fabric import marked_seed_records
    seed,_,_=native_source(model)
    marks=marked_seed_records(MEANS,TRIADS,seed)
    if not marks['original_seed_marking_injective'] or len(marks['records'])!=189:
        raise ValueError('original signed initial carrier source ownership mismatch')
    mode='typed' if model.typed_consumer else 'raw'
    seeds,step,_=native_future_contract(model,mode)
    if set(seeds)!=set(marks['records']):
        raise ValueError('original native source seeds differ from marked producer')
    from scripts.x369_indexed_source_fibre_audit import RAW_CONSUMERS,TYPED_CONSUMERS
    consumers=TYPED_CONSUMERS if model.typed_consumer else RAW_CONSUMERS
    return source_consumer_maxcut(
        seeds,controls,step,marks,consumers,budget=budget,seconds=seconds,
        export_realization=export_realization,
        # A source method must actually produce its own meaningful
        # Stage13/pants semantics; no arbitrary neural float coercion.
        stage13_chart=getattr(model,'source_hypervoxel_stage13',None),
        pants_chart=getattr(model,'source_pants_signature',None))

def certificate():
    marks_data={
        'original_seed_marking_injective':True,
        'records':{
            bytes((0,)):dict(mean=(0,0,0),signed_residual=(0,0,0)),
            bytes((1,)):dict(mean=(1,0,0),signed_residual=(0,0,0))}}
    def step(h,u):
        i=h[0]
        return (('obs',i),),bytes(((i+2)%4 if i<2 else i,))
    result=source_consumer_maxcut(
        (bytes((0,)),bytes((1,))),('tick',),step,
        marks_data,('obs',),budget=4,
        export_realization=True)
    reports=result['source_graph']['charts']
    assert reports['original_189_seed_mark_only']['status']=='refuted'
    assert reports['entire_original_recurrent_bytes']['status']=='finite_closed'
    assert result['source_graph']['finite_closed_graph_for_realization']
    assert result['independently_replayed_negative_evidence'][
        'original_189_seed_mark_only']['obs']['replayed']
    return dict(
        finite_original_coarse_mark_refuted=True,
        original_byte_fine_closed=True,
        source_negative_independently_replayed=True,
        original_neural_checkpoint_executed=False,
        agda_kernel_checked=False)

if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True))
