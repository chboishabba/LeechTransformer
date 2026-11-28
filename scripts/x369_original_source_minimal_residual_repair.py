"""Exact MINIMAL consumer residual above a chosen original-source coarse chart.

For a finite, fully successor-closed actual original source graph,
reuse the existing x369_nonlinear_closure.refine partition. Stable
classes split every coarse-chart fibre into precisely as many classes
as are needed for the specified ORIGINAL consumer portfolio and
controls. An additional within-fibre index is then a reversible
consumer-sufficient repair of the coarse observer (not full bytes).

No claim beyond the explicitly closed finite source domain. Do not
confuse observer-minimal with a lossless history codec or extrapolate
from a bounded neural exploration.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from collections import defaultdict
from scripts.check_x369_nonlinear_closure import refine
from scripts.x369_future_realization import (
    compile_realization,verify_realization,pack)

def residual_repair(states,seeds,controls,chart,step):
    states=tuple(states);seeds=tuple(seeds);controls=tuple(controls)
    if not states or not seeds or not controls or (
        not callable(chart) or not callable(step)):
        raise ValueError('closed original source states/seeds/controls/callbacks required')
    if len(set(states))!=len(states) or len(set(seeds))!=len(seeds):
        raise ValueError('original source state identity ambiguous')
    if not set(seeds)<=set(states):
        raise ValueError('seed not in original closed source domain')
    codes={h:chart(h) for h in states}
    for code in codes.values():
        try:hash(code)
        except TypeError as e:raise ValueError('original coarse chart must be hashable') from e
    table={(h,u):step(h,u) for h in states for u in controls}
    if any(value is None or value[1] not in codes for value in table.values()):
        raise ValueError('original source graph not fully closed')
    partitions,_=refine(states,controls,codes.__getitem__,
                        lambda h,u:table[h,u])
    final=partitions[-1]
    classes_by_code=defaultdict(set)
    for h in states:
        classes_by_code[codes[h]].add(final[h])
    # Within each original coarse fibre, use a deterministic ordered
    # section, avoiding dependence on Python hash seed or source bytes.
    ordered_ids={}
    for code in classes_by_code:
        ids=sorted(classes_by_code[code])
        ordered_ids[code]={j:i for i,j in enumerate(ids)}
    repaired={h:(codes[h],ordered_ids[codes[h]][final[h]]) for h in states}
    exact_reopen={}
    for h in states:
        key=repaired[h]
        if key in exact_reopen and final[exact_reopen[key]]!=final[h]:
            raise AssertionError('repaired class encoding mixed original future classes')
        exact_reopen.setdefault(key,h)
    # Every original stable class has a unique repaired code. The code
    # identifies a FUTURE-CONSUMER CLASS, not an original byte history.
    if len(exact_reopen)!=len(set(final.values())):
        raise AssertionError('insufficient within-coarse source residual labels')
    direct=compile_realization(states,seeds,controls,codes,table,final)
    original_replay=verify_realization(
        direct,seeds,controls,codes.__getitem__,step)
    if original_replay['classes']!=len(exact_reopen):
        raise AssertionError('repaired minimal source class mismatch')
    max_fibre=max(len(xs) for xs in classes_by_code.values())
    return dict(
        source_closed_finite_domain=len(states),
        original_coarse_fibres=len(classes_by_code),
        exact_minimal_future_consumer_classes=len(exact_reopen),
        residual_class_counts_by_coarse_code=[
            dict(source_coarse_code=pack(k),required_distinct_classes=len(v))
            for k,v in classes_by_code.items()],
        maximal_required_residual_classes_in_one_fibre=max_fibre,
        minimal_worst_case_extra_binary_bits=(max_fibre-1).bit_length(),
        # A JSON-safe source-owned table, no bytes dictionary keys or
        # repr-only semantic identities. Existing pack is exact.
        exact_source_fibre_rows=[
            dict(original_source=pack(h),
                 original_coarse_code=pack(codes[h]),
                 minimal_residual_index=repaired[h][1],
                 stable_future_class=final[h],
                 exact_repaired_code=pack(repaired[h]))
            for h in states],
        representative_original_sources=[
            dict(repaired_code=pack(k),source_original_bytes=pack(v))
            for k,v in exact_reopen.items()],
        original_source_finite_realization=direct,
        original_source_replay=original_replay,
        original_exact_finite_consumer_descent=True,
        actual_original_history_bytes_losslessly_recovered=False,
        original_unrestricted_float_source_certified=False)

def certificate():
    states=tuple(bytes((i,)) for i in range(4))
    seeds=(states[0],states[1])
    chart=lambda h:h[0]%2
    def step(h,u):
        i=h[0]
        return (('obs',i),),states[(i+2)%4]
    out=residual_repair(states,seeds,('rotate',),chart,step)
    assert out['original_coarse_fibres']==2
    assert out['exact_minimal_future_consumer_classes']==4
    assert out['maximal_required_residual_classes_in_one_fibre']==2
    assert out['minimal_worst_case_extra_binary_bits']==1
    assert out['original_source_replay']['minimal']
    return dict(
        closed_fixture_original_states=4,
        initial_coarse_fibres=2,
        exact_future_stable_classes=4,
        minimum_extra_bits_per_fibre_worst_case=1,
        original_source_replay_exact=True,
        source_neural_checkpoint_executed=False,
        agda_kernel_checked=False)

if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True))
