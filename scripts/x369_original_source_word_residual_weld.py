"""Arbitrary-word SAME ORIGINAL SOURCE residual transport, no neural equivariance.

Build on existing actual native source step, original signed-task action,
and byte-XOR inverse. Neither replace the trained MoE nor assume the
original task action preserves its learned recurrent history.

For every finite chronological word u1...uk:
  actual_k = canonical_task_k XOR residual_k.
Each residual is the *total* difference at that endpoint. Intermediate
source histories/observations are independently replayable; do not claim
a linear cocycle for the nonlinear learned step.

Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from hashlib import sha256
from scripts.x369_original_marked_seed_fabric import signed_original_task_action
from scripts.x369_original_source_signed_defect_transport import (
    source_step_exact,signed_action_of_control,original_source_defect_packet,
    reopen_source_defect,_xor)

def _sha(b):
    if type(b) is not bytes:raise ValueError('source history must be bytes')
    return sha256(b).hexdigest()

def _encode_obs_val(v):
    if isinstance(v, bytes):return ['bytes', v.hex()]
    if isinstance(v, tuple):return ['tuple', [_encode_obs_val(x) for x in v]]
    if isinstance(v, list):return [_encode_obs_val(x) for x in v]
    if isinstance(v, dict):return {k: _encode_obs_val(val) for k, val in v.items()}
    return v

def _decode_obs_val(v):
    if isinstance(v, list) and len(v) == 2 and v[0] == 'bytes':return bytes.fromhex(v[1])
    if isinstance(v, list) and len(v) == 2 and v[0] == 'tuple':return tuple(_decode_obs_val(x) for x in v[1])
    if isinstance(v, list):return [_decode_obs_val(x) for x in v]
    if isinstance(v, dict):return {k: _decode_obs_val(val) for k, val in v.items()}
    return v

def _encode_obs(obs):
    return {k: _encode_obs_val(v) for k, v in obs.items()}

def _decode_obs(obs):
    return {k: _decode_obs_val(v) for k, v in obs.items()}

def original_word_transport(initial,source_mark,word,source_step,original_seed,
                            *,max_depth=1000):
    """Full source-first word residual with source-owned observation trace.

    A receipt is a proof-by-replay about the exact ORIGINAL callback on
    given inputs; it is not a theorem on unobserved floating histories.
    """
    if type(initial) is not bytes or not isinstance(source_mark,dict) or (
        'mean' not in source_mark or 'signed_residual' not in source_mark):
        raise ValueError('original initial bytes and actual signed source mark required')
    if not callable(source_step) or not callable(original_seed):
        raise ValueError('original callbacks required')
    word=tuple(word)
    if len(word)>max_depth or type(max_depth) is not int or max_depth<0:
        raise ValueError('bounded original word length required')
    mean=tuple(source_mark['mean'])
    residual=tuple(source_mark['signed_residual'])
    if original_seed(mean,residual)!=initial:
        raise ValueError('initial byte history not the source-owned marked seed')
    history=initial
    rows=[]
    for index,u in enumerate(word):
        g=signed_action_of_control(u)
        mean,residual=signed_original_task_action(g,mean,residual)
        canonical=original_seed(mean,residual)
        obs,actual=source_step_exact(source_step,history,u)
        packet=original_source_defect_packet(actual,canonical,obs,u)
        if reopen_source_defect(packet)!=actual:
            raise AssertionError('exact source history not reopened')
        rows.append(dict(step=index,control=u,
            input_source_sha256=_sha(history),
            independent_expected_mark=(mean,residual),
            original_source_consumer_fields=tuple(obs),
            original_source_observations=_encode_obs(obs),
            original_step_defect=packet,
            output_source_sha256=_sha(actual)))
        history=actual
    final_canonical=original_seed(mean,residual)
    final_defect=_xor(history,final_canonical)
    intermediate_nonzero=sum(
        any(v!=0 for v in bytes.fromhex(row['original_step_defect']['exact_xor_residual_hex']))
        for row in rows)
    max_intermediate_nonzero=max((
        row['original_step_defect']['signed_nonzero_residual_bytes']
        for row in rows),default=0)
    if intermediate_nonzero>len(rows):
        raise AssertionError('source intermediate defect audit inconsistency')
    if _xor(final_defect,final_canonical)!=history:
        raise AssertionError('original arbitrary-word BIDI failed')
    return dict(
        chronological_word=word,word_length=len(word),
        original_initial_source_sha256=_sha(initial),
        original_initial_source_hex=initial.hex(),
        original_final_source_sha256=_sha(history),
        independent_expected_final_sha256=_sha(final_canonical),
        independent_expected_final_hex=final_canonical.hex(),
        final_xor_residual_hex=final_defect.hex(),
        total_defect_nonzero_bytes=sum(x!=0 for x in final_defect),
        intermediate_nonzero_defect_steps=intermediate_nonzero,
        maximal_intermediate_nonzero_defect_bytes=max_intermediate_nonzero,
        endpoint_defect_can_cancel_intermediate_divergence=True,
        original_step_receipts=rows,
        original_full_history_reopens_exact=True,
        original_observations_retained=True,
        actual_learned_source_group_equivariance_certified=False,
        arbitrary_word_identity_on_this_replayed_source=True,
        universal_source_float_space_theorem=False)

def reopen_word_transport(receipt):
    """Check every stored chained byte history and canonical XOR receipt."""
    original=bytes.fromhex(receipt['original_initial_source_hex'])
    if _sha(original)!=receipt['original_initial_source_sha256']:
        raise ValueError('altered original initial source history')
    current=original
    for k,row in enumerate(receipt['original_step_receipts']):
        if row['step']!=k or row['input_source_sha256']!=_sha(current):
            raise ValueError('broken original source history chronology')
        current=reopen_source_defect(row['original_step_defect'])
        if _sha(current)!=row['output_source_sha256']:
            raise ValueError('original source step receipt digest altered')
        if tuple(row['original_source_consumer_fields'])!=tuple(
            row['original_source_observations']):
            raise ValueError('original consumer fields omitted in replay')
    if len(receipt['original_step_receipts'])!=receipt['word_length']:
        raise ValueError('invalid original word receipt length')
    if _sha(current)!=receipt['original_final_source_sha256']:
        raise ValueError('original final history digest mismatch')
    canonical=bytes.fromhex(receipt['independent_expected_final_hex'])
    if _sha(canonical)!=receipt['independent_expected_final_sha256']:
        raise ValueError('canonical expected endpoint changed')
    result=_xor(canonical,bytes.fromhex(receipt['final_xor_residual_hex']))
    if result!=current:
        raise ValueError('arbitrary-word source residual does not reconstruct')
    return current

def independently_replay_original_word(receipt,source_step):
    """Re-execute the actual source and every original readout, not labels."""
    if not callable(source_step):raise ValueError('original source required')
    h=bytes.fromhex(receipt['original_initial_source_hex'])
    if tuple(receipt['chronological_word'])!=tuple(
        row['control'] for row in receipt['original_step_receipts']):
        raise ValueError('original chronological controls changed')
    for row in receipt['original_step_receipts']:
        obs,h=source_step_exact(source_step,h,row['control'])
        if obs!=_decode_obs(row['original_source_observations']) or (
            _sha(h)!=row['output_source_sha256']):
            raise AssertionError('actual original learned source replay divergence')
    if h!=reopen_word_transport(receipt):
        raise AssertionError('source history reconstructed with wrong replay')
    return dict(
        source_steps_replayed=receipt['word_length'],
        original_observations_replayed_exact=True,
        original_final_source_bytes_replayed_exact=True,
        source_checkpoint_authenticated_by_caller_only=True)

def native_original_word_residual(model,controls,*,seed_budget=4,
                                  word=None,max_depth=64):
    """One immutable-source history per genuine original marked seed.

    The authenticated runner owns source/checkpoint identity. This adapter
    never substitutes a signed task label for original learned outcomes.
    """
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.x369_source_future_acceptance import native_future_contract
    from scripts.audit_x369_joint_action import MEANS
    from scripts.audit_x369_ordered_signed_task import TRIADS
    from scripts.x369_original_marked_seed_fabric import marked_seed_records
    seed,_,_=native_source(model)
    marks=marked_seed_records(MEANS,TRIADS,seed)
    if not marks['original_seed_marking_injective'] or len(marks['records'])!=189:
        raise ValueError('source initial marked 189-state carrier not exact')
    seeds,step,_=native_future_contract(model,
        'typed' if model.typed_consumer else 'raw')
    if set(seeds)!=set(marks['records']):
        raise ValueError('original source and marked seeds disagree')
    word=tuple(controls if word is None else word)
    if not word or len(word)>max_depth or not 1<=seed_budget<=189:
        raise ValueError('bounded original source input word and seeds required')
    rows=[]
    for index,h in enumerate(seeds[:seed_budget]):
        receipt=original_word_transport(h,marks['records'][h],
                                         word,step,seed,max_depth=max_depth)
        recheck=independently_replay_original_word(receipt,step)
        if not recheck['original_final_source_bytes_replayed_exact']:
            raise AssertionError('actual source word transport failed independent replay')
        rows.append(dict(original_seed_index=index,receipt=receipt,
                         original_source_replay=recheck))
    return dict(
        original_seed_count=len(seeds),audited_seed_count=len(rows),
        original_control_word_length=len(word),
        exact_source_word_transport_receipts=rows,
        all_original_bytes_and_consumers_reopened=True,
        original_source_float_universe_not_certified=True,
        original_checkpoint_authenticated_by_caller_only=True,
        agda_kernel_checked=False)

def certificate():
    from models.geometric_reflection_fibre import GROUP
    seed=lambda m,d:bytes(v+1 for v in m+d)
    mean=(1,0,0);signed=(0,-1,1)
    original=seed(mean,signed)
    mark=dict(mean=mean,signed_residual=signed)
    controls=((GROUP.index((1,0,0)),0),
              (GROUP.index((0,1,0)),1),
              (GROUP.index((0,0,1)),2))
    def unchanged(h,u):return (('raw_source_bytes',h),),h
    for word in ((),controls,controls*2):
        receipt=original_word_transport(original,mark,word,unchanged,seed)
        assert reopen_word_transport(receipt)==original
        assert independently_replay_original_word(receipt,unchanged)[
            'original_observations_replayed_exact']
    return dict(fixture_words=3,
        finite_original_arbitrary_word_bidi=True,
        independent_actual_source_replay=True,
        original_neural_checkpoint_executed=False,
        agda_kernel_checked=False)

if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True))
