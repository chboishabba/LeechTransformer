"""Offline exact source-backed coarse-mark non-descent from GPU receipt.

No new GPU run. Read the REAL committed authenticated native source
receipt; extract two original histories reached by the SAME first
control, verify both are OUTSIDE the canonical 189-seed byte image,
and compare source readout under the SAME second original control.

This directly refutes the '189 seed mark / one unmarked bucket'
consumer quotient on the four sampled reachable source histories.
It does not prove that a finer hyperfabric, Stage13 or full-byte chart
fails, nor does it prove global finite graph closure.

Original Leech-LILA: Anatolii Kornienko. DASHI extension.
"""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.x369_original_marked_seed_fabric import (
    decode_canonical_original_seed_bytes)

FIELDS=('logits_0','logits_1','logits_2','output',
        'ids','weights','local_signs')

def _hash_bytes(h):
    return hashlib.sha256(h).hexdigest()

def _hash_observation(value):
    # Preserve exact dtype/shape/payload tuple as serialized by the
    # already-accepted lossless _encode_obs owner, not decimal summaries.
    return hashlib.sha256(json.dumps(value,sort_keys=True,
                                     separators=(',',':')).encode()).hexdigest()

def extract_original_mark_collision(receipt):
    if (receipt.get('kind')!='new_source_acceptance' or
        not receipt.get('execution_complete') or
        not receipt.get('complete')):
        raise ValueError('incomplete original native source acceptance receipt')
    admission=receipt['stages']['admission']
    if (admission['status']!='admitted' or
        not admission['checkpoint_digest_verified'] or
        not admission['strict_state_dict_loaded'] or
        not admission['native_gfx803_verified']):
        raise ValueError('missing original source/checkpoint GPU admission')
    indexed=receipt['stages']['indexed_fibre']
    if (not indexed['source_and_checkpoint_identity_verified_by_parent'] or
        indexed['checkpoint_sha256']!=receipt['checkpoint_sha256'] or
        indexed['frozen_model_sha256']!=receipt['model_sha256']):
        raise ValueError('source/model checkpoint provenance mismatch')
    if indexed['model_consumer']!='raw':
        raise ValueError('this source witness requires original raw consumer portfolio')
    if indexed['graph_status']=='closed':
        raise ValueError('this witness is intentionally scoped to open source graph')
    marked=indexed['original_marked_seed_fabric']
    if (marked['source_steps']!=756 or marked['successors_also_marked']!=0 or
        marked['successors_unmarked']!=756 or
        marked['original_seed_cardinality']!=189):
        raise ValueError('original marked source departure evidence absent')
    audit=indexed['original_arbitrary_word_source_residual']
    rows=audit['exact_source_word_transport_receipts']
    if len(rows)<2:
        raise ValueError('two independently sourced histories required')
    width=receipt['config']['width'];experts=receipt['config']['experts']
    dtype=receipt['config']['dtype']
    candidates=[]
    for row in rows:
        word=row['receipt'];steps=word['original_step_receipts']
        if len(steps)<2 or word['word_length']<2:
            raise ValueError('original actual source two-step word missing')
        initial=bytes.fromhex(word['original_initial_source_hex'])
        if _hash_bytes(initial)!=word['original_initial_source_sha256']:
            raise ValueError('original marked seed hash mismatch')
        if decode_canonical_original_seed_bytes(
                initial,experts=experts,width=width,dtype=dtype) is None:
            raise ValueError('original initial state not a canonical marked seed')
        postfirst=bytes.fromhex(steps[0]['original_step_defect']['actual_source_hex'])
        if (_hash_bytes(postfirst)!=steps[0]['output_source_sha256'] or
            _hash_bytes(postfirst)!=steps[1]['input_source_sha256'] or
            _hash_bytes(postfirst)!=
                  steps[0]['original_step_defect']['original_source_endpoint_sha256']):
            raise ValueError('original first-step source/provenance mismatch')
        if decode_canonical_original_seed_bytes(
                postfirst,experts=experts,width=width,dtype=dtype) is not None:
            raise ValueError('actual first successor still in canonical seed chart')
        observations=steps[1]['original_source_observations']
        if set(observations)!=set(FIELDS):
            raise ValueError('original seven consumer fields absent')
        candidates.append(dict(seed=row['original_seed_index'],
            initial_sha256=_hash_bytes(initial),
            first_control=steps[0]['control'],
            second_control=steps[1]['control'],
            original_source_after_first_sha256=_hash_bytes(postfirst),
            original_source_after_first_bytes=len(postfirst),
            observation=observations))
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            if (a['first_control']!=b['first_control'] or
                a['second_control']!=b['second_control'] or
                a['original_source_after_first_sha256']==
                    b['original_source_after_first_sha256']):
                continue
            differing={f:(_hash_observation(a['observation'][f]),
                          _hash_observation(b['observation'][f]))
                       for f in FIELDS
                       if a['observation'][f]!=b['observation'][f]}
            if differing:
                return dict(
                    status='receipt_verified_original_coarse_non_descent',
                    original_source_heads=(receipt['source']['head'],),
                    original_checkpoint_sha256=receipt['checkpoint_sha256'],
                    original_model_sha256=receipt['model_sha256'],
                    original_marked_source_chart='189_initial_marks_else_unmarked',
                    same_coarse_fibre='unmarked_original_recurrent_history',
                    first_original_control=a['first_control'],
                    second_original_control=a['second_control'],
                    original_first_source={'seed':a['seed'],
                         'initial_sha256':a['initial_sha256'],
                         'post_first_sha256':a['original_source_after_first_sha256']},
                    original_second_source={'seed':b['seed'],
                         'initial_sha256':b['initial_sha256'],
                         'post_first_sha256':b['original_source_after_first_sha256']},
                    differing_original_consumer_sha256={
                        k:{'left':v[0],'right':v[1]} for k,v in differing.items()},
                    original_consumers_differ=tuple(differing),
                    original_source_bytes_each=a['original_source_after_first_bytes'],
                    original_seed_image_membership_refuted_by_exact_byte_decoder=True,
                    original_consumer_factorization_through_mark_only_refuted=True,
                    receipt_source_sha_and_observation_payload_verified=True,
                    newly_replayed_original_native_gpu=False,
                    unrestricted_float_neural_source_not_certified=True,
                    source_training_provenance_authenticated=False)
    raise ValueError('no original same-control coarse consumer separator in receipt')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    raw=a.receipt.read_bytes()
    receipt=json.loads(raw)
    outcome=extract_original_mark_collision(receipt)
    outcome['original_receipt_file_sha256']=hashlib.sha256(raw).hexdigest()
    result=json.dumps(outcome,indent=2,sort_keys=True)+chr(10)
    if a.output is not None:
        a.output.write_text(result)
    print(result,end='')

if __name__=='__main__':
    main()
