"""Four-way original source consumer residual lower bound from REAL GPU receipt.

Source-free, receipt-based (not a new native replay). Use the already
validated original mark-only collision checker as provenance gate.
Check all four actual unmarked 384-byte source states reached under
the SAME first control and every original named consumer under the SAME
second control. Exact distinct observations give a fixed-chart residual
alphabet lower bound, not a globally minimal neural compression.

Original Leech-LILA: Anatolii Kornienko; DASHI source proof extension.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from scripts.x369_original_marked_seed_fabric import (
    decode_canonical_original_seed_bytes)
from scripts.x369_original_receipt_mark_collision_weld import (
    extract_original_mark_collision,FIELDS,_hash_bytes,_hash_observation)

def certify_four_way_original_source_residual(receipt):
    gate=extract_original_mark_collision(receipt)
    steps=receipt['stages']['indexed_fibre'][
        'original_arbitrary_word_source_residual'][
        'exact_source_word_transport_receipts']
    config=receipt['config']
    samples=[]
    for row in steps:
        word=row['receipt']
        if word['word_length']<2:raise ValueError('original 2-step word missing')
        first,second=word['original_step_receipts'][:2]
        source=bytes.fromhex(first['original_step_defect']['actual_source_hex'])
        if not (_hash_bytes(source)==first['output_source_sha256']==
                second['input_source_sha256']==
                first['original_step_defect']['original_source_endpoint_sha256']):
            raise ValueError('original native source first-step provenance altered')
        if decode_canonical_original_seed_bytes(
            source,experts=config['experts'],width=config['width'],
            dtype=config['dtype']) is not None:
            raise ValueError('source successor does not share unmarked coarse chart')
        obs=second['original_source_observations']
        if set(obs)!=set(FIELDS):
            raise ValueError('original named consumer portfolio is incomplete')
        samples.append(dict(
            seed=row['original_seed_index'],
            first=tuple(first['control']),second=tuple(second['control']),
            history_sha256=_hash_bytes(source),
            bytes=len(source),
            consumer_sha256={name:_hash_observation(obs[name]) for name in FIELDS},
            observations=obs))
    if len(samples)<4:
        raise ValueError('four actual independently marked original source histories required')
    selected=samples[:4]
    first=selected[0]['first'];second=selected[0]['second']
    if any(s['first']!=first or s['second']!=second for s in selected):
        raise ValueError('original sampled source control histories differ')
    if len({s['history_sha256'] for s in selected})!=4:
        raise ValueError('original histories not bytewise distinct')
    pair_rows=[]
    for a,b in combinations(selected,2):
        differing=[f for f in FIELDS if a['observations'][f]!=b['observations'][f]]
        if not differing:
            raise ValueError('two actual source states have same consumer portfolio')
        pair_rows.append(dict(
            left_seed=a['seed'],right_seed=b['seed'],
            differing_original_consumers=differing,
            left_observation_sha256=a['consumer_sha256'],
            right_observation_sha256=b['consumer_sha256']))
    per_field={f:len({s['consumer_sha256'][f] for s in selected}) for f in FIELDS}
    if max(per_field.values())!=4:
        raise ValueError('expected four distinct original outputs for some named field')
    return dict(
        status='receipt_verified_four_way_mark_only_residual_lower_bound',
        source_head=receipt['source']['head'],
        original_checkpoint_sha256=gate['original_checkpoint_sha256'],
        original_model_sha256=gate['original_model_sha256'],
        original_receipt_sha256=hashlib.sha256(
            json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        original_coarse_chart='189 signed initial marks else one unmarked value',
        original_coarse_fibre='unmarked_original_recurrent_history',
        chronological_controls=[list(first),list(second)],
        original_fine_histories=[
            dict(original_seed=s['seed'],state_sha256=s['history_sha256'],
                 state_bytes=s['bytes']) for s in selected],
        original_source_history_count=4,
        all_pairs_checked=len(pair_rows),
        actual_source_pairwise_readout_separators=pair_rows,
        distinct_original_readouts_by_consumer=per_field,
        at_least_residual_labels_in_unmarked_fibre=4,
        at_least_fixed_binary_residual_bits_in_unmarked_fibre=2,
        lower_bound_relative_to_this_coarse_chart=True,
        full_source_global_residual_minimum_claim=False,
        source_graph_finite_closure_claim=False,
        original_gpu_model_newly_replayed=False,
        parent_original_gpu_receipt_verified=True,
        original_training_provenance_authenticated=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    raw=a.receipt.read_bytes()
    cert=certify_four_way_original_source_residual(json.loads(raw))
    cert['original_receipt_file_sha256']=hashlib.sha256(raw).hexdigest()
    result=json.dumps(cert,sort_keys=True,indent=2)+'\n'
    if a.output:a.output.write_text(result)
    print(result,end='')

if __name__=='__main__':main()
