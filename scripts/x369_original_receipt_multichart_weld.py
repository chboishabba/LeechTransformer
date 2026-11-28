"""Offline whole-source multi-chart audit of the committed native gfx803 receipt.

Original Leech-LILA: Anatolii Kornienko. DASHI proof/geometry extension.
Exact rational chart owners are reused unchanged. Negative witnesses and
lower bounds on observed fibres only; no invented edges or native replay.
"""
import argparse
import hashlib
import json
import math
import struct
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path

from scripts.x369_capped_semantic_geometry import exact_repair, cap_profile
from scripts.x369_original_marked_seed_fabric import decode_canonical_original_seed_bytes
from scripts.x369_original_receipt_mark_collision_weld import extract_original_mark_collision, FIELDS
from scripts.x369_original_source_signed_defect_transport import reopen_source_defect
from scripts.x369_original_source_word_residual_weld import _decode_obs

RECEIPT_SHA256 = 'e7c835a351e52afa88548498305d9477f00f5e3541ee593392f61dfb63f22789'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def packet_bytes(packet, size):
    actual = bytes.fromhex(packet['actual_source_hex'])
    canonical = bytes.fromhex(packet['canonical_source_hex'])
    residual = bytes.fromhex(packet['exact_xor_residual_hex'])
    require(len(actual) == len(canonical) == len(residual) == size,
            'invalid original source packet size')
    require(reopen_source_defect(packet) == actual and
            digest(canonical) == packet['predicted_canonical_endpoint_sha256'] and
            sum(x != 0 for x in residual) == packet['signed_nonzero_residual_bytes'],
            'original source residual packet changed')
    return actual


def chart_values(h, config):
    """Same rational values as native_future_contract, not a new decoder."""
    e, w, dt = config['experts'], config['width'], config['dtype']
    require(dt in ('float32', 'float64'), 'unsupported source float storage')
    scalar = 'f' if dt == 'float32' else 'd'
    count = e * 2 * w
    require(len(h) == struct.calcsize('<' + str(count) + scalar),
            'invalid native byte shape')
    values = struct.unpack('<' + str(count) + scalar, h)
    require(all(math.isfinite(x) for x in values), 'nonfinite history')
    v = tuple(tuple(tuple(F(x) for x in values[(2*i+j)*w:(2*i+j+1)*w])
                    for j in range(2)) for i in range(e))
    six = exact_repair(v, (0,)*6, range(e), 1)['defect']
    mark = decode_canonical_original_seed_bytes(h, experts=e, width=w, dtype=dt)
    return {
        'source_owned_original_189_initial_mark_only':
            ('unmarked_original_recurrent_history',) if mark is None else
            ('marked', tuple(mark['native_input9'])),
        'constructed_semantic_six': six,
        'semantic_plus_cap_profile': (six, cap_profile(v)),
        'full_original_history_bytes': h,
    }


def extract_edges(receipt):
    gate = extract_original_mark_collision(receipt)
    cfg = receipt['config']
    size = cfg['experts'] * 2 * cfg['width'] * (4 if cfg['dtype'] == 'float32' else 8)
    indexed = receipt['stages']['indexed_fibre']
    require(indexed['source_head'] == receipt['source']['head'],
            'source head mismatch')
    words = indexed['original_arbitrary_word_source_residual']['exact_source_word_transport_receipts']
    states, edges, initial = {}, {}, {}

    def add(h, u, packet, origin, payload=None):
        n = packet_bytes(packet, size)
        u = tuple(u)
        require(tuple(packet['control']) == u and len(u) == 2 and
                all(type(x) is int for x in u) and 0 <= u[0] < 12 and 0 <= u[1] < 4,
                'invalid source control')
        hashes = packet['original_observation_sha256']
        require(set(hashes) == set(FIELDS) and all(
            isinstance(v, str) and len(v) == 64 and
            all(c in '0123456789abcdef' for c in v) for v in hashes.values()),
            'invalid observation digest portfolio')
        if payload is not None:
            require(set(payload) == set(FIELDS) and
                    {k: digest(repr(v).encode()) for k, v in _decode_obs(payload).items()}
                    == hashes, 'typed original observation payload changed')
        for b in (h,n):
            require(len(b) == size, 'invalid source size')
            key = digest(b)
            require(key not in states or states[key] == b, 'state hash collision')
            states[key] = b
        key = (digest(h), u)
        row = edges.get(key)
        if row is None:
            edges[key] = dict(successor=digest(n), hashes=hashes,
                              payload=payload, provenance=[origin])
        else:
            require(row['successor'] == digest(n) and row['hashes'] == hashes,
                    'inconsistent repeated source edge')
            if payload is not None:
                require(row['payload'] is None or row['payload'] == payload,
                        'inconsistent repeated exact readout')
                row['payload'] = payload
            row['provenance'].append(origin)
        return n

    for r in words:
        seed = r['original_seed_index']
        word = r['receipt']
        h = bytes.fromhex(word['original_initial_source_hex'])
        require(digest(h) == word['original_initial_source_sha256'] and
                seed not in initial, 'invalid or repeated seed')
        initial[seed] = h
        steps = word['original_step_receipts']
        require(len(steps) == word['word_length'] and
                [s['control'] for s in steps] == word['chronological_word'],
                'word chronology changed')
        for k, s in enumerate(steps):
            require(s['step'] == k and digest(h) == s['input_source_sha256'] and
                    set(s['original_source_consumer_fields']) == set(FIELDS),
                    'word edge chronology or portfolio changed')
            h = add(h, s['control'], s['original_step_defect'],
                    ['word', seed, k], s['original_source_observations'])
            require(digest(h) == s['output_source_sha256'], 'word successor changed')
        c = bytes.fromhex(word['independent_expected_final_hex'])
        d = bytes.fromhex(word['final_xor_residual_hex'])
        require(digest(h) == word['original_final_source_sha256'] and
                digest(c) == word['independent_expected_final_sha256'] and
                len(c) == len(d) == size and bytes(a^b for a,b in zip(c,d)) == h,
                'word endpoint BIDI changed')

    defects = indexed['original_signed_source_defect']['full_original_byte_defect_receipts']
    for i, r in enumerate(defects):
        h = initial.get(r['seed'])
        require(h is not None and digest(h) == r['original_initial_sha256'],
                'signed-defect initial source unavailable')
        h = add(h, r['first_control'], r['first_defect'], ['signed_defect', i, 0])
        require(digest(h) == r['actual_intermediate_sha256'],
                'signed-defect intermediate changed')
        add(h, r['second_defect']['control'], r['second_defect'],
            ['signed_defect', i, 1])
    return gate, states, edges


def analyse_chart(codes, edges):
    """Finite witnessed pair relation, NOT a completion of missing edges.

    A retained-chart repair must distinguish any equal-chart pair with a
    common observed future ending in unequal output or unequal coarse chart.
    Repeated propagation prepends only actually recorded same-control edges.
    """
    groups, available = defaultdict(list), defaultdict(set)
    for h, q in codes.items():
        groups[q].append(h)
    for h, u in edges:
        available[h].add(u)
    pairs = [tuple(sorted((a,b))) for group in groups.values()
             for a,b in combinations(group,2)]
    shared = {pair: sorted(available[pair[0]] & available[pair[1]]) for pair in pairs}
    witnesses = {}
    for a,b in pairs:
        for u in shared[a,b]:
            l,r = edges[a,u], edges[b,u]
            differing = [f for f in FIELDS if
                (l['payload'][f] != r['payload'][f] if
                 l['payload'] is not None and r['payload'] is not None
                 else l['hashes'][f] != r['hashes'][f])]
            if differing:
                witnesses[a,b] = dict(word=[u], kind='consumer',
                    differing_fields=differing,
                    evidence='exact_payload' if l['payload'] is not None and
                    r['payload'] is not None else 'distinct_stored_digest')
                break
            if codes[l['successor']] != codes[r['successor']]:
                witnesses[a,b] = dict(word=[u], kind='retained_chart',
                                      differing_fields=[], evidence='exact_chart')
                break
    while True:
        additions = {}
        for a,b in pairs:
            if (a,b) in witnesses:
                continue
            for u in shared[a,b]:
                l,r = edges[a,u]['successor'], edges[b,u]['successor']
                end = tuple(sorted((l,r)))
                if end in witnesses:
                    w = witnesses[end]
                    additions[a,b] = dict(w, word=[u]+w['word'])
                    break
        if not additions:
            break
        witnesses.update(additions)

    # Deterministic witnessed clique: valid lower bound even if partial
    # evidence is not a complete equivalence relation. No optimality claim.
    best = []
    for group in groups.values():
        adjacent = {a: {b for b in group if tuple(sorted((a,b))) in witnesses}
                    for a in group}
        order = sorted(group, key=lambda a: (-len(adjacent[a]),a))
        for start in order:
            clique = [start]
            for b in order:
                if b != start and all(b in adjacent[a] for a in clique):
                    clique.append(b)
            if len(clique) > len(best):
                best = clique
    return dict(
        status='refuted_on_observed_source' if witnesses else 'sample_inconclusive',
        states=len(codes), coarse_fibres=len(groups),
        same_chart_pairs=len(pairs),
        same_chart_common_control_comparisons=sum(map(len,shared.values())),
        pairs_without_common_recorded_control=sum(not us for us in shared.values()),
        witnessed_incompatible_pairs=len(witnesses),
        residual_clique=best,
        at_least_residual_labels_in_one_fibre=len(best),
        at_least_fixed_width_residual_bits=(max(1,len(best))-1).bit_length(),
        clique_witnesses=[dict(left=a,right=b,**witnesses[tuple(sorted((a,b)))])
                          for a,b in combinations(best,2)],
        all_pair_witnesses=[dict(left=a,right=b,**w) for (a,b),w in sorted(witnesses.items())],
        maximum_clique_certified=False, full_source_sufficiency_certified=False)


def certify(raw, expected_sha=RECEIPT_SHA256):
    require(digest(raw) == expected_sha, 'receipt file SHA mismatch')
    receipt = json.loads(raw)
    gate, states, edges = extract_edges(receipt)
    values = {h: chart_values(b,receipt['config']) for h,b in states.items()}
    charts = {name: analyse_chart({h:q[name] for h,q in values.items()}, edges)
              for name in next(iter(values.values()))}
    # The full-byte lift is a lossless section, not a compression. Do not
    # compile a finite quotient from the observed partial graph.
    return dict(status='offline_original_multichart_audit',
        receipt_file_sha256=digest(raw),
        source_head=receipt['source']['head'],
        checkpoint_sha256=gate['original_checkpoint_sha256'],
        model_sha256=gate['original_model_sha256'],
        observed_states=len(states), observed_edges=len(edges),
        exact_payload_edges=sum(r['payload'] is not None for r in edges.values()),
        digest_only_edges=sum(r['payload'] is None for r in edges.values()),
        charts=charts,
        state_chart_rows=[dict(history_sha256=h,
                               charts={name:repr(q) for name,q in v.items()
                                       if name!='full_original_history_bytes'})
                          for h,v in sorted(values.items())],
        edge_provenance=[dict(state=h,control=u,successor=r['successor'],
                            observation_sha256=r['hashes'],
                            exact_payload_retained=r['payload'] is not None,
                            provenance=r['provenance'])
                         for (h,u),r in sorted(edges.items())],
        source_graph_finite_closure_certified=False,
        original_gpu_model_newly_replayed=False,
        training_provenance_authenticated=False,
        source_owned_stage13_and_pants_semantics_certified=False,
        agda_kernel_checked=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--sha256',default=RECEIPT_SHA256)
    p.add_argument('--output',type=Path)
    a = p.parse_args()
    result = json.dumps(certify(a.receipt.read_bytes(),a.sha256),sort_keys=True,indent=2)+'\n'
    if a.output:
        a.output.write_text(result)
    print(result,end='')


if __name__ == '__main__':
    main()
