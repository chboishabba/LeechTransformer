"""Whole-source retained execution and independently indexed SLR context tests."""
import json
import struct
import unittest
from pathlib import Path
from scripts.x369_whole_source_indexed_transport import (
    offline, read_context, pack_joint, reopen_joint, joint_step, stage13_view)
from scripts.x369_original_receipt_multichart_weld import digest
from scripts.x369_source_acceptance import parse_args, stage_plan

ROOT=Path(__file__).resolve().parents[1]/'docs/receipts'
FILES=('x369_original_native_minimal_consumer_20261128.json',
       'x369_original_targeted_semantic_20261128.json','x369_reachable_capacity_orbit_native.json')
CFG=dict(experts=4,width=12,dtype='float32')


def context_fixture():
    body=dict(schema='slr.query-context-capsule.v1',family_ref='family:test',
              query_ref='query:test',consumer_ref='consumer:test',creates_semantic_authority=False,
              writes_database=False,world_truth_paid=False)
    cap='sha256:'+'0'*64
    env=dict(schema='slr.query-context-capsule-envelope.v1',capsule=body,capsule_digest=cap,
             creates_semantic_authority=False,writes_database=False)
    j=dict(schema='slr.query-context-expanded.v1',capsule_envelope=env,capsule_digest=cap,
           exact_current_replay=True,family=dict(family_ref='family:test'),
           cohort=dict(family_ref='family:test',query_ref='query:test',consumer_ref='consumer:test'),
           creates_semantic_authority=False,writes_database=False)
    raw=json.dumps(j).encode()
    return read_context(raw,expected_sha=digest(raw),expected_capsule=cap,
                        family='family:test',query='query:test')


class WholeSourceTransport(unittest.TestCase):
    def test_actual_three_receipt_recorded_transport(self):
        r=offline(*((ROOT/p).read_bytes() for p in FILES))
        self.assertEqual((r['histories'],r['edges']),(171,159))
        self.assertEqual((r['exact_recorded_edge_squares'],r['digest_only_edges_without_payload']),(137,22))
        self.assertTrue(r['all_histories_reopened'])
        self.assertFalse(r['original_gpu_newly_replayed'])
        self.assertFalse(r['closed_source_graph_certified'])

    def test_independent_indices_survive_identical_neural_state(self):
        h=struct.pack('<96f',*([0.,-0.,1.,2.,-3.,4.]*16))
        c=context_fixture()
        i=dict(history=0,physical_time='physical-t',refinement=4,frobenius='F',version='v')
        p=pack_joint(h,CFG,c,i)
        def step(b,u):
            self.assertEqual((b,u),(h,(8,0)))
            return ('original raw consumer',),b
        def index_transport(old,prev,new,u):
            self.assertEqual(prev,new)
            return dict(old,history=1)
        obs,q=joint_step(p,(8,0),step,lambda old,u:old,index_transport)
        self.assertEqual(obs,('original raw consumer',))
        self.assertEqual(reopen_joint(q),(h,c,dict(i,history=1)))
        self.assertNotEqual(p,q)
        with self.assertRaises(ValueError):stage13_view(q,None,None)
        with self.assertRaises(ValueError):pack_joint(h,CFG,c,dict(history=1))
        with self.assertRaises(ValueError):
            read_context(c.raw,expected_sha=c.raw_sha256,expected_capsule='wrong',
                         family=c.family_ref,query=c.query_ref)

    def test_exact_mode_requires_all_three_and_excludes_other_modes(self):
        args=['--checkpoint','not-read.pt','--sha256','0'*64,'--source-head','0'*40,
              '--output','not-written.json','--fibre-budget','189',
              '--only-stages','admission','indexed_fibre',
              '--receipt-exact-retained-orbit',str(ROOT/FILES[2])]
        with self.assertRaises(SystemExit):parse_args(args)
        args+=['--receipt-exact-retained-targeted',str(ROOT/FILES[1]),
               '--receipt-targeted-multichart',str(ROOT/FILES[0])]
        self.assertEqual(stage_plan(parse_args(args)),('admission','indexed_fibre'))
        with self.assertRaises(SystemExit):parse_args(args+['--receipt-capacity-fibre',str(ROOT/FILES[1])])


if __name__=='__main__':unittest.main()
