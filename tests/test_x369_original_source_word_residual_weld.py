"""Regression for arbitrary-word original source byte/observation BIDI."""
import unittest
from models.geometric_reflection_fibre import GROUP
from scripts.x369_original_marked_seed_fabric import signed_original_task_action
from scripts.x369_original_source_word_residual_weld import (
    original_word_transport,reopen_word_transport,
    independently_replay_original_word,certificate)

SEED=lambda m,d:bytes(v+1 for v in m+d)
MARK=dict(mean=(1,0,0),signed_residual=(0,-1,1))
INITIAL=SEED(MARK['mean'],MARK['signed_residual'])
WORD=((GROUP.index((1,0,0)),0),(GROUP.index((0,1,0)),1),
      (GROUP.index((0,0,1)),3))

class OriginalSourceArbitraryWord(unittest.TestCase):
    def test_zero_word_and_all_prefixes_reopen(self):
        def step(h,u):return (('real_history',h),),h
        for n in range(8):
            w=(WORD*3)[:n]
            receipt=original_word_transport(INITIAL,MARK,w,step,SEED)
            self.assertEqual(receipt['word_length'],n)
            self.assertEqual(reopen_word_transport(receipt),INITIAL)
            self.assertTrue(independently_replay_original_word(receipt,step)[
                'original_final_source_bytes_replayed_exact'])
            self.assertFalse(receipt['actual_learned_source_group_equivariance_certified'])

    def test_actual_learned_step_and_canonical_task_differ(self):
        def step(h,u):
            next_h=bytes(((h[0]+1)%3,))+h[1:]
            return (('source_before',h),('actual_control',u)),next_h
        # The original mock and the independent signed-task word
        # BOTH return to their initial endpoint after six controls.
        # Endpoint equality does not mean the intermediate paths agree.
        receipt=original_word_transport(INITIAL,MARK,WORD*2,step,SEED)
        self.assertEqual(receipt['word_length'],6)
        self.assertEqual(receipt['total_defect_nonzero_bytes'],0)
        self.assertGreater(receipt['intermediate_nonzero_defect_steps'],0)
        self.assertGreater(receipt['maximal_intermediate_nonzero_defect_bytes'],0)
        self.assertEqual(reopen_word_transport(receipt),INITIAL)
        self.assertTrue(independently_replay_original_word(receipt,step)[
            'original_observations_replayed_exact'])
        # Test a genuinely different endpoint, rather than asserting
        # divergence at a word where both actions have returned.
        prefix=original_word_transport(INITIAL,MARK,(WORD*2)[:5],step,SEED)
        self.assertEqual(prefix['word_length'],5)
        self.assertGreater(prefix['total_defect_nonzero_bytes'],0)
        self.assertEqual(reopen_word_transport(prefix),
                         bytes(((INITIAL[0]+5)%3,))+INITIAL[1:])
        self.assertTrue(independently_replay_original_word(prefix,step)[
            'original_observations_replayed_exact'])

    def test_sixteen_step_chronological_source_task_matching(self):
        def step(h,u):
            mean=tuple(x-1 for x in h[:3])
            d=tuple(x-1 for x in h[3:6])
            g=GROUP[u[0]]
            m2,d2=signed_original_task_action(g,mean,d)
            return (('signed_task_triad',m2+d2),),SEED(m2,d2)
        word=WORD*5+(WORD[0],)
        receipt=original_word_transport(INITIAL,MARK,word,step,SEED)
        self.assertEqual(receipt['total_defect_nonzero_bytes'],0)
        self.assertEqual(reopen_word_transport(receipt),
                         bytes.fromhex(receipt['independent_expected_final_hex']))
        self.assertTrue(independently_replay_original_word(receipt,step)[
            'original_final_source_bytes_replayed_exact'])

    def test_modified_receipts_and_source_replay_rejected(self):
        def step(h,u):return (('obs',h),),h
        receipt=original_word_transport(INITIAL,MARK,WORD,step,SEED)
        forged=dict(receipt,final_xor_residual_hex='00'*len(INITIAL))
        with self.assertRaises(ValueError):reopen_word_transport(forged)
        forged2=dict(receipt,chronological_word=tuple(reversed(WORD)))
        with self.assertRaises(ValueError):
            independently_replay_original_word(forged2,step)
        def bad(h,u):return (('obs',b'different'),),h
        with self.assertRaises(AssertionError):
            independently_replay_original_word(receipt,bad)

    def test_invalid_seed_and_word_budget_fail_closed(self):
        def step(h,u):return (('obs',h),),h
        with self.assertRaises(ValueError):
            original_word_transport(b'not-source',MARK,WORD,step,SEED)
        with self.assertRaises(ValueError):
            original_word_transport(INITIAL,MARK,WORD,step,SEED,max_depth=2)
        with self.assertRaises(ValueError):
            original_word_transport(INITIAL,MARK,((17,0),),step,SEED)

    def test_static_source_free_certificate(self):
        result=certificate()
        self.assertTrue(result['finite_original_arbitrary_word_bidi'])
        self.assertFalse(result['original_neural_checkpoint_executed'])

if __name__=='__main__':
    unittest.main()
