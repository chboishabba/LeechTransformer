"""Source-authenticated indexed-fibre acceptance stage plan and fail-closed guards.

Only parsing/planning; no GPU, checkpoint read, CI or model inference.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
import unittest
from unittest.mock import patch
from scripts.x369_source_acceptance import parse_args,stage_plan,STAGE_ORDER

BASE=('--checkpoint','missing.pt','--sha256','0'*64,
      '--source-head','0'*40,'--output','unused.json')

class IndexedFibreStagePlan(unittest.TestCase):
    def test_single_authenticated_stage_can_be_selected(self):
        args=parse_args((*BASE,'--fibre-budget','189',
                         '--only-stages','admission','indexed_fibre'))
        self.assertEqual(stage_plan(args),('admission','indexed_fibre'))
        self.assertIn('indexed_fibre',STAGE_ORDER)
        self.assertEqual(args.fibre_budget,189)
        self.assertFalse(args.plan)

    def test_multi_stage_uses_existing_full_source_admission(self):
        args=parse_args((*BASE,'--fibre-budget','1000',
                         '--future-budget','1000',
                         '--simultaneous-weld'))
        selected=stage_plan(args)
        self.assertIn('indexed_fibre',selected)
        self.assertIn('future_consumer',selected)
        self.assertLess(selected.index('future_consumer'),
                        selected.index('indexed_fibre'))

    def test_no_implicit_audit_or_unfunded_stage(self):
        args=parse_args(BASE)
        self.assertNotIn('indexed_fibre',stage_plan(args))
        with self.assertRaises(SystemExit):
            parse_args((*BASE,'--only-stages','indexed_fibre'))
        with self.assertRaises(SystemExit):
            parse_args((*BASE,'--fibre-budget','188'))
        with self.assertRaises(SystemExit):
            parse_args((*BASE,'--fibre-budget','-1'))
        with self.assertRaises(SystemExit):
            parse_args((*BASE,'--fibre-budget','189','--width','6'))

    def test_control_face_audit_requires_authenticated_fibre_stage(self):
        a=parse_args((*BASE,'--fibre-budget','189',
                      '--face-seeds','4','--only-stages','admission','indexed_fibre'))
        self.assertEqual(a.face_seeds,4)
        self.assertEqual(stage_plan(a),('admission','indexed_fibre'))
        with self.assertRaises(SystemExit):
            parse_args((*BASE,'--face-seeds','1'))
        with self.assertRaises(SystemExit):
            parse_args((*BASE,'--fibre-budget','189','--face-seeds','4',
                        '--only-stages','admission'))
        with self.assertRaises(SystemExit):
            parse_args((*BASE,'--fibre-budget','189','--face-seeds','190'))
        with self.assertRaises(SystemExit):
            parse_args((*BASE,'--fibre-budget','189','--face-seeds','-1'))

    def test_original_stage13_chart_admission_requires_native_fibre_stage(self):
        a=parse_args((*BASE,'--fibre-budget','189','--stage13-seeds','4',
                      '--only-stages','admission','indexed_fibre'))
        self.assertEqual(a.stage13_seeds,4)
        self.assertIn('indexed_fibre',stage_plan(a))
        for more in (
          ('--stage13-seeds','1'),
          ('--fibre-budget','189','--stage13-seeds','190'),
          ('--fibre-budget','189','--stage13-seeds','-1'),
          ('--fibre-budget','189','--stage13-seeds','3',
           '--only-stages','admission')):
            with self.subTest(more=more):
                with self.assertRaises(SystemExit):
                    parse_args((*BASE,*more))

    def test_plan_never_authenticates_source(self):
        args=parse_args((*BASE,'--plan','--fibre-budget','189'))
        self.assertTrue(args.plan)
        self.assertIn('indexed_fibre',stage_plan(args))


if __name__=='__main__':
    unittest.main()
