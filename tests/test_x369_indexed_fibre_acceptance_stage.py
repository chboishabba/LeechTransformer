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

    def test_marked_seed_original_source_audit_requires_owner_stage(self):
        args=parse_args((*BASE,'--fibre-budget','189',
            '--marked-seed-budget','189',
            '--only-stages','admission','indexed_fibre'))
        self.assertEqual(args.marked_seed_budget,189)
        self.assertEqual(stage_plan(args),('admission','indexed_fibre'))
        for more in (('--marked-seed-budget','1'),
                     ('--fibre-budget','189','--marked-seed-budget','190'),
                     ('--fibre-budget','189','--marked-seed-budget','-1'),
                     ('--fibre-budget','189','--marked-seed-budget','3',
                      '--only-stages','admission')):
            with self.subTest(options=more):
                with self.assertRaises(SystemExit):
                    parse_args((*BASE,*more))

    def test_closed_finite_quotient_export_requires_indexed_fibre(self):
        a=parse_args((*BASE,'--fibre-budget','189',
            '--fibre-realization',
            '--only-stages','admission','indexed_fibre'))
        self.assertTrue(a.fibre_realization)
        self.assertIn('indexed_fibre',stage_plan(a))
        with self.assertRaises(SystemExit):
            parse_args((*BASE,'--fibre-realization'))
        with self.assertRaises(SystemExit):
            parse_args((*BASE,'--fibre-budget','189',
                         '--fibre-realization','--only-stages','admission'))

    def test_original_source_signed_defect_requires_authenticated_stage(self):
        a=parse_args((*BASE,'--fibre-budget','189',
            '--signed-defect-seeds','4','--signed-defect-depth','2',
            '--only-stages','admission','indexed_fibre'))
        self.assertEqual(a.signed_defect_seeds,4)
        self.assertEqual(a.signed_defect_depth,2)
        self.assertIn('indexed_fibre',stage_plan(a))
        for options in (
            ('--signed-defect-seeds','1'),
            ('--fibre-budget','189','--signed-defect-seeds','190'),
            ('--fibre-budget','189','--signed-defect-seeds','-1'),
            ('--fibre-budget','189','--signed-defect-seeds','2',
             '--only-stages','admission'),
            ('--fibre-budget','189','--signed-defect-seeds','4',
             '--signed-defect-depth','3')):
            with self.subTest(options=options):
                with self.assertRaises(SystemExit):
                    parse_args((*BASE,*options))

    def test_original_six_face_defect_attribution_guard(self):
        a=parse_args((*BASE,'--fibre-budget','189',
            '--face-seeds','3','--source-face-attribution',
            '--only-stages','admission','indexed_fibre'))
        self.assertEqual(a.face_seeds,3)
        self.assertTrue(a.source_face_attribution)
        self.assertIn('indexed_fibre',stage_plan(a))
        for options in (
            ('--source-face-attribution',),
            ('--fibre-budget','189','--source-face-attribution'),
            ('--fibre-budget','189','--face-seeds','3',
             '--source-face-attribution','--only-stages','admission')):
            with self.subTest(options=options):
                with self.assertRaises(SystemExit):
                    parse_args((*BASE,*options))

    def test_one_pass_original_marked_consumer_maxcut_guard(self):
        a=parse_args((*BASE,'--fibre-budget','189',
                      '--marked-consumer-maxcut',
                      '--only-stages','admission','indexed_fibre'))
        self.assertTrue(a.marked_consumer_maxcut)
        self.assertEqual(stage_plan(a),('admission','indexed_fibre'))
        for bad in (
            ('--marked-consumer-maxcut',),
            ('--fibre-budget','189','--marked-consumer-maxcut',
             '--only-stages','admission')):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit):
                    parse_args((*BASE,*bad))

    def test_arbitrary_original_source_word_residual_acceptance_guard(self):
        args=parse_args((*BASE,'--fibre-budget','189',
            '--word-residual-seeds','4','--word-residual-depth','4',
            '--only-stages','admission','indexed_fibre'))
        self.assertEqual(args.word_residual_seeds,4)
        self.assertEqual(args.word_residual_depth,4)
        self.assertEqual(stage_plan(args),('admission','indexed_fibre'))
        for options in (
            ('--word-residual-seeds','1'),
            ('--fibre-budget','189','--word-residual-seeds','190'),
            ('--fibre-budget','189','--word-residual-seeds','-1'),
            ('--fibre-budget','189','--word-residual-seeds','4',
             '--word-residual-depth','5'),
            ('--fibre-budget','189','--word-residual-seeds','4',
             '--only-stages','admission')):
            with self.subTest(options=options):
                with self.assertRaises(SystemExit):
                    parse_args((*BASE,*options))

    def test_original_source_witness_first_probe_authentication_guard(self):
        a=parse_args((*BASE,'--fibre-budget','189',
            '--source-witness-seeds','4',
            '--only-stages','admission','indexed_fibre'))
        self.assertEqual(a.source_witness_seeds,4)
        self.assertIn('indexed_fibre',stage_plan(a))
        for options in (
            ('--source-witness-seeds','1'),
            ('--fibre-budget','189','--source-witness-seeds','190'),
            ('--fibre-budget','189','--source-witness-seeds','-1'),
            ('--fibre-budget','189','--source-witness-seeds','4',
             '--only-stages','admission')):
            with self.subTest(options=options):
                with self.assertRaises(SystemExit):
                    parse_args((*BASE,*options))

    def test_plan_never_authenticates_source(self):
        args=parse_args((*BASE,'--plan','--fibre-budget','189'))
        self.assertTrue(args.plan)
        self.assertIn('indexed_fibre',stage_plan(args))


if __name__=='__main__':
    unittest.main()
