"""Guard tests for receipt-targeted native continuation; no GPU needed."""
import copy
import json
import unittest
from pathlib import Path
from scripts.x369_source_acceptance import parse_args, stage_plan
from scripts.x369_original_receipt_targeted_native import verify_parent, native_targeted

PATH=Path(__file__).resolve().parents[1]/'docs/receipts/x369_original_native_minimal_consumer_20261128.json'
BASE=['--checkpoint','not-read.pt','--sha256','0'*64,'--source-head','0'*40,
      '--output','not-written.json','--receipt-targeted-multichart',str(PATH)]


class TargetedGuard(unittest.TestCase):
    def test_targeted_plan_and_conflicts(self):
        a=parse_args(BASE+['--fibre-budget','189','--only-stages','admission','indexed_fibre'])
        self.assertEqual(stage_plan(a),('admission','indexed_fibre'))
        for extra in (['--fibre-budget','189'],
                      ['--fibre-budget','190','--only-stages','admission','indexed_fibre'],
                      ['--fibre-budget','189','--only-stages','admission','indexed_fibre','--source-witness-seeds','4']):
            with self.assertRaises(SystemExit):
                parse_args(BASE+extra)

    def test_exact_parent_contract_and_source_owner_guards(self):
        old=json.loads(PATH.read_text())
        current=copy.deepcopy(old)
        current['source']['head']='new-audit-only-head'
        current['source']['files']['scripts/new_audit.py']='a'*64
        verify_parent(old,current)
        for key in ('checkpoint_sha256','model_sha256','config','execution_contract'):
            bad=copy.deepcopy(current)
            bad[key]=None
            with self.assertRaises(ValueError):
                verify_parent(old,bad)
        bad=copy.deepcopy(current)
        k=next(k for k in bad['source']['files'] if k.startswith('models/'))
        bad['source']['files'][k]='f'*64
        with self.assertRaises(ValueError):
            verify_parent(old,bad)

    def test_pinned_receipt_checked_before_model(self):
        with self.assertRaises(ValueError):
            native_targeted(None,b'{}',{})


if __name__=='__main__':
    unittest.main()
