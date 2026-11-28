import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts import run_x369_registered_learning_campaign as c


class Campaign(unittest.TestCase):
    def test_progress_only_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            a=SimpleNamespace(directory=Path(d),device='cpu',dtype='float32',
                width=384,experts=4,steps=2,report_every=1,chunk_steps=1,seconds=80,
                lr=.001,weight_decay=.1,audit_only=False,verify_all=True)
            names=list(c.NAMES)
            plan=dict(schema=1,fixed_horizon=2,arms=[
                dict(name=f'{arm}-{seed}',contract=dict(arm=arm,seed=seed,
                    dtype='float32',width=384,experts=4,horizon=2))
                for seed in c.SEEDS for arm in names])
            calls=[]
            def fake_run(v):
                calls.append((v.arm,v.seed))
                entry=next(e for e in plan['arms'] if e['name']==f'{v.arm}-{v.seed}')
                v.output.write_bytes(b'fixture')
                c.write_json(v.receipt,dict(contract=entry['contract'],
                    parent_checkpoint_sha256=v.resume_sha256,first_step=1,last_step=1,
                    checkpoint_sha256=c.sha(v.output),initial_state_sha256='same',
                    final_model_sha256='fixture',status='CHUNK_CHECKPOINTED',
                    per_step_training_losses=[1.],sampled_trajectory=[],
                    classification='fixture',heldout_never_trained=True,
                    heldout_checkpoint_selection=False,
                    chunk_training_and_diagnostic_seconds=0.,chunk_peak_gpu_bytes=None))
            with patch.object(c,'registration',return_value=plan),patch.object(c,'run',side_effect=fake_run):
                for _ in range(2):
                    r=c.advance(a)
                    self.assertEqual(r['arms_required'],18)
                    self.assertEqual(r['arms_complete'],0)
                self.assertEqual(calls,[(names[0],41),(names[1],41)])
                self.assertFalse(r['heldout_selection'])
                a.audit_only=True
                c.advance(a)
                self.assertEqual(len(calls),2)
                (a.directory/plan['arms'][0]['name']/'chunk-000001.pt').write_bytes(b'tamper')
                with self.assertRaises(ValueError):c.advance(a)

if __name__=='__main__':unittest.main()
