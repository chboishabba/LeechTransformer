"""CPU fixture: exact resume, fixed learning split, no physical GPU claim."""
import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.run_x369_mod3_checkpointed import run


class CheckpointedMod3(unittest.TestCase):
    def test_fixed_horizon_resume_and_source_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def args(name,chunk=2,resume=None,sha=None,**kw):
                v=dict(arm='pair_mlp',seed=41,device='cpu',dtype='float64',
                    width=12,experts=4,steps=2,report_every=1,chunk_steps=chunk,
                    seconds=80,lr=.001,weight_decay=.1,resume=resume,
                    resume_sha256=sha,output=root/(name+'.pt'),receipt=root/(name+'.json'))
                v.update(kw)
                return argparse.Namespace(**v)
            full=run(args('full'))
            first=run(args('first',1))
            self.assertEqual(first['status'],'CHUNK_CHECKPOINTED')
            self.assertEqual(first['last_step'],1)
            second=run(args('second',1,root/'first.pt',first['checkpoint_sha256']))
            self.assertEqual(second['status'],'FIXED_HORIZON_COMPLETE')
            self.assertEqual(second['last_step'],2)
            self.assertEqual(second['final_model_sha256'],full['final_model_sha256'])
            self.assertEqual(second['sampled_trajectory'],full['sampled_trajectory'])
            self.assertTrue(second['heldout_never_trained'])
            self.assertFalse(second['heldout_checkpoint_selection'])
            self.assertFalse(second['physical_gpu_executed'])
            self.assertFalse(second['grokking_certified'])
            with self.assertRaises(ValueError):
                run(args('bad',1,root/'first.pt','0'*64))
            with self.assertRaises(ValueError):
                run(args('changed',1,root/'first.pt',first['checkpoint_sha256'],seed=42))


if __name__=='__main__':unittest.main()
