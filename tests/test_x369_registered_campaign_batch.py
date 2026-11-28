import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts import run_x369_registered_campaign_batch as b


class Batch(unittest.TestCase):
    def test_existing_registration_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=SimpleNamespace(directory=root,chunks=2,chunk_steps=100,seconds=80,hard_timeout=90)
            c=dict(width=384,experts=4,dtype='float32',horizon=2000,report_every=100,lr=.001,weight_decay=.1)
            p=dict(runner_device='gfx803',heldout_selection=False,arms=[dict(contract=c.copy()) for _ in range(18)])
            f=root/'registration.json';f.write_text(json.dumps(p))
            old=f.read_bytes();cmd=b.command(a)
            self.assertEqual(cmd[cmd.index('--device')+1],'gfx803')
            self.assertEqual(cmd[cmd.index('--steps')+1],'2000')
            result=dict(status='FIXED_CAMPAIGN_COMPLETE',arms_required=18,arms_complete=18,
                        heldout_selection=False,arms=[dict(completed_steps=2000) for _ in range(18)])
            with patch.object(b.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(result),stderr='')) as r:
                b.run(a)
                self.assertEqual(r.call_count,1)
                self.assertEqual(r.call_args.kwargs['timeout'],90)
            self.assertEqual(f.read_bytes(),old)
            p['arms'][-1]['contract']['horizon']=2010
            f.write_text(json.dumps(p))
            with self.assertRaises(ValueError):b.command(a)

if __name__=='__main__':unittest.main()
