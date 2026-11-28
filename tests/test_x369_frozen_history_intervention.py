import unittest
import torch
from scripts.audit_x369_frozen_history_intervention import matched_nuisance,norm_audit,probe
from scripts.train_x369_mod3_codec_gpu import Learner
from scripts.audit_x369_typed_mod3_prediction_consumer import TASK_PAIRS


class HistoryIntervention(unittest.TestCase):
    def test_matched_per_history_geometry(self):
        torch.manual_seed(41)
        h=torch.randn(3,4,24,dtype=torch.float64);h[0,0]=0
        r=h.roll(1,-1)
        v=matched_nuisance(h,r,3701)
        self.assertTrue(torch.equal(v,matched_nuisance(h,r,3701)))
        self.assertFalse(torch.equal(v,r))
        q=norm_audit(h,r,v)
        self.assertLess(q['max_squared_norm_difference'],1e-10)
        self.assertLess(q['max_squared_displacement_difference'],1e-10)
        self.assertTrue(torch.equal(v[0,0],h[0,0]))

    def test_readonly_actual_source_fixture(self):
        torch.manual_seed(41);m=Learner(24,4).double().eval()
        before={k:v.clone() for k,v in m.state_dict().items()}
        R=torch.eye(24,dtype=torch.float64).roll(1,0)
        r=probe(m,None,R,torch.tensor(TASK_PAIRS),[3701])
        self.assertTrue(r['original_source_replay_exact'])
        self.assertTrue(r['no_cross_example_intervention'])
        self.assertFalse(r['semantic_opposition_inferred'])
        self.assertEqual(r['interventions']['baseline']['reroute']['KL_from_baseline'],0.)
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))

if __name__=='__main__':unittest.main()
