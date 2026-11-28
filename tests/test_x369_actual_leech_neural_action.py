import unittest
import torch
from scripts.audit_x369_actual_leech_neural_action import matrix,retained,reopen,rawbytes,probe
from scripts.train_x369_mod3_codec_gpu import Learner
from scripts.audit_x369_typed_mod3_prediction_consumer import TASK_PAIRS


def hadamard():
    # A rational orthogonal fixture, not an actual Leech source receipt.
    H=[[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
    return [[('1/2' if H[i%4][j%4]>0 else '-1/2') if i//4==j//4 else '0'
             for j in range(24)] for i in range(24)]


class NativeRechart(unittest.TestCase):
    def test_retained_exact_roundtrip(self):
        R=matrix(hadamard(),torch.device('cpu'))
        for dt in (torch.float32,torch.float64):
            torch.manual_seed(41);x=torch.randn(2,48,dtype=dt)
            x[0,0]=-0.;x[0,1]=1.e-20
            p=retained(x,R)
            self.assertTrue(torch.equal(rawbytes(reopen(p,R)),rawbytes(x)))
        bad=hadamard();bad[0][0]='2'
        with self.assertRaises(ValueError):matrix(bad,torch.device('cpu'))

    def test_actual_learner_rechart_fixture(self):
        torch.manual_seed(41)
        m=Learner(24,4).double().eval()
        R=matrix(hadamard(),torch.device('cpu'))
        r=probe(m,None,R,torch.tensor(TASK_PAIRS,dtype=torch.long))
        self.assertTrue(r['all_original_step_squares_exact'])
        self.assertEqual(len(r['steps']),2)
        self.assertFalse(r['original_neural_lattice_membership_claimed'])
        self.assertFalse(r['semantic_causal_intervention_certified'])

if __name__=='__main__':unittest.main()
