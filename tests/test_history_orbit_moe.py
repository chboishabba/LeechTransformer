"""Regress a *real* independently learned history-bearing sparse MoE."""
import unittest
import torch
from models.history_orbit_moe import HistoryOrbitMoE

class HistoryOrbitMoETests(unittest.TestCase):
    def test_shapes_and_independent_expert_parameters(self):
        torch.manual_seed(41)
        m=HistoryOrbitMoE(d_model=24,num_experts=4,top_k=2,expert_width=48)
        self.assertEqual(len(m.experts),4)
        self.assertNotEqual(m.experts[0][0].weight.data_ptr(),m.experts[1][0].weight.data_ptr())
        x=torch.randn(3,5,24)
        y,h,idx,sign,w=m(x)
        self.assertEqual(tuple(y.shape),(3,5,24))
        self.assertEqual(tuple(h.shape),(3,4,24))
        self.assertEqual(tuple(idx.shape),(3,5,2))
        self.assertTrue(torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1))))
        self.assertTrue(bool(torch.all((sign==1)|(sign==-1))))

    def test_history_continuation_exact(self):
        torch.manual_seed(52)
        m=HistoryOrbitMoE(24,4,2,48).eval()
        x=torch.randn(2,7,24)
        full=m(x)
        first=m(x[:,:3])
        second=m(x[:,3:],first[1])
        self.assertTrue(torch.equal(torch.cat((first[0],second[0]),dim=1),full[0]))
        self.assertTrue(torch.equal(second[1],full[1]))
        self.assertTrue(torch.equal(torch.cat((first[2],second[2]),dim=1),full[2]))

    def test_gradients_reach_router_and_selected_experts(self):
        torch.manual_seed(63)
        m=HistoryOrbitMoE(24,4,2,48)
        x=torch.randn(8,4,24)
        y,h,indices,_,_=m(x,train_sign=True)
        (y.square().mean()+h.square().mean()).backward()
        self.assertGreater(float(m.router.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.orientation.weight.grad.abs().sum()),0)
        chosen=set(indices.reshape(-1).tolist())
        for i in chosen:
            self.assertGreater(float(m.experts[i][0].weight.grad.abs().sum()),0)

    def test_only_selected_history_updates(self):
        torch.manual_seed(18)
        m=HistoryOrbitMoE(24,5,1,48)
        x=torch.randn(1,24)
        old=torch.randn(1,5,24)
        _,new,ids,_,_=m.step(x,old)
        chosen=ids[0,0].item()
        for e in range(5):
            if e!=chosen:
                self.assertTrue(torch.equal(new[0,e],old[0,e]))

if __name__=="__main__":
    unittest.main()
