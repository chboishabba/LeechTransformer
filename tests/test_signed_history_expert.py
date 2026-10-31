"""Test signed history expert contract; no semantics assigned to sign."""
import unittest
import torch
from models.signed_history_expert import SignedHistoryExpert, signed_training_objective


class SignedHistoryTests(unittest.TestCase):
    def test_two_orientations_and_state_inversion(self):
        torch.manual_seed(25)
        net=SignedHistoryExpert(24,48).eval()
        x=torch.randn(2,3,24);h=torch.randn_like(x)
        plus=torch.ones(2,3,1);minus=-plus
        yp,hp,_,_=net(x,h,forced_sign=plus)
        ym,hm,_,_=net(-x,-h,forced_sign=minus)
        self.assertTrue(torch.equal(yp,-ym))
        self.assertTrue(torch.equal(hp,-hm))
        z,zh=net.recanonicalize(yp,hp,minus)
        zz,zzh=net.recanonicalize(z,zh,minus)
        self.assertTrue(torch.equal(zz,yp))
        self.assertTrue(torch.equal(zzh,hp))

    def test_antisymmetric_learned_router(self):
        torch.manual_seed(23)
        net=SignedHistoryExpert(24,48).eval()
        x=torch.randn(2,24);h=torch.randn_like(x)
        a=net(x,h)
        b=net(-x,-h)
        self.assertTrue(torch.allclose(a[3],-b[3],rtol=0,atol=1e-6))
        self.assertTrue(torch.equal(a[2],-b[2]))
        # The router switches chart, while the canonical expert's outputs
        # need not be odd (and GELU is not odd).

    def test_inverse_orientation_gradient_flow(self):
        torch.manual_seed(6)
        net=SignedHistoryExpert(24,48)
        x=torch.randn(4,24);h=torch.randn_like(x)
        loss, sign, _ = signed_training_objective(
            net,x,h,torch.randn_like(x),torch.randn_like(h),
            route_sign=-torch.ones(4,1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for p in (*net.expert.parameters(),*net.update.parameters()):
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.abs().sum()),0.0)
        # Hard sign selection has no router gradient; not falsely promoted
        self.assertIsNone(net.router.weight.grad)

    def test_router_depends_on_history(self):
        net=SignedHistoryExpert(24,48).eval()
        with torch.no_grad():
            net.router.weight.zero_();net.router.bias.zero_()
            net.router.weight[0,24]=1.
        x=torch.zeros(1,24)
        a=torch.zeros_like(x);a[0,0]=1
        b=-a
        self.assertEqual(net(x,a)[2].item(),1)
        self.assertEqual(net(x,b)[2].item(),-1)


if __name__=="__main__":
    unittest.main()
