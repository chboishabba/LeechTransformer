"""Integration and checkpoint compatibility tests for opt-in history MoE."""
import unittest
import torch
from models.model import LeechConfig, LeechGPT


class LeechMoEIntegrationTests(unittest.TestCase):
    def config(self, experts=0):
        return LeechConfig(vocab_size=41,d_model=48,n_layers=2,n_heads=2,
                           block_size=12,dropout=0.0,lambda_geo=0.0,
                           moe_experts=experts,moe_top_k=2)

    def test_default_dense_state_dict_roundtrip_and_output_arity(self):
        torch.manual_seed(1)
        dense=LeechGPT(self.config()).eval()
        clone=LeechGPT(self.config()).eval()
        clone.load_state_dict(dense.state_dict(),strict=True)
        x=torch.tensor([[1,2,3]])
        out=dense(x)
        self.assertEqual(len(out),4)
        self.assertTrue(torch.equal(out[0],clone(x)[0]))
        self.assertEqual(len(dense(x,return_expert_histories=True)),5)
        self.assertIsNone(dense(x,return_expert_histories=True)[-1])

    def test_sparse_training_receives_expert_gradients(self):
        torch.manual_seed(2)
        model=LeechGPT(self.config(experts=4))
        tokens=torch.tensor([[1,2,3,4],[3,1,2,5]])
        logits,hidden,loss,cache,mem=model(
            tokens,targets=tokens,return_expert_histories=True)
        self.assertEqual(logits.shape,(2,4,41))
        self.assertEqual(len(mem),2)
        self.assertEqual(mem[0].shape,(2,4,48))
        loss.backward()
        self.assertTrue(any(
            p.grad is not None and float(p.grad.abs().sum())>0
            for p in model.blocks[0].moe.experts.parameters()))

    def test_stateful_cached_generation_matches_full(self):
        torch.manual_seed(3)
        model=LeechGPT(self.config(experts=4)).eval()
        seq=torch.tensor([[1,4,3,2,5]])
        with torch.no_grad():
            full=model(seq,return_expert_histories=True)
            cached=None
            histories=None
            partial=[]
            for t in range(seq.shape[1]):
                part=model(seq[:,t:t+1],past_key_values=cached,use_cache=True,
                           past_expert_histories=histories,
                           return_expert_histories=True)
                partial.append(part[0])
                cached, histories=part[3],part[4]
            joined=torch.cat(partial,1)
        self.assertTrue(torch.allclose(joined,full[0],rtol=1e-4,atol=1e-5))
        for h0,h1 in zip(histories,full[4]):
            self.assertTrue(torch.allclose(h0,h1,rtol=1e-4,atol=1e-5))

    def test_refuse_history_drop_during_cached_continuation(self):
        model=LeechGPT(self.config(experts=4)).eval()
        first=model(torch.tensor([[1]]),use_cache=True,
                    return_expert_histories=True)
        with self.assertRaises(ValueError):
            model(torch.tensor([[2]]),use_cache=True,past_key_values=first[3])


if __name__=="__main__":
    unittest.main()
