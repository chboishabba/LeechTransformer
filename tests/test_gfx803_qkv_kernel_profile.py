"""CPU-only invariants for the QKV profiler micro-repro."""
import unittest

import torch

from models.model import LeechConfig, LeechGPT
from scripts.profile_gfx803_leech_qkv import qkv_contract, qkv_input


class QkvProfileContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model=LeechGPT(LeechConfig(vocab_size=37,d_model=48,n_heads=2,
            n_layers=2,block_size=8,dropout=0.,lambda_geo=0.)).eval()
        self.ids=torch.randint(0,37,(2,5))

    def test_input_matches_the_real_embedding_and_ln1_path(self):
        actual=qkv_input(self.model,self.ids)
        expected=self.model.blocks[0].ln1(
            self.model.tok_emb(self.ids)+self.model.pos_emb[:,:5,:])
        self.assertTrue(torch.equal(actual,expected))

    def test_gemm_contract_is_derived_from_the_real_linear_module(self):
        contract=qkv_contract(self.model,self.ids)
        self.assertEqual(contract["logical_input_shape_btd"],[2,5,48])
        self.assertEqual(contract["logical_weight_shape_out_in"],[144,48])
        self.assertEqual(contract["logical_output_shape_btd"],[2,5,144])
        self.assertEqual(contract["flattened_gemm_m_n_k"],[10,144,48])

    def test_qkv_input_rejects_context_overflow(self):
        with self.assertRaises(ValueError):
            qkv_input(self.model,torch.randint(0,37,(1,9)))


if __name__ == "__main__":
    unittest.main()
