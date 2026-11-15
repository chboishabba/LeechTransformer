"""R4b isolated CPU contracts; no SLR data/PG or F5 read in unit tests."""
from types import SimpleNamespace
import unittest

import torch

from models.model import LeechConfig, LeechGPT
from models.x369_fair_c3_residual import R4_GEOMETRIES
from scripts.run_x369_r4_three_arm_source_maxcut import (
    train_arm, validate_source_docs, validated_r1_summary,
)


class R4bSourceContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_source_authority_must_be_native_success_and_SHA_bound(self):
        sha = "1" * 64
        tok = "2" * 64
        r1 = {
            "kind": "x369_three_repo_physical_maxcut",
            "parser_authority": "slr_native_pg",
            "all_stage_exit_codes_zero": True,
            "F5_raw_source_opened": False,
            "original_tokenizer_sha256": tok,
            "original_SLR_source_export_sha256": sha,
            "stage_execution_receipts": {
                "slr_native_or_legacy_pg_cpu": {
                    "output_sha256": sha, "exit_code": 0}},
            "verified_parser_source_manifest": {
                "source_splits": ["development", "train"],
                "source_manifest_rows": 2,
                "F5_reserved_source_read": False},
        }
        validated_r1_summary(r1, jsonl_sha=sha, tokenizer_sha=tok)
        for key, value in (
                ("parser_authority", "sensiblaw_legacy"),
                ("F5_raw_source_opened", True),
                ("all_stage_exit_codes_zero", False),
                ("original_tokenizer_sha256", "3" * 64)):
            bad = {**r1, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validated_r1_summary(bad, jsonl_sha=sha, tokenizer_sha=tok)
        with self.assertRaises(ValueError):
            validated_r1_summary(r1, jsonl_sha="0" * 64, tokenizer_sha=tok)

    def test_document_contract_rejects_repeated_source_and_f5(self):
        cfg = LeechConfig(vocab_size=128, d_model=48, n_heads=2,
                          n_layers=2, block_size=8)
        def row(name, split, sha):
            return dict(id=name, split=split, source_sha=sha,
                        token_ids=[1,2,3,4], alignment=SimpleNamespace(
                            visibility="full_context"))
        rows = [row("a","train","x"), row("b","development","y")]
        a,b = validate_source_docs(rows, cfg)
        self.assertEqual((len(a), len(b)), (1,1))
        with self.assertRaises(ValueError):
            validate_source_docs([rows[0], row("b","development","x")], cfg)
        with self.assertRaises(ValueError):
            validate_source_docs([rows[0], row("c","reserved","z")], cfg)
        with self.assertRaises(ValueError):
            validate_source_docs([rows[0], {**rows[1], "alignment":
                SimpleNamespace(visibility="prefix")}], cfg)

    def test_three_arms_have_equal_trainable_capacity_and_real_updates(self):
        torch.manual_seed(41)
        cfg = LeechConfig(
            vocab_size=128, d_model=48, n_heads=2, n_layers=2,
            block_size=8, dropout=0., moe_experts=4, moe_top_k=2,
            moe_expert_width=24, moe_guarded_ffn=True,
            moe_route_geometry="c2_invariant", moe_deployed_parity=True,
            moe_even_width=24)
        source=LeechGPT(cfg).eval()
        train=[dict(id="fit",source_sha="0" * 64,token_ids=[1,2,3,4])]
        dev=[dict(id="dev",source_sha="1" * 64,token_ids=[5,6,7,8])]
        args=SimpleNamespace(device="cpu",lr=1e-4,steps=2,eval_every=1,
                             clip_norm=1.)
        results=[train_arm(source,g,train,dev,args=args,seed=41)
                 for g in R4_GEOMETRIES]
        self.assertEqual(len(results),3)
        self.assertEqual(
            len({row["residual_trainable_parameter_count"] for row in results}),1)
        self.assertTrue(all(row["residual_trainable_parameter_count"]==192
                            for row in results))
        self.assertTrue(all(len(row["step_trace"])==3 for row in results))
        self.assertTrue(all(row["best_step"] in (0,1,2) for row in results))
        self.assertTrue(all(row["initial_ce_nats"]==results[0]["initial_ce_nats"]
                            for row in results))


if __name__ == "__main__":
    unittest.main()
