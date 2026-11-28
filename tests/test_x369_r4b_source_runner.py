"""R4b isolated CPU contracts; no SLR data/PG or F5 read in unit tests."""
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

import torch

from models.model import LeechConfig, LeechGPT
from models.x369_fair_c3_residual import (
    R4_GEOMETRIES, clone_c2_with_zero_c3_residual,
    make_residual_delta_payload, restore_residual_delta)
from scripts.run_x369_r4_three_arm_source_maxcut import (
    train_arm, validate_source_docs, validated_r1_summary,
    physical_device_initial_parity,
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
        physical = physical_device_initial_parity(source, dev[0], "cpu")
        self.assertEqual(len(physical["bitwise_initial_logits_hidden_and_history_equal_for_arms"]),3)
        args=SimpleNamespace(device="cpu",lr=1e-4,steps=2,eval_every=1,
                             clip_norm=1.)
        with TemporaryDirectory() as folder:
            results=[train_arm(
                source,g,train,dev,args=args,seed=41,
                delta_dir=folder,
                source_sha256="a"*64,
                source_jsonl_sha256="b"*64)
                for g in R4_GEOMETRIES]
            for row in results:
                self.assertTrue(row["saved_delta_replay_same_device_verified"])
                self.assertEqual(row["reloaded_development_ce_nats"],
                                 row["best_dev_ce_nats"])
                path = Path(folder) / row["fitted_delta_artifact"]
                self.assertTrue(path.is_file())
                payload = torch.load(path, map_location="cpu", weights_only=True)
                self.assertFalse(payload["source_base_parameters_included"])
                self.assertEqual(
                    set(payload["residual_weights"]),
                    {f"blocks.{i}.moe.c3_residual_weight"
                     for i in range(cfg.n_layers)})
        self.assertEqual(len(results),3)
        self.assertEqual(
            len({row["residual_trainable_parameter_count"] for row in results}),1)
        self.assertTrue(all(row["residual_trainable_parameter_count"]==192
                            for row in results))
        self.assertTrue(all(len(row["step_trace"])==3 for row in results))
        self.assertTrue(all(row["best_step"] in (0,1,2) for row in results))
        self.assertTrue(all(row["initial_ce_nats"]==results[0]["initial_ce_nats"]
                            for row in results))


    def test_saved_delta_source_sha_geometry_tensor_tamper_fail_closed(self):
        torch.manual_seed(41)
        cfg = LeechConfig(
            vocab_size=128, d_model=48, n_heads=2, n_layers=2,
            block_size=8, dropout=0., moe_experts=4, moe_top_k=2,
            moe_expert_width=24, moe_guarded_ffn=True,
            moe_route_geometry="c2_invariant", moe_deployed_parity=True,
            moe_even_width=24)
        source = LeechGPT(cfg).eval()
        arm = clone_c2_with_zero_c3_residual(source, "c2_c3_residual")
        with torch.no_grad():
            arm.blocks[0].moe.c3_residual_weight.fill_(0.0125)
        payload = make_residual_delta_payload(
            arm, source_sha256="1"*64, source_jsonl_sha256="2"*64,
            seed=41, best_step=2)
        recovered = restore_residual_delta(
            source, payload, source_sha256="1"*64,
            source_jsonl_sha256="2"*64)
        self.assertTrue(torch.equal(
            arm.blocks[0].moe.c3_residual_weight,
            recovered.blocks[0].moe.c3_residual_weight))
        ids = torch.tensor([[1,2,3,4]], dtype=torch.long)
        with torch.no_grad():
            self.assertTrue(torch.equal(
                arm(ids, student_only=True)[0],
                recovered(ids, student_only=True)[0]))
        with self.assertRaises(ValueError):
            restore_residual_delta(
                source, payload, source_sha256="0"*64,
                source_jsonl_sha256="2"*64)
        with self.assertRaises(ValueError):
            restore_residual_delta(
                source, payload, source_sha256="1"*64,
                source_jsonl_sha256="0"*64)
        with self.assertRaises(ValueError):
            restore_residual_delta(
                source, {**payload, "geometry":"raw"},
                source_sha256="1"*64, source_jsonl_sha256="2"*64)
        with self.assertRaises(ValueError):
            restore_residual_delta(
                source, {**payload, "residual_weights":{}},
                source_sha256="1"*64, source_jsonl_sha256="2"*64)
        corrupt = dict(payload["residual_weights"])
        corrupt["blocks.0.moe.c3_residual_weight"] = torch.full(
            (1,96), float("nan"))
        with self.assertRaises(ValueError):
            restore_residual_delta(
                source, {**payload, "residual_weights": corrupt},
                source_sha256="1"*64, source_jsonl_sha256="2"*64)
        with self.assertRaises(ValueError):
            make_residual_delta_payload(
                arm, source_sha256="missing", source_jsonl_sha256="2"*64,
                seed=41, best_step=2)
        with self.assertRaises(ValueError):
            restore_residual_delta(
                source, {**payload, "feature_semantics": "old_fixed_mask_v1"},
                source_sha256="1"*64, source_jsonl_sha256="2"*64)
        with self.assertRaises(ValueError):
            restore_residual_delta(
                source, {**payload, "geometry_source_sha256": "0"*64},
                source_sha256="1"*64, source_jsonl_sha256="2"*64)


if __name__ == "__main__":
    unittest.main()
