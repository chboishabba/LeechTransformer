"""Source-level tests for native progress writer, independent of GPU hardware."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from scripts.native_training_progress import TrainingProgress


class NativeProgressTests(unittest.TestCase):
    def test_receipt_counts_and_no_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"progress.jsonl"
            seen=[]
            meter=TrainingProgress(label="test",device="cpu",total_steps=3,
                progress_every=2,jsonl=p,print_fn=lambda *a,**kw:seen.append(a[0]))
            meter.step(rows=2,sequence_length=8,epoch=1,loss=1.25)
            self.assertEqual(p.read_text(),"")
            meter.step(rows=1,sequence_length=8,epoch=1,loss=1.0)
            meter.step(rows=2,sequence_length=8,epoch=1,loss=.5)
            rows=[json.loads(x) for x in p.read_text().splitlines()]
            self.assertEqual([r["steps_completed"] for r in rows],[2,3])
            self.assertEqual(rows[-1]["target_tokens_completed"],35)
            self.assertEqual(rows[-1]["rows_completed"],5)
            self.assertEqual(rows[-1]["fraction"],1.)
            self.assertEqual(rows[-1]["train_loss"],.5)
            self.assertFalse(rows[-1]["source_model_quality_inferred"])
            self.assertFalse(rows[-1]["distributed_network_measured"])
            self.assertEqual(json.loads(p.with_suffix(".latest.json").read_text()),rows[-1])
            self.assertEqual(meter.summary()["next_token_targets"],35)
            self.assertEqual(len(seen),2)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"progress.jsonl"
            p.write_text("old data")
            with self.assertRaises(FileExistsError):
                TrainingProgress(label="test",device="cpu",total_steps=1,jsonl=p)
            self.assertEqual(p.read_text(),"old data")

    def test_bad_counts_rejected(self):
        with self.assertRaises(ValueError):
            TrainingProgress(label="test",device="cpu",total_steps=0)
        with self.assertRaises(ValueError):
            TrainingProgress(label="test",device="cpu",total_steps=1,progress_every=0)
        meter=TrainingProgress(label="test",device="cpu",total_steps=1,print_fn=lambda *a,**kw:None)
        with self.assertRaises(ValueError): meter.step(rows=1,sequence_length=0,epoch=1)
        with self.assertRaises(ValueError): meter.step(rows=0,sequence_length=2,epoch=1)

    def test_interval_nonnegative(self):
        meter=TrainingProgress(label="test",device="cpu",total_steps=1,print_fn=lambda *a,**kw:None)
        record=meter.step(rows=1,sequence_length=2,epoch=1,loss=0.)
        self.assertGreater(record["interval_steps_per_second"],0)
        self.assertEqual(record["estimated_remaining_train_seconds"],0)
        self.assertEqual(record["target_tokens_completed"],1)


if __name__=="__main__":
    unittest.main()
