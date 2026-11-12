"""Minimal native HIP/CUDA training progress and throughput receipts.

Only sampled at training-step boundaries; CUDA synchronize makes timings
completed-device wall time rather than enqueue-only host time. No CPU fallback.
JSONL is append-only, latest JSON is atomic; callbacks are observational only.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import time


class TrainingProgress:
    def __init__(self, *, label, device, total_steps, progress_every=5,
                 jsonl=None, print_fn=print):
        if total_steps <= 0 or progress_every <= 0:
            raise ValueError("total_steps and progress_every must be positive")
        self.label = label
        self.device = str(device)
        self.total_steps = int(total_steps)
        self.every = int(progress_every)
        self.jsonl = Path(jsonl) if jsonl is not None else None
        self.print_fn = print_fn
        self.steps = 0
        self.rows = 0
        self.target_tokens = 0
        self.last_report = 0
        self.last_tokens = 0
        self.last_rows = 0
        self.last_steps = 0
        self._sync()
        self.start = self.last_time = time.perf_counter()
        if self.jsonl is not None:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
            if self.jsonl.exists():
                raise FileExistsError("refuse overwrite of prior progress: " + str(self.jsonl))
            # Reserve the file before training, keeping earlier receipts immutable.
            self.jsonl.touch(exist_ok=False)

    def _sync(self):
        # The actual tensors and module are already on the requested device.
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize(self.device)

    def step(self, *, rows, sequence_length, epoch, loss=None, phase="train"):
        if rows <= 0 or sequence_length < 1:
            raise ValueError("batch rows must be positive and token length >=1")
        self.steps += 1
        self.rows += int(rows)
        # Next-token CE target count, not raw input token count.
        self.target_tokens += int(rows) * max(0, int(sequence_length) - 1)
        if self.steps % self.every != 0 and self.steps != self.total_steps:
            return
        self._sync()
        now = time.perf_counter()
        elapsed = max(now - self.start, 1e-12)
        interval = max(now - self.last_time, 1e-12)
        report = {
            "schema": 1, "kind": "native_training_progress",
            "label": self.label, "device": self.device, "phase": phase,
            "epoch": int(epoch), "steps_completed": self.steps,
            "steps_expected": self.total_steps, "fraction": self.steps / self.total_steps,
            "rows_completed": self.rows, "target_tokens_completed": self.target_tokens,
            "elapsed_train_seconds": elapsed, "interval_seconds": interval,
            "steps_per_second": self.steps / elapsed,
            "target_tokens_per_second": self.target_tokens / elapsed,
            "interval_steps_per_second": (self.steps-self.last_steps) / interval,
            "interval_target_tokens_per_second": (self.target_tokens-self.last_tokens) / interval,
            "estimated_remaining_train_seconds": max(0, self.total_steps-self.steps) * elapsed / self.steps,
            "train_loss": float(loss) if loss is not None else None,
            "timing_scope": "loop_elapsed_including_interepoch_validation_or_calibration",
            "total_wall_time_includes_initial_loading": False,
            "source_model_quality_inferred": False,
            "distributed_network_measured": False,
        }
        if self.device.startswith("cuda"):
            import torch
            report["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(self.device))
            report["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(self.device))
            report["gpu_memory_kind"] = "pytorch_cuda_allocator_not_total_device_usage"
        if self.jsonl is not None:
            with self.jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
                fp.flush()
            latest = self.jsonl.with_suffix(".latest.json")
            temp = latest.with_suffix(".tmp")
            temp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
            os.replace(temp, latest)
        self.print_fn("[{}] {}/{} ({:.1%}) steps | {:.2f} target tok/s | {:.3f} steps/s | {:.1f}s train elapsed | {:.1f}s estimated train remaining | GPU allocated peak {} MiB".format(
            self.label, self.steps, self.total_steps, report["fraction"],
            report["interval_target_tokens_per_second"], report["interval_steps_per_second"],
            elapsed, report["estimated_remaining_train_seconds"],
            round(report.get("gpu_peak_allocated_bytes", 0) / (1024 * 1024), 1)),
            flush=True)
        self.last_time = now
        self.last_steps = self.steps
        self.last_rows = self.rows
        self.last_tokens = self.target_tokens
        self.last_report = self.steps
        return report

    def summary(self):
        self._sync()
        elapsed = max(time.perf_counter()-self.start, 1e-12)
        return {
            "steps": self.steps, "rows": self.rows,
            "next_token_targets": self.target_tokens,
            "loop_elapsed_seconds": elapsed,
            "loop_steps_per_second": self.steps/elapsed,
            "loop_target_tokens_per_second": self.target_tokens/elapsed,
            "timing_scope": "loop_elapsed_including_interepoch_validation_or_calibration",
            "memory_scope": "pytorch_allocator_not_system_memory"
        }
