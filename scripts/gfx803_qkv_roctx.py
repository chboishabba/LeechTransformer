"""Minimal ROCTX marker for original unhooked QKV forced-solution runs.

Copied conceptually from the existing gfx803 Leech profiler. Does not import
the model or alter execution. Missing marker library is reported as a missing
attribution surface; the trace join subsequently rejects it.
"""
from __future__ import annotations

import ctypes
import os


class RoctxRange:
    def __init__(self, label: str):
        self.label = label.encode("utf-8")
        self.lib = None
        self.available = False
        for name in (os.environ.get("ROCTX_LIBRARY", ""),
                     "librocprofiler-sdk-roctx.so.1",
                     "librocprofiler-sdk-roctx.so",
                     "libroctx64.so", "libroctx64.so.1"):
            if not name:
                continue
            try:
                candidate = ctypes.CDLL(name)
            except OSError:
                continue
            if hasattr(candidate, "roctxRangePushA") and hasattr(candidate, "roctxRangePop"):
                candidate.roctxRangePushA.argtypes = [ctypes.c_char_p]
                candidate.roctxRangePushA.restype = ctypes.c_int
                candidate.roctxRangePop.restype = ctypes.c_int
                self.lib = candidate
                self.available = True
                break

    def __enter__(self):
        if self.available:
            if hasattr(self.lib, "roctxProfilerResume"):
                self.lib.roctxProfilerResume(ctypes.c_uint64(0))
            self.lib.roctxRangePushA(self.label)
        return self

    def __exit__(self, *exc):
        if self.available:
            self.lib.roctxRangePop()
            if hasattr(self.lib, "roctxProfilerPause"):
                self.lib.roctxProfilerPause(ctypes.c_uint64(0))
