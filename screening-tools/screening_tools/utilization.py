# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU and iGPU utilisation readers.

CPU utilisation comes from psutil on every OS. iGPU utilisation is read
from the Windows "GPU Engine" performance counters (the same counters Task
Manager uses) through the PDH API, called with ctypes. On other operating
systems the iGPU reading is reported as unavailable rather than estimated.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

import psutil

_ENGINE_RE = re.compile(r"pid_(\d+)_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_(\d+)_eng_(\d+)_engtype_(\w*)")


class CpuReader:
    def __init__(self):
        self.proc = psutil.Process()
        self.ncpu = psutil.cpu_count() or 1
        self.prime()

    def prime(self) -> None:
        # psutil keeps the system-wide reference point per thread, so the
        # sampling thread must call this before its first read.
        psutil.cpu_percent(None)
        self.proc.cpu_percent(None)

    def read(self) -> dict:
        return {
            "cpu_percent": psutil.cpu_percent(None),
            "process_cpu_percent": self.proc.cpu_percent(None) / self.ncpu,
        }


class GpuReader:
    """Interface: ``available`` and ``read()`` -> dict or None."""

    available = False
    reason = "iGPU utilisation is read from Windows performance counters; not available on this OS"

    def read(self) -> dict | None:
        return None

    def close(self) -> None:
        pass


class WindowsGpuReader(GpuReader):
    PDH_FMT_DOUBLE = 0x00000200
    PDH_MORE_DATA = 0x800007D2
    COUNTER = r"\GPU Engine(*)\Utilization Percentage"

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class FmtValue(ctypes.Structure):
            _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

        class FmtItem(ctypes.Structure):
            _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", FmtValue)]

        self._ct, self._FmtItem = ctypes, FmtItem
        pdh = ctypes.WinDLL("pdh.dll")
        pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        pdh.PdhGetFormattedCounterArrayW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        for fn in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData", "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
            getattr(pdh, fn).restype = wintypes.DWORD
        self.pdh = pdh
        self.query = wintypes.HANDLE()
        self.counter = wintypes.HANDLE()
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0:
            raise OSError("PdhOpenQueryW failed")
        if pdh.PdhAddEnglishCounterW(self.query, self.COUNTER, 0, ctypes.byref(self.counter)) != 0:
            pdh.PdhCloseQuery(self.query)
            raise OSError("GPU Engine performance counters are not available")
        pdh.PdhCollectQueryData(self.query)
        self.pid = os.getpid()
        self.available = True
        self.reason = ""

    def _items(self) -> list[tuple[str, float]]:
        ct = self._ct
        from ctypes import wintypes

        if self.pdh.PdhCollectQueryData(self.query) != 0:
            return []
        size, count = wintypes.DWORD(0), wintypes.DWORD(0)
        rc = self.pdh.PdhGetFormattedCounterArrayW(self.counter, self.PDH_FMT_DOUBLE, ct.byref(size), ct.byref(count), None)
        if rc != self.PDH_MORE_DATA or size.value == 0:
            return []
        buf = (ct.c_byte * size.value)()
        rc = self.pdh.PdhGetFormattedCounterArrayW(self.counter, self.PDH_FMT_DOUBLE, ct.byref(size), ct.byref(count), buf)
        if rc != 0:
            return []
        items = ct.cast(buf, ct.POINTER(self._FmtItem))
        out = []
        for i in range(count.value):
            it = items[i]
            if it.FmtValue.CStatus in (0, 1):  # PDH_CSTATUS_VALID_DATA / NEW_DATA
                out.append((it.szName or "", float(it.FmtValue.doubleValue)))
        return out

    def read(self) -> dict | None:
        engines: dict[tuple, float] = defaultdict(float)
        own: dict[tuple, float] = defaultdict(float)
        for name, value in self._items():
            m = _ENGINE_RE.search(name)
            if not m:
                continue
            pid, luid, phys, eng, etype = m.groups()
            key = (luid, phys, eng, etype or "other")
            engines[key] += value
            if int(pid) == self.pid:
                own[key] += value
        if not engines:
            return None
        by_type: dict[str, float] = defaultdict(float)
        for (_, _, _, etype), v in engines.items():
            by_type[etype] = max(by_type[etype], min(v, 100.0))
        return {
            "gpu_percent": round(max(by_type.values()), 2),
            "process_gpu_percent": round(min(max(own.values()), 100.0), 2) if own else 0.0,
            "gpu_by_engine": {k: round(v, 2) for k, v in sorted(by_type.items())},
        }

    def close(self) -> None:
        if self.available:
            self.pdh.PdhCloseQuery(self.query)
            self.available = False


def make_gpu_reader() -> GpuReader:
    if sys.platform == "win32":
        try:
            return WindowsGpuReader()
        except OSError as exc:
            reader = GpuReader()
            reader.reason = f"Windows GPU Engine counters unavailable: {exc}"
            return reader
    return GpuReader()
