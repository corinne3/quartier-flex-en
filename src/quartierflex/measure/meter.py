"""
meter.py: the compute-energy meter.

The problem
-----------
We want to know how much electricity a strategy consumes to DECIDE.
On Windows, without dedicated hardware, we cannot read the processor's
power directly. So we combine several measurements:

1. CPU time (reliable, measured by the OS)
   - of the Python process (rules, ML, harness)            -> cpu_s_self
   - of the Ollama processes (where the LLM actually runs!) -> cpu_s_external
   Crucial point: the LLM does NOT run in Python but in the Ollama
   server. If we measured only Python, an LLM agent would look free!

2. Conversion to energy (estimate, "TDP" method)
       energy (kWh) = CPU seconds × (TDP / number of logical cores) / 3,600,000
   Same principle as CodeCarbon without a hardware sensor.

3. Machine-independent metrics (comparable anywhere)
   - number of decisions, LLM calls, input/output tokens, wall time.
   They let someone else redo the calculation with THEIR machine
   or with another factor (e.g. an LLM in a data center).

4. Optional: CodeCarbon (whole machine, for cross-checking), see maybe_codecarbon.

The "buckets"
-------------
Each measurement is filed in a bucket:
- "setup"     : cost paid ONCE (training a model, having an LLM generate
                code). To be amortized over time.
- "shared"    : recurring cost SHARED by the whole portfolio (one forecast/hour).
- "per_house" : recurring cost PER HOME (one decision/home/hour).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import psutil

from ..config import ComputeParams

BUCKETS = ("setup", "shared", "per_house")


@dataclass
class Bucket:
    wall_s: float = 0.0            # elapsed wall time
    cpu_s_self: float = 0.0        # CPU time of the Python process
    cpu_s_external: float = 0.0    # CPU time of the watched processes (Ollama), retained value
    cpu_s_ext_process: float = 0.0 # ... measured process by process (method 1)
    cpu_s_ext_system: float = 0.0  # ... measured on the whole machine (method 2)
    llm_active_s: float = 0.0      # compute time reported by Ollama itself (method 3)
    n_calls: int = 0               # number of measured passes (decisions...)
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def cpu_s(self) -> float:
        return self.cpu_s_self + self.cpu_s_external


@dataclass
class Meter:
    compute: ComputeParams = field(default_factory=ComputeParams)
    watch: tuple[str, ...] = ("ollama",)   # extra process names to watch
    buckets: dict[str, Bucket] = field(default_factory=lambda: {b: Bucket() for b in BUCKETS})

    def __post_init__(self) -> None:
        self._me = psutil.Process(os.getpid())
        self._ext: list[psutil.Process] = []
        self._ext_refresh_at = 0.0
        self._stack: list[str] = []
        self._llm_wait_s = 0.0      # time spent waiting for the LLM server
        self.n_logical_cores = psutil.cpu_count(logical=True) or 1
        self.ext_names: list[str] = []
        self.ext_readable = False
        self.idle_rate = 0.0        # machine background activity (CPU s per s)
        if self.watch:
            self._diagnose_external()
            self.idle_rate = self._calibrate_background()

    # ------------------------------------------------------------ Diagnostic
    def _diagnose_external(self) -> None:
        """
        Checks that we can READ the CPU time of the Ollama processes.
        On Windows, if Ollama runs as a system service or with different
        permissions, psutil gets "access denied": method 1 would return 0.
        That is what happened on the first run -> method 2 as fallback.
        """
        for p in self._external_procs():
            try:
                self.ext_names.append(p.name())
                p.cpu_times()
                self.ext_readable = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    @staticmethod
    def _system_busy_s() -> float:
        """"Busy" CPU seconds of the WHOLE machine, all cores summed."""
        t = psutil.cpu_times()
        idle = t.idle + getattr(t, "iowait", 0.0)
        return sum(t) - idle

    def _calibrate_background(self, seconds: float = 2.0) -> float:
        """Measures background activity (antivirus, system...) to subtract it later."""
        b0, w0 = self._system_busy_s(), time.perf_counter()
        time.sleep(seconds)
        b1, w1 = self._system_busy_s(), time.perf_counter()
        return max(0.0, (b1 - b0) / (w1 - w0))

    # ------------------------------------------------------------------ CPU
    def _external_procs(self) -> list[psutil.Process]:
        """List (cached 5 s) of Ollama processes: ollama.exe, runners..."""
        now = time.monotonic()
        if now >= self._ext_refresh_at:
            procs = []
            for p in psutil.process_iter(["name"]):
                name = (p.info.get("name") or "").lower()
                if any(w in name for w in self.watch):
                    procs.append(p)
            self._ext = procs
            self._ext_refresh_at = now + 5.0
        return self._ext

    @staticmethod
    def _cpu_of(p: psutil.Process) -> float:
        """CPU time (user + system) of an external process, in seconds."""
        try:
            t = p.cpu_times()
            return t.user + t.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0

    def _snapshot(self) -> tuple[float, float, float, float]:
        ext = sum(self._cpu_of(p) for p in self._external_procs()) if self.watch else 0.0
        sysb = self._system_busy_s() if self.watch else 0.0
        # time.process_time() = CPU time of OUR process (all threads),
        # best resolution available from Python.
        return time.perf_counter(), time.process_time(), ext, sysb

    @contextmanager
    def measure(self, bucket: str):
        """
        Usage:
            with meter.measure("per_house"):
                action = strategy.decide(obs)
        Everything that happens in the block (including inside Ollama) is counted.
        """
        self._stack.append(bucket)
        wait0 = self._llm_wait_s
        w0, c0, e0, s0 = self._snapshot()
        try:
            yield
        finally:
            w1, c1, e1, s1 = self._snapshot()
            b = self.buckets[bucket]
            wall = w1 - w0
            b.wall_s += wall
            # Resolution problem: the OS CPU clock ticks in steps of
            # ~10 to 16 ms. A "rule" decision takes a few microseconds:
            # its measured CPU time would be 0 (or 16 ms by bad luck).
            # Solution: the Python code in this block runs without waiting (no
            # input/output), EXCEPT while waiting for the LLM's replies.
            # So Python CPU >= (wall time - LLM wait). We take the max
            # of the two estimates (the max also handles multi-threading).
            active = max(0.0, wall - (self._llm_wait_s - wait0))
            self_cpu = max(c1 - c0, active)
            b.cpu_s_self += self_cpu
            # LLM cost (Ollama processes), two methods:
            #  1. CPU time of the Ollama processes (precise, but may get "access denied");
            #  2. CPU time of the WHOLE machine - our process - background activity
            #     (always readable; assumes nothing else is running: close your apps).
            # We keep the larger one: method 1 can only underestimate.
            proc = max(0.0, e1 - e0)
            system = max(0.0, (s1 - s0) - (c1 - c0) - self.idle_rate * wall) if self.watch else 0.0
            b.cpu_s_ext_process += proc
            b.cpu_s_ext_system += system
            b.cpu_s_external += max(proc, system)
            b.n_calls += 1
            self._stack.pop()

    # ------------------------------------------------------------------ LLM
    def record_llm(self, tokens_in: int, tokens_out: int, duration_s: float = 0.0,
                   active_s: float = 0.0) -> None:
        """
        Called by the harness on each LLM call; filed in the current bucket.
        active_s = compute time reported by Ollama (prompt reading + generation).
        """
        self._llm_wait_s += duration_s
        bucket = self._stack[-1] if self._stack else "setup"
        b = self.buckets[bucket]
        b.llm_active_s += active_s
        b.llm_calls += 1
        b.tokens_in += int(tokens_in)
        b.tokens_out += int(tokens_out)

    # ---------------------------------------------------------------- Energy
    def energy_kwh(self, bucket: str) -> float:
        """TDP estimate: CPU seconds × power per core, × PUE."""
        b = self.buckets[bucket]
        w_per_core = self.compute.cpu_tdp_w / self.n_logical_cores
        joules = b.cpu_s * w_per_core * self.compute.pue
        return joules / 3.6e6

    def energy_kwh_wattmeter(self, bucket: str, measured_w: float) -> float:
        """
        "Wattmeter" mode: if you measured the machine's power during
        the run with a plug-in wattmeter, energy = (measured P - idle P) × duration.
        """
        b = self.buckets[bucket]
        return max(0.0, measured_w - self.compute.idle_power_w) * b.wall_s / 3.6e6

    def diagnostics(self) -> dict:
        return {"ollama_processes": self.ext_names, "ollama_cpu_readable": self.ext_readable,
                "background_cpu_rate": round(self.idle_rate, 3)}

    def report(self) -> dict:
        out = {}
        for name, b in self.buckets.items():
            out[name] = {
                "wall_s": b.wall_s,
                "cpu_s_self": b.cpu_s_self,
                "cpu_s_external": b.cpu_s_external,
                "cpu_s_ext_process": b.cpu_s_ext_process,
                "cpu_s_ext_system": b.cpu_s_ext_system,
                "llm_active_s": b.llm_active_s,
                # Method 3 (cross-check): while computing, Ollama occupies the whole
                # processor -> energy ≈ active time × TDP.
                "energy_kwh_llm_active": b.llm_active_s * self.compute.cpu_tdp_w * self.compute.pue / 3.6e6,
                "n_calls": b.n_calls,
                "llm_calls": b.llm_calls,
                "tokens_in": b.tokens_in,
                "tokens_out": b.tokens_out,
                "energy_kwh": self.energy_kwh(name),
            }
        return out


# =============================================================================
# CodeCarbon (optional)
# =============================================================================
@contextmanager
def maybe_codecarbon(project_name: str, enabled: bool = True):
    """
    If CodeCarbon is installed (pip install codecarbon) and enabled, measures the
    WHOLE machine during the block. Returns a dict filled on exit:
        {"energy_kwh": ..., "emissions_kg": ...}   or {} if unavailable.

    Why "whole machine"? CodeCarbon cannot isolate a process.
    => Close other applications during measurements!
    On Windows without a sensor, it also estimates from the TDP: it is a
    CROSS-CHECK of our estimator, not absolute truth.
    """
    result: dict = {}
    tracker = None
    if enabled:
        try:
            from codecarbon import OfflineEmissionsTracker

            tracker = OfflineEmissionsTracker(
                project_name=project_name,
                country_iso_code="FRA",   # French electricity mix
                log_level="error",
                save_to_file=False,
            )
            tracker.start()
        except Exception:
            tracker = None
    try:
        yield result
    finally:
        if tracker is not None:
            try:
                emissions = tracker.stop()
                data = tracker.final_emissions_data
                result["energy_kwh"] = float(getattr(data, "energy_consumed", 0.0))
                result["emissions_kg"] = float(emissions or 0.0)
            except Exception:
                pass
