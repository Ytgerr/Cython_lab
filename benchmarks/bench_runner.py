"""
Benchmark runner — compares four implementations:
  1. Pure Python (lists / random module)
  2. Cython untyped (compiled, PyObject* everywhere)
  3. Cython typed   (cdef + typed memoryviews + nogil + libc rand)
  4. C              (binary, compiled from src/c_impl/matrix_ops.c)

Operations:
  * matmul        — O(n^3) triple loop
  * matrix_add    — O(n^2) double loop
  * monte_carlo   — dice game simulation (roll 2 dice, win if sum >= 7)

Usage:
    python benchmarks/bench_runner.py [--sizes 64 128] [--repeats 5]
    python benchmarks/bench_runner.py --mc-samples 1000000

Output:
    benchmarks/results/bench_results.csv
    benchmarks/results/bench_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import tracemalloc
from typing import Callable, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Import Python / Cython implementations
# ---------------------------------------------------------------------------
from src.py_impl.matrix_ops import (
    matmul as py_matmul,
    matrix_add as py_add,
    simulate_dice_game as py_mc,
)

try:
    from src.cy_untyped.matrix_ops_cy import (
        matmul as cy_u_matmul,
        matrix_add as cy_u_add,
        simulate_dice_game as cy_u_mc,
    )
    CY_UNTYPED_OK = True
except ImportError:
    CY_UNTYPED_OK = False
    print("[WARN] cy_untyped not built — run: python setup.py build_ext --inplace")

try:
    from src.cy_typed.matrix_ops_typed import (
        matmul as cy_t_matmul,
        matrix_add as cy_t_add,
        simulate_dice_game as cy_t_mc,
    )
    CY_TYPED_OK = True
except ImportError:
    CY_TYPED_OK = False
    print("[WARN] cy_typed not built — run: python setup.py build_ext --inplace")


# ---------------------------------------------------------------------------
# C binary: compile + run
# ---------------------------------------------------------------------------

C_SRC  = os.path.join(ROOT, "src", "c_impl", "matrix_ops.c")
C_EXE  = os.path.join(ROOT, "src", "c_impl", "matrix_bench.exe")

# MSVC paths (same ones used by setup.py)
_MSVC_CL = (
    r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
    r"\VC\Tools\MSVC\14.50.35717\bin\HostX86\x64\cl.exe"
)
_MSVC_INCS = [
    r"/I C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\ucrt",
    r"/I C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\um",
    r"/I C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\shared",
    r"/I C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\include",
]
_MSVC_LIBS = [
    r"/LIBPATH:C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\ucrt\x64",
    r"/LIBPATH:C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\um\x64",
    r"/LIBPATH:C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\lib\x64",
]


def _compile_c() -> bool:
    """Compile matrix_ops.c to a standalone exe. Returns True on success."""
    # Try gcc first (simpler)
    gcc_cmd = ["gcc", "-O2", "-o", C_EXE, C_SRC, "-lm"]
    try:
        r = subprocess.run(gcc_cmd, capture_output=True, timeout=30)
        if r.returncode == 0:
            print("[C] Compiled with gcc")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fall back to MSVC cl.exe
    if not os.path.exists(_MSVC_CL):
        print("[WARN] C compiler not found (gcc or MSVC). Skipping C benchmark.")
        return False

    inc_str = " ".join(_MSVC_INCS)
    lib_str = " ".join(f'"{l}"' for l in _MSVC_LIBS)
    cmd = (
        f'"{_MSVC_CL}" /O2 /nologo {inc_str} '
        f'"{C_SRC}" /Fe:"{C_EXE}" /link {lib_str}'
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
    if r.returncode == 0:
        print("[C] Compiled with MSVC")
        return True
    print(f"[WARN] C compilation failed:\n{r.stderr.decode(errors='replace')}")
    return False


def _run_c_bench() -> List[Dict]:
    """Run the C binary, parse its JSON output."""
    if not os.path.exists(C_EXE):
        if not _compile_c():
            return []
    try:
        r = subprocess.run([C_EXE], capture_output=True, timeout=120)
        if r.returncode != 0:
            print(f"[WARN] C binary failed: {r.stderr.decode(errors='replace')}")
            return []
        return json.loads(r.stdout.decode())
    except Exception as e:
        print(f"[WARN] C benchmark error: {e}")
        return []


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def measure(fn: Callable, *args, repeats: int = 5) -> Tuple[float, int]:
    """Returns (best_wall_seconds, peak_memory_bytes)."""
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
    return best, peak


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_list_matrix(n: int) -> List[List[float]]:
    rng = np.random.default_rng(42)
    return rng.random((n, n)).tolist()


def make_np_matrix(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.ascontiguousarray(rng.random((n, n)), dtype=np.float64)


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

def run_benchmarks(sizes: List[int], repeats: int, mc_samples: int) -> List[Dict]:
    records: List[Dict] = []

    # ── Matrix operations ──────────────────────────────────────────────────
    for N in sizes:
        print(f"\n{'='*60}")
        print(f"  Matrix size: {N}x{N}  (repeats={repeats})")
        print(f"{'='*60}")

        A_list = make_list_matrix(N)
        B_list = make_list_matrix(N)
        A_np   = make_np_matrix(N)
        B_np   = make_np_matrix(N)

        matrix_impls = [("python", py_matmul, py_add, A_list, B_list)]
        if CY_UNTYPED_OK:
            matrix_impls.append(("cy_untyped", cy_u_matmul, cy_u_add, A_list, B_list))
        if CY_TYPED_OK:
            matrix_impls.append(("cy_typed", cy_t_matmul, cy_t_add, A_np, B_np))

        for (name, fn_mm, fn_add, A, B) in matrix_impls:
            skip_matmul = (name == "python" and N > 256)

            if skip_matmul:
                t_mm, m_mm = float("nan"), 0
                print(f"  [{name:12s}] matmul        SKIPPED (N>{256})")
            else:
                t_mm, m_mm = measure(fn_mm, A, B, repeats=repeats)
                print(f"  [{name:12s}] matmul        {t_mm*1000:10.3f} ms  "
                      f"mem={m_mm/1024:.1f} KB")

            t_add, m_add = measure(fn_add, A, B, repeats=repeats)
            print(f"  [{name:12s}] matrix_add    {t_add*1000:10.3f} ms  "
                  f"mem={m_add/1024:.1f} KB")

            for op, t, m in [("matmul", t_mm, m_mm),
                              ("matrix_add", t_add, m_add)]:
                records.append({"impl": name, "N": N, "op": op,
                                 "time_s": t, "mem_bytes": m})

    # ── Monte Carlo dice game ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Monte Carlo dice game  (n={mc_samples:,} rounds, repeats={repeats})")
    print(f"{'='*60}")

    mc_impls = [("python", py_mc)]
    if CY_UNTYPED_OK:
        mc_impls.append(("cy_untyped", cy_u_mc))
    if CY_TYPED_OK:
        mc_impls.append(("cy_typed", cy_t_mc))

    for (name, fn_mc) in mc_impls:
        t_mc, m_mc = measure(fn_mc, mc_samples, repeats=repeats)
        win_prob = fn_mc(mc_samples)
        print(f"  [{name:12s}] monte_carlo   {t_mc*1000:10.3f} ms  "
              f"mem={m_mc/1024:.1f} KB  win_prob~{win_prob:.4f} (true=0.5833)")
        records.append({"impl": name, "N": mc_samples, "op": "monte_carlo",
                         "time_s": t_mc, "mem_bytes": m_mc, "result": win_prob})

    # ── C standalone binary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  C standalone binary")
    print(f"{'='*60}")
    c_records = _run_c_bench()
    for r in c_records:
        op   = r["op"]
        N    = r["N"]
        t_s  = r["time_s"]
        print(f"  [{'c':12s}] {op:14s} {t_s*1000:10.3f} ms  (N={N})")
    records.extend(c_records)

    return records


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(records: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path  = os.path.join(out_dir, "bench_results.csv")
    json_path = os.path.join(out_dir, "bench_results.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["impl", "N", "op",
                                                "time_s", "mem_bytes", "result"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nResults saved to:\n  {csv_path}\n  {json_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cython lab benchmark runner")
    parser.add_argument("--sizes",      nargs="+", type=int, default=[64, 128])
    parser.add_argument("--repeats",    type=int,  default=5)
    parser.add_argument("--mc-samples", type=int,  default=1_000_000)
    parser.add_argument("--out",        default="benchmarks/results")
    args = parser.parse_args()

    print("Cython Lab — Benchmark Runner")
    print(f"Matrix sizes : {args.sizes}")
    print(f"Repeats      : {args.repeats}")
    print(f"MC samples   : {args.mc_samples:,}")

    records = run_benchmarks(args.sizes, args.repeats, args.mc_samples)
    save_results(records, args.out)


if __name__ == "__main__":
    main()
