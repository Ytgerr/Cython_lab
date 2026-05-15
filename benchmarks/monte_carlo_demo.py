"""
monte_carlo_demo.py — Visual Monte Carlo pi estimation demo.

Shows two plots:
  1. Scatter plot: random darts inside/outside the unit circle
  2. Convergence: how pi estimate improves as n_samples grows

Run:
    python benchmarks/monte_carlo_demo.py
    python benchmarks/monte_carlo_demo.py --samples 5000 --save
"""

import argparse
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "benchmarks", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate(n: int, seed: int = 42):
    """Return arrays xs, ys, inside_mask for n random points."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1, n)
    ys = rng.uniform(0, 1, n)
    inside = xs**2 + ys**2 < 1.0
    return xs, ys, inside


def convergence(max_n: int = 100_000, steps: int = 200, seed: int = 42):
    """Return (ns, pi_estimates) showing convergence."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1, max_n)
    ys = rng.uniform(0, 1, max_n)
    inside = xs**2 + ys**2 < 1.0

    ns = np.unique(np.logspace(1, np.log10(max_n), steps).astype(int))
    pi_ests = []
    for n in ns:
        pi_ests.append(4.0 * inside[:n].sum() / n)
    return ns, np.array(pi_ests)


# ---------------------------------------------------------------------------
# Plot 1: Dart scatter
# ---------------------------------------------------------------------------

def plot_scatter(n_show: int = 2000, save: bool = True):
    xs, ys, inside = simulate(n_show)
    pi_est = 4.0 * inside.sum() / n_show

    fig, ax = plt.subplots(figsize=(7, 7))

    # Quarter circle arc
    theta = np.linspace(0, np.pi / 2, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2, zorder=5)

    # Points
    ax.scatter(xs[inside],  ys[inside],  s=4, color="#2ecc71", alpha=0.6,
               label=f"Inside circle  ({inside.sum():,})")
    ax.scatter(xs[~inside], ys[~inside], s=4, color="#e74c3c", alpha=0.6,
               label=f"Outside circle ({(~inside).sum():,})")

    # Square boundary
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        f"Monte Carlo pi estimation\n"
        f"n = {n_show:,}   pi ≈ {pi_est:.4f}   (true pi = {np.pi:.4f})",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="lower left")

    # Formula annotation
    ax.text(0.5, 0.08,
            r"$\pi \approx 4 \times \frac{\mathrm{inside}}{n}$"
            f" = 4 × {inside.sum():,} / {n_show:,} = {pi_est:.4f}",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#aaa", alpha=0.9),
            transform=ax.transAxes)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "monte_carlo_scatter.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Convergence
# ---------------------------------------------------------------------------

def plot_convergence(max_n: int = 500_000, save: bool = True):
    ns, pi_ests = convergence(max_n)
    error = np.abs(pi_ests - np.pi)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Monte Carlo pi — Convergence", fontsize=14, fontweight="bold")

    # Top: pi estimate vs true pi
    ax1.semilogx(ns, pi_ests, color="#3498db", linewidth=1.5, label="pi estimate")
    ax1.axhline(np.pi, color="#e74c3c", linestyle="--", linewidth=1.5,
                label=f"True pi = {np.pi:.6f}")
    ax1.fill_between(ns, np.pi - 0.05, np.pi + 0.05,
                     alpha=0.08, color="#e74c3c")
    ax1.set_ylabel("Estimated pi", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_ylim(2.8, 3.5)

    # Bottom: absolute error (log-log)
    ax2.loglog(ns, error, color="#e67e22", linewidth=1.5, label="|error|")
    # Theoretical 1/sqrt(n) convergence
    c = error[0] * np.sqrt(ns[0])
    ax2.loglog(ns, c / np.sqrt(ns), "k--", linewidth=1, alpha=0.6,
               label=r"$O(1/\sqrt{n})$ reference")
    ax2.set_xlabel("Number of samples (log scale)", fontsize=11)
    ax2.set_ylabel("|pi_est - pi|  (log scale)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle="--", which="both")

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "monte_carlo_convergence.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Speedup comparison (Python vs Cython untyped vs Cython typed vs C)
# ---------------------------------------------------------------------------

def plot_speedup(save: bool = True):
    """
    Benchmark all implementations for monte_carlo_pi and plot speedup bars.
    """
    N = 1_000_000
    repeats = 5

    from src.py_impl.matrix_ops import monte_carlo_pi as py_mc

    impls = {"Pure Python": py_mc}

    try:
        from src.cy_untyped.matrix_ops_cy import monte_carlo_pi as cy_u_mc
        impls["Cython\nuntyped"] = cy_u_mc
    except ImportError:
        pass

    try:
        from src.cy_typed.matrix_ops_typed import monte_carlo_pi as cy_t_mc
        impls["Cython\ntyped"] = cy_t_mc
    except ImportError:
        pass

    # C binary via subprocess
    c_exe = os.path.join(ROOT, "src", "c_impl", "matrix_bench.exe")
    c_time = None
    if os.path.exists(c_exe):
        import subprocess, json as _json
        try:
            r = subprocess.run([c_exe], capture_output=True, timeout=60)
            if r.returncode == 0:
                data = _json.loads(r.stdout.decode())
                for rec in data:
                    if rec["op"] == "monte_carlo":
                        c_time = rec["time_s"]
                        break
        except Exception:
            pass

    # Time Python/Cython implementations
    times = {}
    for label, fn in impls.items():
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(N)
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        times[label] = best
        print(f"  {label.replace(chr(10),' '):20s}: {best*1000:.2f} ms")

    if c_time is not None:
        times["C"] = c_time
        print(f"  {'C':20s}: {c_time*1000:.2f} ms")

    baseline = times["Pure Python"]
    labels   = list(times.keys())
    speedups = [baseline / times[l] for l in labels]
    colors   = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"][:len(labels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Monte Carlo pi  (n = {N:,})", fontsize=14, fontweight="bold")

    # Time bars
    t_ms = [times[l] * 1000 for l in labels]
    bars1 = ax1.bar(labels, t_ms, color=colors, edgecolor="white", alpha=0.9)
    ax1.set_ylabel("Time (ms, log scale)")
    ax1.set_yscale("log")
    ax1.set_title("Execution time")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, t in zip(bars1, t_ms):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.15,
                 f"{t:.1f} ms", ha="center", va="bottom", fontsize=9)

    # Speedup bars
    bars2 = ax2.bar(labels, speedups, color=colors, edgecolor="white", alpha=0.9)
    ax2.axhline(1, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.6)
    ax2.set_ylabel("Speedup vs Pure Python")
    ax2.set_title("Speedup")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, s in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{s:.1f}x", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "monte_carlo_speedup.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo pi demo")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Points for scatter plot (default: 2000)")
    parser.add_argument("--max-n",   type=int, default=500_000,
                        help="Max samples for convergence plot (default: 500000)")
    parser.add_argument("--save",    action="store_true", default=True)
    args = parser.parse_args()

    print("=== Monte Carlo pi demo ===")
    print(f"Scatter plot  : {args.samples:,} points")
    print(f"Convergence   : up to {args.max_n:,} samples")
    print()

    print("[1/3] Scatter plot...")
    plot_scatter(n_show=args.samples, save=args.save)

    print("[2/3] Convergence plot...")
    plot_convergence(max_n=args.max_n, save=args.save)

    print("[3/3] Speedup comparison...")
    plot_speedup(save=args.save)

    print()
    print("Done! Charts saved to benchmarks/results/")
    print("  monte_carlo_scatter.png    — dart scatter inside/outside circle")
    print("  monte_carlo_convergence.png — pi estimate converges to true pi")
    print("  monte_carlo_speedup.png    — Python vs Cython vs C speedup")


if __name__ == "__main__":
    main()
