"""
make_report.py — Generate charts + HTML report from bench_results.json

Usage:
    python benchmarks/make_report.py

Output:
    benchmarks/results/report.html
    benchmarks/results/chart_time.png
    benchmarks/results/chart_speedup.png
    benchmarks/results/chart_memory.png
    benchmarks/results/chart_overview.png
    benchmarks/results/chart_monte_carlo.png
"""
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), 'results')
JSON_PATH     = os.path.join(RESULTS_DIR, 'bench_results.json')
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'report_template.html')

IMPL_LABELS = {
    'python':     'Pure Python',
    'cy_untyped': 'Cython untyped',
    'cy_typed':   'Cython typed',
    'c':          'C',
}
IMPL_ORDER  = ['python', 'cy_untyped', 'cy_typed', 'c']
IMPL_COLORS = {
    'python':     '#e74c3c',
    'cy_untyped': '#e67e22',
    'cy_typed':   '#2ecc71',
    'c':          '#3498db',
}

MATRIX_OPS = ['matmul', 'matrix_add']
MC_OP      = 'monte_carlo'

OP_LABELS = {
    'matmul':      'matmul  O(n^3)',
    'matrix_add':  'matrix_add  O(n^2)',
    'monte_carlo': 'Dice Game  (roll 2 dice, win if sum ≥ 7)',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(JSON_PATH, encoding='utf-8') as f:
    records = json.load(f)

matrix_records = [r for r in records if r['op'] in MATRIX_OPS]
mc_records     = [r for r in records if r['op'] == MC_OP]

matrix_sizes = sorted(set(r['N'] for r in matrix_records))
mc_sizes     = sorted(set(r['N'] for r in mc_records))


def get(impl, N, op, field='time_s'):
    for r in records:
        if r['impl'] == impl and r['N'] == N and r['op'] == op:
            return r[field]
    return None


def ms(v):
    return v * 1000 if v is not None else None


def speedup(base, val):
    return base / val if val and val > 0 else None


# ---------------------------------------------------------------------------
# Chart 1: Time per matrix operation
# ---------------------------------------------------------------------------
def chart_time():
    ops   = MATRIX_OPS
    sizes = matrix_sizes
    fig, axes = plt.subplots(1, len(ops), figsize=(14, 5), sharey=False)
    if len(ops) == 1:
        axes = [axes]
    fig.suptitle('Execution Time - Matrix Operations', fontsize=15, fontweight='bold', y=1.02)

    for ax, op in zip(axes, ops):
        x       = np.arange(len(sizes))
        n_impl  = len(IMPL_ORDER)
        width   = 0.18
        offsets = np.linspace(-(n_impl - 1) / 2 * width, (n_impl - 1) / 2 * width, n_impl)

        for k, impl in enumerate(IMPL_ORDER):
            times = [ms(get(impl, N, op)) for N in sizes]
            valid = [t if t is not None else 0 for t in times]
            bars  = ax.bar(x + offsets[k], valid, width,
                           label=IMPL_LABELS[impl],
                           color=IMPL_COLORS[impl], edgecolor='white', linewidth=0.5, alpha=0.9)
            for bar, t in zip(bars, times):
                if t is not None and t > 0:
                    h   = bar.get_height()
                    lbl = f'{t:.3f}' if t < 1 else f'{t:.1f}'
                    ax.text(bar.get_x() + bar.get_width() / 2, h * 1.05,
                            lbl, ha='center', va='bottom', fontsize=7.5, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{N}x{N}' for N in sizes])
        ax.set_title(OP_LABELS.get(op, op), fontweight='bold')
        ax.set_ylabel('Time (ms, log scale)')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlabel('Matrix size')

    handles = [mpatches.Patch(color=IMPL_COLORS[i], label=IMPL_LABELS[i]) for i in IMPL_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0),
               frameon=True, fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'chart_time.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Chart 2: Speedup vs Pure Python (matrix ops)
# ---------------------------------------------------------------------------
def chart_speedup():
    ops   = MATRIX_OPS
    sizes = matrix_sizes
    fig, axes = plt.subplots(1, len(ops), figsize=(14, 5), sharey=False)
    if len(ops) == 1:
        axes = [axes]
    fig.suptitle('Speedup vs Pure Python - Matrix Operations', fontsize=15, fontweight='bold', y=1.02)

    impls_to_plot = ['cy_untyped', 'cy_typed', 'c']
    n_imp   = len(impls_to_plot)
    width   = 0.22
    offsets = np.linspace(-(n_imp - 1) / 2 * width, (n_imp - 1) / 2 * width, n_imp)

    for ax, op in zip(axes, ops):
        x = np.arange(len(sizes))
        for k, impl in enumerate(impls_to_plot):
            speedups = []
            for N in sizes:
                base = get('python', N, op)
                val  = get(impl, N, op)
                speedups.append(speedup(base, val))

            valid = [s if s is not None else 0 for s in speedups]
            bars  = ax.bar(x + offsets[k], valid, width,
                           label=IMPL_LABELS[impl],
                           color=IMPL_COLORS[impl], edgecolor='white', linewidth=0.5, alpha=0.9)
            for bar, s in zip(bars, speedups):
                if s is not None and s > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f'{s:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.axhline(1, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, label='Python (1x)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{N}x{N}' for N in sizes])
        ax.set_title(OP_LABELS.get(op, op), fontweight='bold')
        ax.set_ylabel('Speedup (x times)')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlabel('Matrix size')
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'chart_speedup.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Chart 3: Memory usage (Python/Cython only)
# ---------------------------------------------------------------------------
def chart_memory():
    ops   = MATRIX_OPS
    sizes = matrix_sizes
    fig, axes = plt.subplots(1, len(ops), figsize=(14, 5))
    if len(ops) == 1:
        axes = [axes]
    fig.suptitle('Peak Memory Usage - Matrix Operations', fontsize=15, fontweight='bold', y=1.02)

    impls_mem = ['python', 'cy_untyped', 'cy_typed']
    n_impl  = len(impls_mem)
    width   = 0.22
    offsets = np.linspace(-(n_impl - 1) / 2 * width, (n_impl - 1) / 2 * width, n_impl)

    for ax, op in zip(axes, ops):
        x = np.arange(len(sizes))
        for k, impl in enumerate(impls_mem):
            mems    = [get(impl, N, op, 'mem_bytes') for N in sizes]
            mems_kb = [m / 1024 if m is not None else 0 for m in mems]
            bars    = ax.bar(x + offsets[k], mems_kb, width,
                             label=IMPL_LABELS[impl],
                             color=IMPL_COLORS[impl], edgecolor='white', linewidth=0.5, alpha=0.9)
            for bar, m in zip(bars, mems_kb):
                if m > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f'{m:.0f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{N}x{N}' for N in sizes])
        ax.set_title(OP_LABELS.get(op, op), fontweight='bold')
        ax.set_ylabel('Peak memory (KB)')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlabel('Matrix size')

    handles = [mpatches.Patch(color=IMPL_COLORS[i], label=IMPL_LABELS[i]) for i in impls_mem]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0),
               frameon=True, fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'chart_memory.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Chart 4: Monte Carlo — time + speedup
# ---------------------------------------------------------------------------
def chart_monte_carlo():
    if not mc_records:
        print('[WARN] No monte_carlo records - skipping chart_monte_carlo')
        return None

    N = mc_sizes[0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Monte Carlo Dice Game  (n = {N:,} rounds, win if sum ≥ 7)  |  True probability = 21/36 ≈ 0.5833',
                 fontsize=14, fontweight='bold', y=1.02)

    impls_present = [i for i in IMPL_ORDER if get(i, N, MC_OP) is not None]
    x     = np.arange(len(impls_present))
    width = 0.5

    # --- Chart 1: Execution time ---
    ax1 = axes[0]
    times  = [ms(get(impl, N, MC_OP)) for impl in impls_present]
    colors = [IMPL_COLORS[i] for i in impls_present]
    bars   = ax1.bar(x, times, width, color=colors, edgecolor='white', linewidth=0.5, alpha=0.9)
    for bar, t in zip(bars, times):
        if t is not None:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
                     f'{t:.1f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([IMPL_LABELS[i] for i in impls_present], fontsize=10)
    ax1.set_ylabel('Time (ms, log scale)')
    ax1.set_yscale('log')
    ax1.set_title('Execution Time', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # --- Chart 2: Speedup ---
    ax2 = axes[1]
    py_t     = get('python', N, MC_OP)
    sp_impls = [i for i in impls_present if i != 'python']
    speedups = [speedup(py_t, get(impl, N, MC_OP)) for impl in sp_impls]

    x2      = np.arange(len(sp_impls))
    colors2 = [IMPL_COLORS[i] for i in sp_impls]
    bars2   = ax2.bar(x2, speedups, width, color=colors2, edgecolor='white', linewidth=0.5, alpha=0.9)
    for bar, s in zip(bars2, speedups):
        if s is not None:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{s:.1f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.axhline(1, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, label='Python (1x)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([IMPL_LABELS[i] for i in sp_impls], fontsize=10)
    ax2.set_ylabel('Speedup vs Pure Python')
    ax2.set_title('Speedup vs Pure Python', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)

    # --- Chart 3: Win probability result (the actual experiment result) ---
    ax3 = axes[2]
    results = []
    for impl in impls_present:
        r = next((rec.get('result') for rec in records
                  if rec['impl'] == impl and rec['N'] == N and rec['op'] == MC_OP), None)
        results.append(r)

    TRUE_PROB = 21 / 36  # 0.5833...
    colors3 = [IMPL_COLORS[i] for i in impls_present]
    bars3 = ax3.bar(x, [r if r is not None else 0 for r in results],
                    width, color=colors3, edgecolor='white', linewidth=0.5, alpha=0.9)
    for bar, r in zip(bars3, results):
        if r is not None:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f'{r:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.axhline(TRUE_PROB, color='#8e44ad', linestyle='--', linewidth=2,
                label=f'Теория = {TRUE_PROB:.4f}')
    ax3.set_xticks(x)
    ax3.set_xticklabels([IMPL_LABELS[i] for i in impls_present], fontsize=10)
    ax3.set_ylabel('Вероятность выигрыша')
    ax3.set_title('Результат эксперимента\n(все должны дать ~0.5833)', fontweight='bold')
    ax3.set_ylim(0.55, 0.62)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'chart_monte_carlo.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Chart 5: Overview (2x2 grid)
# ---------------------------------------------------------------------------
def chart_overview():
    sizes = matrix_sizes
    mc_N  = mc_sizes[0] if mc_sizes else None

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    for impl in IMPL_ORDER:
        times = [ms(get(impl, N, 'matmul')) for N in sizes]
        pairs = [(N, t) for N, t in zip(sizes, times) if t is not None]
        if pairs:
            ns, ts = zip(*pairs)
            ax1.plot(ns, ts, 'o-', color=IMPL_COLORS[impl],
                     label=IMPL_LABELS[impl], linewidth=2, markersize=8)
            for N, t in pairs:
                ax1.annotate(f'{t:.1f}ms', (N, t), textcoords='offset points',
                             xytext=(5, 5), fontsize=8)
    ax1.set_yscale('log')
    ax1.set_title('matmul - Time (log scale)', fontweight='bold')
    ax1.set_xlabel('Matrix size N')
    ax1.set_ylabel('Time (ms)')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([f'{N}x{N}' for N in sizes])

    ax2 = fig.add_subplot(gs[0, 1])
    for impl in ['cy_untyped', 'cy_typed', 'c']:
        speedups = [speedup(get('python', N, 'matmul'), get(impl, N, 'matmul')) for N in sizes]
        pairs    = [(N, s) for N, s in zip(sizes, speedups) if s is not None]
        if pairs:
            ns, ss = zip(*pairs)
            ax2.plot(ns, ss, 'o-', color=IMPL_COLORS[impl],
                     label=IMPL_LABELS[impl], linewidth=2, markersize=8)
            for N, s in pairs:
                ax2.annotate(f'{s:.0f}x', (N, s), textcoords='offset points',
                             xytext=(5, 5), fontsize=9, fontweight='bold')
    ax2.axhline(1, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, label='Python (1x)')
    ax2.set_title('matmul - Speedup vs Pure Python', fontweight='bold')
    ax2.set_xlabel('Matrix size N')
    ax2.set_ylabel('Speedup (x times)')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([f'{N}x{N}' for N in sizes])

    ax3 = fig.add_subplot(gs[1, 0])
    N_ref   = max(sizes)
    all_ops = MATRIX_OPS + ([MC_OP] if mc_records else [])
    op_speedups = []
    op_names    = []
    for op in all_ops:
        ref_N = mc_N if op == MC_OP else N_ref
        base  = get('python', ref_N, op)
        val   = get('cy_typed', ref_N, op)
        s     = speedup(base, val)
        if s:
            op_speedups.append(s)
            op_names.append(OP_LABELS.get(op, op))

    bars = ax3.barh(op_names, op_speedups, color='#2ecc71', edgecolor='white', alpha=0.9)
    for bar, s in zip(bars, op_speedups):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f'{s:.0f}x', va='center', fontsize=10, fontweight='bold')
    ax3.axvline(1, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_title('Cython typed speedup (all ops)', fontweight='bold')
    ax3.set_xlabel('Speedup vs Pure Python')
    ax3.grid(axis='x', alpha=0.3, linestyle='--')

    ax4 = fig.add_subplot(gs[1, 1])
    if mc_records and mc_N:
        impls_mc  = [i for i in IMPL_ORDER if get(i, mc_N, MC_OP) is not None]
        mc_times  = [ms(get(impl, mc_N, MC_OP)) for impl in impls_mc]
        colors_mc = [IMPL_COLORS[i] for i in impls_mc]
        bars4 = ax4.bar(range(len(impls_mc)), mc_times, color=colors_mc,
                        edgecolor='white', linewidth=0.5, alpha=0.9)
        for bar, t in zip(bars4, mc_times):
            if t is not None:
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
                         f'{t:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax4.set_xticks(range(len(impls_mc)))
        ax4.set_xticklabels([IMPL_LABELS[i] for i in impls_mc], fontsize=9)
        ax4.set_yscale('log')
        ax4.set_title(f'Dice Game  (n={mc_N:,} rounds)', fontweight='bold')
        ax4.set_ylabel('Time (ms, log scale)')
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
    else:
        ax4.text(0.5, 0.5, 'No Monte Carlo data', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=12, color='#aaa')
        ax4.set_title('Dice Game', fontweight='bold')

    fig.suptitle('Cython Lab - Benchmark Overview', fontsize=16, fontweight='bold', y=1.01)
    path = os.path.join(RESULTS_DIR, 'chart_overview.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# HTML Report (uses report_template.html)
# ---------------------------------------------------------------------------
def make_html(chart_paths):
    sizes   = matrix_sizes
    mc_N    = mc_sizes[0] if mc_sizes else None
    all_ops = MATRIX_OPS + ([MC_OP] if mc_records else [])

    # Build table rows
    def fmt_ms(v):
        if v is None:  return '-'
        if v < 0.01:   return f'{v * 1000:.2f} us'
        if v < 1:      return f'{v:.3f} ms'
        return f'{v:.1f} ms'

    def fmt_sp(v, color='#27ae60'):
        if v is None: return '-'
        return f'<b style="color:{color}">{v:.1f}x</b>'

    def size_label(op, N):
        return f'n={N:,}' if op == MC_OP else f'{N}x{N}'

    table_rows_html = ''
    for op in all_ops:
        op_sizes = [mc_N] if op == MC_OP else sizes
        for N in op_sizes:
            py_t   = get('python',     N, op)
            cy_u   = get('cy_untyped', N, op)
            cy_t   = get('cy_typed',   N, op)
            c_t    = get('c',          N, op)
            py_m   = get('python',     N, op, 'mem_bytes')
            cy_t_m = get('cy_typed',   N, op, 'mem_bytes')
            sp_u   = speedup(py_t, cy_u)
            sp_t   = speedup(py_t, cy_t)
            sp_c   = speedup(py_t, c_t)
            py_kb  = py_m / 1024 if py_m else 0
            cyt_kb = cy_t_m / 1024 if cy_t_m else 0
            mr     = '%.1fx' % (py_m / cy_t_m) if cy_t_m and cy_t_m > 0 else '-'
            table_rows_html += (
                '<tr>'
                f'<td>{size_label(op, N)}</td>'
                f'<td>{OP_LABELS.get(op, op)}</td>'
                f'<td>{fmt_ms(ms(py_t))}</td>'
                f'<td>{fmt_ms(ms(cy_u))}</td>'
                f'<td>{fmt_ms(ms(cy_t))}</td>'
                f'<td>{fmt_ms(ms(c_t))}</td>'
                f'<td>{fmt_sp(sp_u, "#e67e22")}</td>'
                f'<td>{fmt_sp(sp_t)}</td>'
                f'<td>{fmt_sp(sp_c, "#3498db")}</td>'
                f'<td>{py_kb:.0f} KB</td>'
                f'<td>{cyt_kb:.0f} KB</td>'
                f'<td>{mr}</td>'
                '</tr>\n'
            )

    # Build charts HTML
    def img_tag(path, title):
        if not path:
            return ''
        rel = os.path.basename(path)
        return (
            f'<figure>'
            f'<img src="{rel}" alt="{title}" style="max-width:100%;border-radius:8px;'
            f'box-shadow:0 2px 12px rgba(0,0,0,0.15)">'
            f'<figcaption>{title}</figcaption>'
            f'</figure>\n'
        )

    charts_html = (
        img_tag(chart_paths.get('overview'),    'Overview: matmul time, speedup, per-op speedup, Monte Carlo') +
        img_tag(chart_paths.get('monte_carlo'), 'Monte Carlo pi - execution time and speedup') +
        img_tag(chart_paths.get('time'),        'Execution time by operation and implementation (log scale)') +
        img_tag(chart_paths.get('speedup'),     'Speedup vs Pure Python by operation') +
        img_tag(chart_paths.get('memory'),      'Peak memory usage by operation')
    )

    # Compute headline speedup numbers
    def best_speedup(op, impl):
        best = None
        ref_sizes = [mc_N] if op == MC_OP else sizes
        for N in (ref_sizes or []):
            base = get('python', N, op)
            val  = get(impl, N, op)
            s    = speedup(base, val)
            if s and (best is None or s > best):
                best = s
        return best or 0

    sp_untyped_mm = best_speedup('matmul', 'cy_untyped')
    sp_typed_mm   = best_speedup('matmul', 'cy_typed')
    sp_typed_mc   = best_speedup(MC_OP,    'cy_typed')
    sp_c_mc       = best_speedup(MC_OP,    'c')

    # Load template and substitute placeholders
    with open(TEMPLATE_PATH, encoding='utf-8') as f:
        template = f.read()

    html = (template
            .replace('{{TABLE_ROWS}}',    table_rows_html)
            .replace('{{CHARTS}}',        charts_html)
            .replace('{{SP_UNTYPED_MM}}', f'~{sp_untyped_mm:.0f}x')
            .replace('{{SP_TYPED_MM}}',   f'~{sp_typed_mm:.0f}x')
            .replace('{{SP_TYPED_MC}}',   f'~{sp_typed_mc:.0f}x')
            .replace('{{SP_C_MC}}',       f'~{sp_c_mc:.0f}x'))

    path = os.path.join(RESULTS_DIR, 'report.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('Generating charts...')
    p_time        = chart_time()
    p_speedup     = chart_speedup()
    p_memory      = chart_memory()
    p_monte_carlo = chart_monte_carlo()
    p_overview    = chart_overview()

    print('Generating HTML report...')
    p_html = make_html({
        'time':        p_time,
        'speedup':     p_speedup,
        'memory':      p_memory,
        'monte_carlo': p_monte_carlo,
        'overview':    p_overview,
    })

    print()
    print('Done! Open the report:')
    print(f'  {p_html}')
