"""Plot the independent sphere accuracy and fixed-source native convergence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    sphere = json.loads((HERE/'sphere_results.json').read_text())
    summary = json.loads((HERE/'benchmark_summary.json').read_text())
    native16 = json.loads((HERE/'native_pair16.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), constrained_layout=True)
    for case, color in zip(sphere['cases'], ['#7ca6c8', '#245b8a']):
        rows = sorted((r for r in case['models'].values() if r['model'] == 'refined'), key=lambda r: r['factor'])
        axes[0].plot([r['factor'] for r in rows],
                     [100*r['vs_integrated_continuum_sphere']['L1_over_emitted'] for r in rows],
                     'o-', color=color, label=f"K = {case['angles']:,}")
    for pair, color in zip(summary['original_grid_coarsening'], ['#aaaaaa', '#7ca6c8', '#245b8a']):
        axes[1].plot([2, 4], [100*pair['factors'][str(f)]['source16_over_source8_minus_one'] for f in (2, 4)],
                     'o-', color=color, label=f"K = {pair['angles']:,}")
    axes[1].axhline(0, color='.6', lw=.7)
    case = native16['cases'][-1]
    selected = [case['models']['adaptive_f2'], case['models']['refined_f4']]
    baseline = selected[0]['median_seconds']
    x, width = np.arange(2), .34
    axes[2].bar(x-width/2, [r['first_solve_seconds']/baseline for r in selected], width,
                color='#a8c1d4', label='First solve (cold templates)')
    axes[2].bar(x+width/2, [r['median_seconds']/baseline for r in selected], width,
                color='#245b8a', label='Warmed median')
    axes[2].set_xticks(x, ['Existing 8 cells', 'Refined 64 cells'])
    axes[2].set(title=f"Solver cost: source $16^3$, K={case['angles']:,}",
                ylabel='Time / existing warmed solver')
    axes[0].set(title='Uniform isolated sphere', ylabel='Integrated-cell L1 / emitted photons [%]', ylim=(0, 6.4))
    axes[1].set(title=r'Original-grid dependence; $Q=0.9$',
                ylabel=r'$b_{\rm HI}^2(16^3)/b_{\rm HI}^2(8^3)-1$ [%]', ylim=(-14, 1))
    axes[1].text(.04, .06, r'Both measured on the same $8^3$ grid',
                 transform=axes[1].transAxes, fontsize=8)
    for ax in axes[:2]:
        ax.set_xscale('log', base=2)
        ticks = [2, 4, 8] if ax is axes[0] else [2, 4]
        ax.set_xticks(ticks, [f'{f} ({f**3} cells)' for f in ticks])
        ax.set_xlabel('Subdivisions per original cell axis')
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(HERE/'refinement_results.png', dpi=180)
    fig.savefig(HERE/'refinement_results.pdf')


if __name__ == '__main__':
    main()
