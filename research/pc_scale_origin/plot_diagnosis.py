"""Plot measured production spectra and the independent front-size intervention."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    large = json.loads((HERE/'large_spectral_results.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), constrained_layout=True)
    ax = axes[0]
    for item, color in zip([r for r in large['comparisons'] if r['dose']==.9], ['#2467a5', '#c15132']):
        rows = [r for r in item['shared_shells'] if r['k_hMpc_inv'] < .12]
        k = np.array([r['k_hMpc_inv'] for r in rows])
        y = np.array([r['fine_over_coarse_b2_minus_one']['mean'] for r in rows])*100
        std = np.array([r['fine_over_coarse_b2_minus_one']['seed_sample_std'] for r in rows])*100
        ax.plot(k, y, 'o-', ms=3.5, color=color, label=f"5 / {500/item['coarse_grid']:g} $h^{{-1}}$ cMpc")
        ax.fill_between(k, y-std, y+std, color=color, alpha=.15)
    ax.axhline(0, color='.6', lw=.7)
    ax.set(xscale='log', xlabel=r'$k\ [h\,\mathrm{cMpc}^{-1}]$',
           ylabel=r'Fine / coarse $b_{\mathrm{HI}}^2-1$ [%]',
           title=r'500 $h^{-1}$ cMpc box: $Q=0.9$')
    ax.set_xticks([.015, .03, .06, .1], ['0.015', '0.03', '0.06', '0.1'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.15)
    ax = axes[1]
    rows = {}
    for name in ('factorial_results.json', 'factorial_high_angles.json',
                 'factorial_front_angular_check.json', 'factorial_front_final_check.json'):
        data = json.loads((HERE/name).read_text())
        for c in data['contrasts']:
            if c['dose']==.9 and c['effect']=='front_resolution_at_fixed_coarse_sources':
                rows[c['angles']] = (c['b2_ratio']-1)*100
    k = sorted(rows)
    ax.plot(k, [rows[v] for v in k], 'o-', color='#2467a5')
    ax.axhline(0, color='.6', lw=.7)
    ax.set_xscale('log', base=2)
    ticks = [96, 384, 1536, 6144, 12288]
    ax.set_xticks(ticks, [str(t) for t in ticks], rotation=30)
    ax.set(xlabel='Directions per source', ylabel=r'Coarse / fine front $b_{\mathrm{HI}}^2-1$ [%]',
           title='Small fixture: fixed sources and density', ylim=(0, 16))
    ax.text(.04, .08, r'Gas grid $16^3$ versus $32^3$; $Q=0.9$'+'\n'+r'Both measured on the same $8^3$ grid',
            transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=.15)
    fig.savefig(HERE/'scale_origin.png', dpi=180)
    fig.savefig(HERE/'scale_origin.pdf')


if __name__ == '__main__':
    main()
