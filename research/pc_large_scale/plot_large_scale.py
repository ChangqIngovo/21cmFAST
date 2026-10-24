"""Scientific diagnostics for the large-volume comparison (no fitted references)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_plots(root, reports, rows, resolution, config):
    root = Path(root)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 130, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight'})
    colors = ['#2166ac', '#b2182b', '#16856b', '#8055a5']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    base = [r for r in rows if r['variant'] == 'baseline']
    for i, grid in enumerate(config['grids']):
        for model, style, label in [('pc', '-', 'PC'), ('es_matched_mass', '--', 'ES')]:
            groups = [[r for r in base if r['grid']==grid and r['dose']==q and r['model']==model]
                      for q in config['photon_doses']]
            q = [np.mean([r['Q_mass'] for r in g]) for g in groups]
            ratio = [np.mean([r['emitted_over_absorbed'] for r in g]) for g in groups]
            bias = [np.mean([r['b_HI_squared_k0'] for r in g]) for g in groups]
            name = f'{label}, {config["box_hMpc"]/grid:g} $h^{{-1}}$ cMpc'
            axes[0, 0].plot(q, ratio, style, color=colors[i], label=name)
            axes[0, 1].plot(q, bias, style, color=colors[i], label=name)
        times = [[r['pc_seconds'] for r in base if r['grid']==grid and r['dose']==q and r['model']=='pc']
                 for q in config['photon_doses']]
        axes[1, 1].plot(config['photon_doses'], [np.median(t) for t in times], 'o-',
                         color=colors[i], label=f'{grid}$^3$')
    axes[0, 0].axhline(1, color='.6', linewidth=.8)
    axes[0, 0].set(xlabel=r'$Q_{\rm HII}^{M}$', ylabel='Emitted / ionized atom count', title='Photon accounting')
    axes[0, 1].set(xlabel=r'$Q_{\rm HII}^{M}$', ylabel=r'$b_{\rm HI}^{2}$',
                   yscale='log', title=r'$0.04 \leq k < 0.06\ h\,\mathrm{cMpc}^{-1}$')
    axes[0, 0].legend(fontsize=8)
    for i, coarse in enumerate(config['grids'][1:]):
        for model, style in [('pc', '-'), ('es_matched_mass', '--')]:
            data = [r for r in resolution if r['coarse_grid']==coarse and r['model']==model
                    and r['mean_ratio'] is not None]
            axes[1, 0].errorbar([r['dose'] for r in data], [r['mean_ratio'] for r in data],
                yerr=[r['sample_std'] or 0 for r in data], color=colors[i], linestyle=style,
                label=f'{model}: {config["native_grid"]}/{coarse}', capsize=2)
    axes[1, 0].axhline(1, color='.6', linewidth=.8)
    axes[1, 0].axhspan(.95, 1.05, color='.9', zorder=-1)
    axes[1, 0].set(xlabel='Target ionized mass fraction', ylabel='Fine / coarse $b_{HI}^{2}$',
                   title='Paired seeds; bars show sample standard deviation')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(xlabel='Target ionized mass fraction', ylabel='Seconds including mass calibration',
                   yscale='log', title='Median over seeds; concurrent case execution')
    axes[1, 1].legend()
    fig.suptitle(f'21cmFAST sources + adaptive PC | z={config["redshift"]:g} | L={config["box_hMpc"]:g} $h^{{-1}}$ cMpc')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'comparison.{ext}')
    plt.close(fig)

    wanted = sorted(set(min(config['photon_doses'], key=lambda x:abs(x-q)) for q in (.2, .5, .8)))
    fig, axes = plt.subplots(2, len(wanted), figsize=(4*len(wanted), 7), squeeze=False)
    cap = np.load(root/'fixtures'/f'seed{config["seeds"][0]}'/f'n{config["native_grid"]}'/'capacity.npy')
    im = None
    for col, dose in enumerate(wanted):
        r = next(r for r in reports if r['task']['variant']=='baseline'
                  and r['task']['seed']==config['seeds'][0] and r['task']['grid']==config['native_grid']
                  and r['task']['dose']==dose)
        with np.load(root/'cases'/r['task']['directory']/'fields.npz') as arrays:
            for row, model in enumerate(('es_matched_mass', 'pc')):
                x = arrays[model+'_absorbed']/cap
                im = axes[row, col].imshow(x[:, :, x.shape[2]//2].T, origin='lower', vmin=0, vmax=1,
                    cmap='magma', extent=(0, config['box_hMpc'], 0, config['box_hMpc']))
                axes[row, col].set_title(f'{model}, Qmass={r["spectra"][model]["Q_mass"]:.4f}')
                axes[row, col].set_xlabel(r'$h^{-1}$ cMpc')
    fig.subplots_adjust(right=.88, hspace=.35, wspace=.3)
    fig.colorbar(im, cax=fig.add_axes([.91,.2,.02,.6]), label=r'$x_{\rm HII}$')
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'morphology.{ext}')
    plt.close(fig)

    fig, axes = plt.subplots(1, len(wanted), figsize=(4.3*len(wanted), 4), squeeze=False)
    for col, dose in enumerate(wanted):
        ax = axes[0, col]
        for i, n in enumerate(config['grids']):
            rr = [r for r in reports if r['task']['variant']=='baseline' and r['task']['grid']==n
                  and r['task']['dose']==dose]
            for model, style in [('pc', '-'), ('es_matched_mass', '--')]:
                first = rr[0]['spectra'][model]['shells']
                kvals = [s['k_hMpc_inv'] for s in first]
                mean = np.mean([[s['b_HI_squared'] for s in r['spectra'][model]['shells']] for r in rr], axis=0)
                ax.loglog(kvals, mean, style, color=colors[i], label=f'{model}, {n}$^3$')
        ax.set(xlabel=r'$k\ [h\,\mathrm{cMpc}^{-1}]$', ylabel=r'$b_{\rm HI}^{2}(k)$',
               title=f'Target ionized mass fraction = {dose:g}')
        ax.axvspan(*config['k0_band_hMpc_inv'], color='.9', zorder=-1)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'bias_spectra.{ext}')
    plt.close(fig)
