#!/usr/bin/env python3
"""Plot measured angular accuracy, cost and a separate continuum-front test.

All plotted values are read from recorded result JSONs. No solver is rerun.
The two native fixtures use different source constructions and are not treated
as a spatial-resolution convergence sequence. The sphere panel has an analytic
continuum reference; the native panels have angularly refined voxel references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
COLORS = ['#666666', '#2878B5', '#009E73']
PROFILES = ['original', 'compact', 'boost-1.25']
LABELS = ['Original', 'Compact: same rays', 'Compact: +25% angles']


def read_values(results_dir, previous_dir):
    budget = json.loads((results_dir/'budgeted_results.json').read_text())
    selected = [next(c for c in budget['cases'] if c['name'] == name)
                for name in ('native-frozen', 'native64-L')]
    values = []
    for case in selected:
        rows = [next(r for r in case['profiles'] if r['profile'] == p) for p in PROFILES]
        baseline_seconds = float(np.median(rows[0]['times']))
        values.append({
            'case':case['name'],
            'angular_L1_percent':[100*r['vs_reference']['L1_absorption_over_emitted'] for r in rows],
            'median_seconds':[float(np.median(r['times'])) for r in rows],
            'runtime_ratio':[float(np.median(r['times']))/baseline_seconds for r in rows],
            'ray_memory_MiB':[r['audit']['ray_allocated_bytes']/2**20 for r in rows],
        })
    native32 = json.loads((previous_dir/'comparison_native_refined_results.json').read_text())
    c32 = next(c for c in native32['cases'] if c['name']=='native-frozen')
    native64 = json.loads((previous_dir/'native64_reference_results.json').read_text())
    c64 = next(r for r in native64['references'] if r['angles']==768)
    reference_changes = [100*c32['reference_angular_change']['L1_absorption_over_emitted'],
                         100*c64['previous_reference_vs_current']['L1_absorption_over_emitted']]
    front = json.loads((results_dir/'front_results.json').read_text())
    sphere = next(c for c in front['cases'] if c['name']=='isolated')
    keys = ['original-fibonacci', 'balanced-coarse', 'balanced-subcells']
    front_values = {
        'labels':keys,
        'continuum_L1_percent':[100*sphere['models'][k]['vs_analytic']['L1_over_Q'] for k in keys],
        'median_seconds':[sphere['models'][k]['median_seconds'] for k in keys],
        'ray_counts':[sphere['models'][k]['audit']['rays_materialized'] for k in keys],
        'radius_cells':sphere['metadata']['radius_cells'],
        'native_runtime_ratio':front['native']['models']['balanced-subcells']['median_seconds'] /
                               front['native']['models']['original-fibonacci']['median_seconds'],
    }
    if len(set(front_values['ray_counts'])) != 1:
        raise ValueError('Sphere angular counts changed; update the comparison annotation')
    return {'native':values, 'reference_angular_change_percent':reference_changes, 'sphere':front_values}


def grouped_bars(ax, values, key, ylabel, title, fmt):
    x = np.arange(2)
    width = .22
    for i in range(3):
        heights = [row[key][i] for row in values]
        bars = ax.bar(x+(i-1)*width, heights, width=width, color=COLORS[i], zorder=3)
        ax.bar_label(bars, labels=[fmt.format(v) for v in heights], padding=5, fontsize=11)
    ax.set_xticks(x, [r'E-INTEGRAL $32^3$', r'L-INTEGRAL $64^3$'])
    ax.set_xlim(-.55, 1.55)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=13)
    ax.grid(axis='y', alpha=.2, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    return x


def plot(results_dir=HERE, previous_dir=HERE.parent/'pc_reconsidered', output=HERE/'accuracy_cost.png', pdf=None):
    values = read_values(Path(results_dir), Path(previous_dir))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
                         'axes.labelsize':12, 'xtick.labelsize':11, 'ytick.labelsize':11,
                         'savefig.facecolor':'white'})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.8))
    fig.subplots_adjust(left=.085, right=.98, top=.83, bottom=.105, hspace=.58, wspace=.30)
    fig.suptitle('Accuracy and cost of photon-conservation updates', fontsize=19, y=.975)
    fig.text(.5, .932, 'Native panels: initially neutral, frozen source fields; same photon budget',
             ha='center', fontsize=12, color='#444444')
    fig.legend(handles=[Patch(facecolor=c, label=l) for c,l in zip(COLORS,LABELS)],
               loc='upper center', bbox_to_anchor=(.5,.91), frameon=False, ncol=3, fontsize=12)
    ax = axes[0,0]
    grouped_bars(ax, values['native'], 'angular_L1_percent', r'Absorption $L_1/Q$ (%)',
                 'A  Difference from high-angle voxel reference', '{:.3f}')
    ax.set_ylim(0, .72)
    for x, error in enumerate(values['reference_angular_change_percent']):
        ax.hlines(error, x-.36, x+.36, color='#333333', ls='--', lw=1.2)
        ax.text(x, error+.016, f'Reference change: {error:.3f}%', ha='center', va='bottom',
                fontsize=9, color='#333333', bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1})
    ax = axes[0,1]
    grouped_bars(ax, values['native'], 'runtime_ratio', 'Median time / original',
                 'B  Runtime at the same source field', '{:.2f}×')
    ax.axhline(1, color='#333333', ls='--', lw=1.2)
    ax.set_ylim(0, 1.62)
    ax.text(.98,.96,'Sequential repeats; shared host; no compilation', transform=ax.transAxes,
            ha='right',va='top',fontsize=9,color='#555555')
    ax = axes[1,0]
    grouped_bars(ax, values['native'], 'ray_memory_MiB', 'Allocated ray memory (MiB)',
                 'C  Ray memory at the measured accuracy', '{:.1f}')
    ax.set_ylim(0, 91)
    ax.text(.02,.96,'Ray storage only; not whole-process peak memory',transform=ax.transAxes,
            ha='left',va='top',fontsize=9,color='#555555')
    ax = axes[1,1]
    sphere = values['sphere']
    bars = ax.bar(np.arange(3),sphere['continuum_L1_percent'],color=['#666666','#D55E00','#CC79A7'],width=.6,zorder=3)
    ax.bar_label(bars,labels=[f'{v:.2f}%' for v in sphere['continuum_L1_percent']],padding=5,fontsize=11)
    ax.set_xticks(np.arange(3),['Original\ncoarse cells','Matched rule\ncoarse cells','Matched rule\neight subcells'])
    ax.set_ylabel(r'Continuum sphere $L_1/Q$ (%)')
    ax.set_title('D  Spatial error: separate analytic sphere test',loc='left',fontsize=13,fontweight='bold',pad=13)
    ax.set_ylim(0,16.5)
    ax.grid(axis='y',alpha=.2,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.text(.5,.97,f"Radius {sphere['radius_cells']:g} cells; {sphere['ray_counts'][0]} rays per source",transform=ax.transAxes,
            ha='center',va='top',fontsize=9,color='#555555')
    ax.text(.5,.89,f"Subcell option: {sphere['native_runtime_ratio']:.2f}× native32 runtime",
            transform=ax.transAxes,ha='center',va='top',fontsize=9,color='#555555')
    fig.text(.5,.025,'Native angular errors and continuum spatial errors have different references.\n'
             'The two native fixtures are not a spatial-resolution convergence sequence.',ha='center',fontsize=10,color='#555555')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180)
    if pdf:
        fig.savefig(pdf)
    plt.close(fig)
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=HERE)
    parser.add_argument('--previous-dir',type=Path,default=HERE.parent/'pc_reconsidered')
    parser.add_argument('--output',type=Path,default=HERE/'accuracy_cost.png')
    parser.add_argument('--pdf',type=Path)
    values = plot(**vars(parser.parse_args()))
    print(json.dumps(values,indent=2))
