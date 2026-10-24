"""Matched-source ES and temporal diagnostics for prescribed native source histories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
for name in ('pc_large_scale', 'pc_refined_front', 'pc_reconsidered'):
    sys.path.insert(0, str(HERE.parent / name))
from pc_statistics import es_threshold, es_map, spectrum, total
from fine_state_metrics import fine_state_metrics
from source_budget import integrate_rate_step


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _metadata(history):
    metadata = history.get('metadata', {})
    if not isinstance(metadata, dict):
        metadata = json.loads(str(np.asarray(metadata).item()))
    return metadata


def spectral_setup(history):
    """Explicit box-relative band, usable even for tiny integration fixtures."""
    metadata = _metadata(history)
    box = metadata.get('box_hMpc')
    if box is None:
        box = metadata.get('box_len_cMpc', metadata.get('box_len', 200.)) * metadata.get('hlittle', .678)
    box = float(box)
    n = history['capacity'].shape[0]
    if not np.isfinite(box) or box <= 0 or n < 4:
        raise ValueError('Spectra require a positive box length and grid >= 4')
    fundamental, nyquist = 2*np.pi/box, np.pi*n/box
    edges = np.geomspace(fundamental*(1-1e-10), nyquist*(1+1e-10), 25)
    band = (fundamental*(1-1e-10), min(4*fundamental, nyquist)*(1+1e-10))
    return box, edges, band


def checkpoint_fields(row):
    if 'fields' in row:
        return row['fields']
    path = Path(row['path'])
    expected = row.get('sha256')
    if expected:
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise RuntimeError(f'Checkpoint changed: {path}')
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def field_spectrum(history, absorbed, redshift):
    capacity = history['capacity']
    neutral = 1-total(absorbed)/total(capacity)
    if neutral <= 1e-8:
        return None
    box, edges, band = spectral_setup(history)
    result = spectrum(capacity, absorbed, box, edges, band, redshift)
    result['band_edges_h_cMpc_inv'] = list(band)
    result['band_definition'] = 'Fundamental mode through min(4*fundamental, Nyquist); not a fixed physical k across boxes'
    return result


def enrich_history(report, history):
    for row in report['checkpoints']:
        fields = checkpoint_fields(row)
        row['spectra'] = field_spectrum(history, fields['ionized'], row['redshift'])
    return report


def run_controls(history, output):
    """ES receives exactly the accumulated rate-integrated photons used by PC.

    Native ES is contextual only: it uses its own source proxy and evolving
    density. The matched ES uses the same fixed capacity and cumulative counts
    as PC, with the native sharp-k centre-cell barrier and no calibration.
    """
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'checkpoints').mkdir()
    capacity = history['capacity']
    rates, times, redshifts = (history[name] for name in ('rates', 'times_seconds', 'redshifts'))
    cumulative = np.zeros_like(capacity, dtype=np.longdouble)
    checkpoints = []
    native = []
    for index in range(len(times)):
        if index:
            cumulative += integrate_rate_step(rates[index-1], rates[index], times[index]-times[index-1])
        if index % 4 and index != len(times)-1:
            continue
        source = np.asarray(cumulative, dtype=float)
        threshold, filtering = es_threshold(source, capacity, len(capacity)/2)
        ionized = es_map(source, capacity, threshold, 1.)
        path = output/'checkpoints'/f'node_{index:04d}.npz'
        with path.with_suffix('.npz.partial').open('wb') as stream:
            np.savez_compressed(stream, ionized=ionized, source_emitted=source)
        path.with_suffix('.npz.partial').replace(path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        emitted, absorbed = total(source), total(ionized)
        row = dict(node_index=index, redshift=float(redshifts[index]), time_seconds=float(times[index]),
                   path=str(path.resolve()), sha256=digest, Q_mass=absorbed/total(capacity),
                   Q_volume=float(np.mean(ionized/capacity)), xHI_mass=1-absorbed/total(capacity),
                   xHI_volume=1-float(np.mean(ionized/capacity)), emitted=emitted,
                   photon_residual=emitted-absorbed,
                   photon_residual_over_emitted=(emitted-absorbed)/emitted if emitted else 0.,
                   spectra=field_spectrum(history, ionized, redshifts[index]))
        checkpoints.append(row)
        if 'native_xhi' in history:
            native.append(dict(node_index=index, redshift=float(redshifts[index]),
                               xHI_volume=float(np.mean(history['native_xhi'][index], dtype=np.float64))))
    result = dict(scope='Matched-source fixed-gas ES reconstruction at each time; no photon calibration',
                  model='matched_es', checkpoints=checkpoints, filtering=filtering,
                  native_es_context=native,
                  native_es_context_scope='Independent native coevals use native cumulative source proxies and evolving density; not a source-matched control',
                  seconds=time.perf_counter()-started)
    _json(output/'summary.json', result)
    return result


def compare_histories(new, reference, history):
    """Compare only equal physical times and independently recorded source counts."""
    old = {row['node_index']: row for row in reference['checkpoints']}
    capacity = history['capacity']
    rows = []
    for row in new['checkpoints']:
        if row['node_index'] not in old:
            continue
        prior = old[row['node_index']]
        a, b = checkpoint_fields(row), checkpoint_fields(prior)
        if row['time_seconds'] != prior['time_seconds']:
            raise ValueError('Checkpoint node labels refer to different physical times')
        emitted = total(a['source_emitted'])
        old_emitted = total(b['source_emitted'])
        scale = max(emitted, old_emitted, np.finfo(float).tiny)
        source_l1 = total(abs(a['source_emitted']-b['source_emitted']))/scale
        delta = a['ionized']-b['ionized']
        fine = None
        same_refinement = ('subcells_per_axis' in a and 'subcells_per_axis' in b
                           and int(a['subcells_per_axis']) == int(b['subcells_per_axis']))
        if same_refinement and emitted > 0:
            fine = fine_state_metrics(capacity, int(a['subcells_per_axis']), a, b, emitted)
        an, bn = row.get('spectra'), prior.get('spectra')
        power_change = None
        if an and bn and an['k0_band'] and bn['k0_band']:
            pn, po = an['k0_band']['P_21_mK2_hMpc3'], bn['k0_band']['P_21_mK2_hMpc3']
            if po > 0:
                power_change = pn/po-1
        retained = max(total(x.get('source_retained', np.array([0.])))/scale for x in (a, b))
        rows.append(dict(node_index=row['node_index'], redshift=row['redshift'],
                         time_seconds=row['time_seconds'],
                         new_Q_mass=row['Q_mass'], reference_Q_mass=prior['Q_mass'],
                         absolute_Q_mass_change=abs(row['Q_mass']-prior['Q_mass']),
                         xHI_volume_change=row['xHI_volume']-prior['xHI_volume'],
                         emitted=emitted, reference_emitted=old_emitted,
                         source_L1_over_emitted=source_l1,
                         source_budgets_match=bool(source_l1 <= 1e-12),
                         original_grid_field_L1_over_emitted=total(abs(delta))/scale,
                         resolved_subcell_L1_over_emitted=fine['subcell_L1_over_emitted'] if fine else None,
                         resolved_subcells=fine,
                         k0_power_change=power_change,
                         absolute_k0_power_change=abs(power_change) if power_change is not None else None,
                         maximum_retained_fraction=retained))
    return rows


def plot_results(root, summaries, controls, comparisons):
    """Small reproducible scientific figure; aggregate results remain in JSON."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for report in summaries:
        spec = report.get('task', {})
        if spec.get('seed') != min(r['task']['seed'] for r in summaries):
            continue
        if spec.get('stride') == 1:
            rows = report['checkpoints']
            axes[0].plot([r['redshift'] for r in rows], [r['xHI_volume'] for r in rows],
                         label=f"PC f={spec['f']}, K={spec['angles']}")
    if controls:
        rows = controls[0]['checkpoints']
        axes[0].plot([r['redshift'] for r in rows], [r['xHI_volume'] for r in rows], 'k--', label='Same-source ES')
    axes[0].set(xlabel='Redshift', ylabel=r'$\langle x_{\rm HI}\rangle_V$')
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=7)
    for group in comparisons:
        task = group.get('new', {})
        if task.get('seed') != min(r['task']['seed'] for r in summaries):
            continue
        values = group['checkpoints']
        if group.get('kind') != 'temporal':
            continue
        label=f"f={task['f']}, K={task['angles']}, {group['reference']['stride']}→{task['stride']}"
        for ax, key in zip(axes[1:], ('resolved_subcell_L1_over_emitted', 'absolute_k0_power_change')):
            valid=[r for r in values if r.get(key) is not None and .1 <= r['new_Q_mass'] <= .95]
            ax.plot([r['redshift'] for r in valid], [100*r[key] for r in valid], label=label, linewidth=1)
    for ax in axes[1:]:
        ax.axhline(1, color='.4', ls=':', lw=1)
        ax.set_xlabel('Redshift')
        ax.set_yscale('symlog', linthresh=.01)
        ax.invert_xaxis()
    axes[1].set_ylabel('Time refinement: fine-cell L1 / photons [%]')
    axes[2].set_ylabel('Time refinement: absolute power change [%]')
    axes[2].legend(fontsize=6, ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Prescribed native sources; fixed gas; first seed shown', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(Path(root)/f'evolution_validation.{suffix}', dpi=170)
    plt.close(fig)
