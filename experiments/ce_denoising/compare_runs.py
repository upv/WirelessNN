"""Compare independent cohorts and frozen pilot/large fits on the same test set."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .analyze import (bootstrap_interval, load_rows, load_sample, nominal_pilot_variance,
                      predict, save_csv, summarize)
from .generate import digest, write_json
from .losses import symbol_bce


def load_metrics(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ('mse', 'target_power', 'bce', 'snr_db'):
            row[key] = float(row[key])
        for key in ('iot_db', 'snr_bin_low_db', 'seed', 'num_interferers_configured'):
            row[key] = int(row[key])
    return rows


def paired_change(before, after, metric):
    """Pair whole channel realizations by ID, never by file or method row order."""
    def indexed(rows):
        result = {r['id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate realization in paired comparison')
        return result
    old, new = indexed(before), indexed(after)
    if not old or old.keys() != new.keys():
        raise ValueError('Paired comparison requires the same nonempty test cohort')
    ids = sorted(old)
    if any(old[k]['seed'] != new[k]['seed'] for k in ids):
        raise ValueError('Realization seeds differ for matching IDs')
    delta = np.asarray([new[k][metric] - old[k][metric] for k in ids])
    return float(delta.mean()), bootstrap_interval(delta)


def load_fits(artifacts):
    with np.load(artifacts / 'linear_fits.npz', allow_pickle=False) as data:
        return {name: {k[len(name)+1:]: data[k] for k in data.files if k.startswith(name+'_')}
                for name in ('ridge', 'wiener')}


def frozen_metrics(root, rows, fits, selection):
    metrics = []
    for i, row in enumerate(rows, 1):
        a = load_sample(root, row, ('ls_pilots', 'h_true', 'y', 'x_tx', 'data_mask'))
        x = a['ls_pilots'].reshape(4, -1)
        nv = 4 * nominal_pilot_variance(row)
        y, tx, mask = [torch.from_numpy(a[k].copy()) for k in ('y', 'x_tx', 'data_mask')]
        for name, method in (('ridge', 'ridge_input_only'), ('wiener', 'lmmse_known_inr')):
            reg = selection[name]['alpha'] * (fits[name]['scale'] if name == 'ridge' else nv/4)
            pred = predict(fits[name], x, reg).reshape(a['h_true'].shape).astype(np.complex64)
            with torch.no_grad():
                bce = float(symbol_bce(torch.from_numpy(pred), y, tx, mask, row['modulation'], nv))
            metrics.append({**row, 'method': method,
                            'mse': float(np.mean(abs(pred-a['h_true'])**2)),
                            'target_power': float(np.mean(abs(a['h_true'])**2)), 'bce': bce})
        if i % 100 == 0 or i == len(rows):
            print(f'Frozen pilot fits: {i}/{len(rows)} test records', flush=True)
    return metrics


def with_bce_ci(rows):
    result = summarize(rows)
    for group in result:
        group['bce_ci95'] = bootstrap_interval([r['bce'] for r in rows if r['method']==group['method']])
    return result


def compare(pilot_root, root, pilot_reports, outdir):
    started = time.perf_counter()
    pilot_root, root, pilot_reports, outdir = map(Path, (pilot_root, root, pilot_reports, outdir))
    old_artifacts, artifacts = pilot_reports.parent / 'artifacts', outdir.parent / 'artifacts'
    old_rows, new_rows = load_rows(pilot_root), load_rows(root)
    if {r['seed'] for r in old_rows} & {r['seed'] for r in new_rows}:
        raise ValueError('Pilot and large datasets have overlapping realization seeds')
    fits = load_fits(old_artifacts)
    selection = json.loads((old_artifacts/'analysis.json').read_text())['selection']
    old_metrics = load_metrics(old_artifacts/'per_record_metrics.csv')
    new_metrics = load_metrics(artifacts/'per_record_metrics.csv')
    # Replay three original pilot records before applying the frozen fits elsewhere.
    replay = frozen_metrics(pilot_root, [r for r in old_rows if r['split']=='test'][:3], fits, selection)
    original = {(r['id'],r['method']):r for r in old_metrics}
    for row in replay:
        for key in ('mse','bce'):
            np.testing.assert_allclose(row[key], original[row['id'],row['method']][key], rtol=2e-6, atol=1e-7)
    frozen = frozen_metrics(root, [r for r in new_rows if r['split']=='test'], fits, selection)
    save_csv(artifacts/'pilot_filters_on_large_test.csv', frozen)
    before, after, fixed = [with_bce_ci(m) for m in (old_metrics,new_metrics,frozen)]
    cohorts = []
    for cohort, groups in (('pilot_180_test', before),('large_4050_test',after),('pilot_fits_large_test',fixed)):
        cohorts.extend(dict(cohort=cohort, **g) for g in groups)
    save_csv(outdir/'cohort_comparison.csv', cohorts)
    paired = []
    for method in ('ridge_input_only','lmmse_known_inr'):
        old = [r for r in frozen if r['method']==method]
        new = [r for r in new_metrics if r['method']==method]
        delta_mse, mse_ci = paired_change(old,new,'mse')
        delta_bce, bce_ci = paired_change(old,new,'bce')
        a = next(r for r in fixed if r['method']==method)
        b = next(r for r in after if r['method']==method)
        paired.append(dict(method=method, n=len(new), pilot_fit_mse=a['mse'], large_fit_mse=b['mse'],
                           mse_change=delta_mse, mse_change_ci95=mse_ci,
                           mse_reduction_percent=100*(1-b['mse']/a['mse']),
                           pilot_fit_bce=a['bce'], large_fit_bce=b['bce'],
                           bce_change=delta_bce,bce_change_ci95=bce_ci,
                           pilot_fit_nmse_db=a['nmse_db'],large_fit_nmse_db=b['nmse_db']))
    save_csv(outdir/'same_test_training_size.csv',paired)
    by_mod = defaultdict(list)
    for row in new_metrics:
        by_mod[row['method'],row['modulation']].append(row)
    save_csv(outdir/'by_modulation.csv', [dict(method=m,modulation=mod,n=len(part),
             mse=float(np.mean([r['mse'] for r in part])),bce=float(np.mean([r['bce'] for r in part])))
             for (m,mod),part in by_mod.items()])
    # Paired uncertainty for the practical LS-only vs conditional-LMMSE contrast.
    contrast = {}
    ridge = [r for r in new_metrics if r['method']=='ridge_input_only']
    lmmse = [r for r in new_metrics if r['method']=='lmmse_known_inr']
    for metric in ('mse','bce'):
        delta, interval = paired_change(ridge,lmmse,metric)
        contrast[metric] = dict(lmmse_minus_ridge=delta,ci95=interval)
    result = dict(cohorts=cohorts,paired_training_size=paired,lmmse_vs_ridge=contrast,
                  pilot_selection=selection,large_selection=json.loads((artifacts/'analysis.json').read_text())['selection'],
                  seed_overlap=0,pilot_replay_passed=True,wall_seconds=time.perf_counter()-started,
                  sources={str(p):digest(p) for p in (Path(__file__),old_artifacts/'linear_fits.npz',artifacts/'linear_fits.npz')})
    write_json(artifacts/'comparison.json', result)
    plot_comparison(outdir,cohorts)
    print(json.dumps(paired,indent=2),flush=True)


def plot_comparison(outdir, cohorts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = ('ls_linear','ls_time_average','dft_fixed_window','ridge_input_only','lmmse_known_inr')
    labels = ('LS','Time avg.','DFT window','Ridge','LMMSE\nknown I+N')
    colors = ('#8295aa','#147d92')
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    x = np.arange(len(methods))
    for j,(cohort,label) in enumerate((('pilot_180_test','Pilot: 180 test channels'),('large_4050_test','Large: 4,050 test channels'))):
        rows={r['method']:r for r in cohorts if r['cohort']==cohort}
        for ax,metric,ci in ((axes[0],'nmse_db','nmse_db_ci95'),(axes[1],'bce','bce_ci95')):
            y=np.array([rows[m][metric] for m in methods])
            bounds=np.array([rows[m][ci] for m in methods]).T
            ax.errorbar(x+(j-.5)*.12,y,yerr=np.maximum(np.vstack([y-bounds[0],bounds[1]-y]),0),
                        fmt='o',capsize=3,label=label,color=colors[j])
    axes[0].set_ylabel('Pooled channel NMSE (dB)')
    axes[1].set_ylabel('BCE (nat/bit; equal record weights)')
    axes[0].axhline(0,color='gray',ls=':',lw=1)
    axes[1].axhline(np.log(2),color='gray',ls=':',lw=1)
    for ax in axes:
        ax.set_xticks(x,labels)
        ax.grid(axis='y',alpha=.25)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2)
    fig.suptitle('Pilot vs large dataset: independent test cohorts, 95% bootstrap intervals')
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(outdir/'pilot_vs_large.png',dpi=180)
    plt.close(fig)
    with (outdir/'by_snr_iot.csv').open() as f:
        groups=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
    for ax,iot in zip(axes,(0,10,20)):
        for method,label in zip((*methods,'perfect_reference'),(*labels,'True H reference')):
            rows=sorted((r for r in groups if r['method']==method and int(r['iot_db'])==iot),key=lambda r:int(r['snr_bin_low_db']))
            ax.plot([int(r['snr_bin_low_db'])+5 for r in rows],[float(r['bce']) for r in rows],'o-',label=label.replace('\n',' '))
        ax.axhline(np.log(2),color='gray',ls=':',lw=1)
        ax.set(title='No interference' if iot==0 else f'IoT = {iot} dB',xlabel='SNR bin centre (dB)',xticks=[-15,-5,5])
        ax.grid(alpha=.25)
    axes[0].set_ylabel('BCE (nat/bit; equal record weights)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('4,050 independent test channels / 450 channels per SNR-IoT cell')
    fig.tight_layout(rect=(0,.14,1,.94))
    fig.savefig(outdir/'bce_vs_snr.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dataset',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--pilot-reports',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    compare(args.pilot_dataset,args.dataset,args.pilot_reports,args.outdir)
