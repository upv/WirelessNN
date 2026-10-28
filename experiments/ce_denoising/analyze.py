"""Train-only linear baselines, validation tuning, held-out MSE/NMSE and BCE."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .generate import digest, write_json
from .losses import symbol_bce


def load_rows(root):
    return [json.loads(line) for line in (Path(root) / 'index.jsonl').read_text().splitlines()]


def load_sample(root, row, keys):
    with np.load(Path(root) / 'records' / row['id'] / 'sample.npz', allow_pickle=False) as d:
        return {key: d[key] for key in keys}


def time_matrix(symbols, num_symbols=14):
    return np.stack([np.interp(np.arange(num_symbols), symbols, np.eye(len(symbols))[i])
                     for i in range(len(symbols))], axis=-1)


def hard_window(pilots, num_sc, dmrs, pos_us=3.0, neg_us=1.0):
    """Fixed linear DFT filter and time interpolation, matching windowed_ce.py."""
    m = pilots.shape[-1]
    half = (m + 1) // 2
    tau = 1.0 / (num_sc * 30e3)
    pos, neg = min(int(np.floor(pos_us * 1e-6 / tau)), half-1), min(int(np.floor(neg_us * 1e-6 / tau)), m-half)
    win = np.zeros(m)
    win[:pos+1] = 1
    if neg:
        win[-neg:] = 1
    taps = np.fft.ifft(pilots, axis=-1) * win
    k = np.arange(m)
    signed = np.where(k < half, k, k-m)
    expanded = np.zeros((*taps.shape[:-1], num_sc), dtype=np.complex128)
    # Pilot block centre is 4*k+1; remove the frequency-origin offset.
    expanded[..., signed % num_sc] = taps * np.exp(2j * np.pi * signed / num_sc)
    freq = np.fft.fft(expanded, axis=-1)
    return np.einsum('td,rdf->rtf', time_matrix(dmrs), freq).astype(np.complex64)


def load_training(root, rows):
    obs, clean, truth = [], [], []
    for row in rows:
        a = load_sample(root, row, ('ls_pilots', 'ls_pilots_clean', 'h_true'))
        obs.append(a['ls_pilots'].reshape(4, -1))
        clean.append(a['ls_pilots_clean'].reshape(4, -1))
        truth.append(a['h_true'].reshape(4, -1))
    return [np.concatenate(x).astype(np.complex128) for x in (obs, clean, truth)]


def linear_fit(x, y):
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x-mx, y-my
    cov = xc.conj().T @ xc / len(x)
    cross = xc.conj().T @ yc / len(x)
    ev, u = np.linalg.eigh((cov + cov.conj().T) / 2)
    return dict(mx=mx, my=my, ev=np.maximum(ev, 0), u=u, cross=u.conj().T @ cross,
                scale=float(np.trace(cov).real / len(cov)))


def predict(fit, x, regularizer):
    z = (x-fit['mx']) @ fit['u']
    z /= (fit['ev'] + max(float(regularizer), 1e-10))
    return z @ fit['cross'] + fit['my']


def nominal_pilot_variance(row):
    # Pilot amplitude sqrt(2), average of two pilots: (N0+I0)/(2*2).
    n0 = 10 ** (-row['snr_db'] / 10)
    inr = 0 if row['iot_db'] == 0 else 10 ** (row['iot_db'] / 10)
    return n0 * (1 + inr) / 4


def tune(root, rows, ridge, wiener):
    samples = [(r, load_sample(root, r, ('ls_pilots', 'h_true'))) for r in rows]
    ridge_grid, wiener_grid = (0.0, .001, .01, .1, 1., 10.), (.25, .5, 1., 2., 4.)
    scores = {}
    for kind, fit, candidates in (('ridge', ridge, ridge_grid), ('wiener', wiener, wiener_grid)):
        values = []
        for alpha in candidates:
            total = 0.
            for row, a in samples:
                reg = alpha * (fit['scale'] if kind == 'ridge' else nominal_pilot_variance(row))
                p = predict(fit, a['ls_pilots'].reshape(4, -1), reg).reshape(a['h_true'].shape)
                total += np.mean(np.abs(p-a['h_true'])**2)
            values.append(float(total / len(samples)))
        best = int(np.argmin(values))
        scores[kind] = dict(alpha=float(candidates[best]), validation_mse=values[best],
                            candidates=list(candidates), mse=values)
    return scores


def bootstrap_interval(x, target_power=None, *, seed=79, db=False):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    idx = rng.integers(len(x), size=(2000, len(x)))
    v = x[idx].mean(1)
    if target_power is not None:
        v /= np.asarray(target_power)[idx].mean(1)
    if db:
        v = 10*np.log10(np.maximum(v, 1e-30))
    return np.quantile(v, [.025, .975]).tolist()


def summarize(records, group_keys=()):
    groups = {}
    for row in records:
        key = tuple(row[k] for k in ('method', *group_keys))
        groups.setdefault(key, []).append(row)
    result = []
    for key, part in groups.items():
        mse = np.asarray([r['mse'] for r in part])
        power = np.asarray([r['target_power'] for r in part])
        result.append(dict(zip(('method', *group_keys), key), n=len(part),
                           mse=float(mse.mean()), nmse_db=float(10*np.log10(max(mse.sum()/power.sum(), 1e-30))),
                           median_nmse_db=float(np.median(10*np.log10(np.maximum(mse/power, 1e-30)))),
                           mse_ci95=bootstrap_interval(mse), nmse_db_ci95=bootstrap_interval(mse, power, db=True),
                           bce=float(np.mean([r['bce'] for r in part]))))
    return result


def save_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(root, outdir):
    root, outdir = Path(root), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    splits = {s: [r for r in rows if r['split'] == s] for s in ('train', 'validation', 'test')}
    recipe = json.loads((root / 'config.json').read_text())['recipe']
    if [len(splits[s]) for s in splits] != recipe['counts']:
        raise ValueError('Dataset incomplete: finish generation before analysis')
    if len({r['id'] for r in rows}) != len(rows) or len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('Duplicate realization IDs or seeds')
    # Dataset integrity checks precede fitting; no target test data enters fits.
    for row in rows:
        d = root/'records'/row['id']
        meta = json.loads((d/'meta.json').read_text())
        for file, sha in meta['sha256'].items():
            if digest(d/file) != sha:
                raise ValueError(f'Checksum mismatch: {d/file}')
    print('Fit covariance and ridge on training split', flush=True)
    obs, clean, truth = load_training(root, splits['train'])
    ridge, wiener = linear_fit(obs, truth), linear_fit(clean, truth)
    del obs, clean, truth
    print('Select regularization on validation split', flush=True)
    selected = tune(root, splits['validation'], ridge, wiener)
    print(json.dumps(selected), flush=True)
    # Fitted filters and detailed per-record scores are runtime artifacts, not Git content.
    artifacts = outdir.parent / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    np.savez_compressed(artifacts/'linear_fits.npz', **{f'{name}_{k}':v for name, fit in (('ridge',ridge),('wiener',wiener)) for k,v in fit.items()})
    metrics = []
    keys = ('ls_pilots', 'h_ls', 'h_ls_clean', 'h_true', 'dmrs_symbols', 'y', 'x_tx', 'data_mask')
    for i, row in enumerate(splits['test']):
        a = load_sample(root, row, keys)
        shape = a['h_true'].shape
        pil = a['ls_pilots'].reshape(4, -1)
        estimates = {
            'zero': np.zeros_like(a['h_true']),
            'ls_linear': a['h_ls'],
            'ls_time_average': np.broadcast_to(a['h_ls'][:, a['dmrs_symbols']].mean(1, keepdims=True), shape),
            'dft_fixed_window': hard_window(a['ls_pilots'], shape[-1], a['dmrs_symbols']),
            'ridge_input_only': predict(ridge, pil, selected['ridge']['alpha'] * ridge['scale']).reshape(shape),
            'lmmse_known_inr': predict(wiener, pil, selected['wiener']['alpha'] * nominal_pilot_variance(row)).reshape(shape),
            'ls_clean_floor': a['h_ls_clean'],
            'perfect_reference': a['h_true'],
        }
        y, x, mask = [torch.from_numpy(a[k].copy()) for k in ('y','x_tx','data_mask')]
        nv = 4 * nominal_pilot_variance(row)
        for method, pred in estimates.items():
            pred = np.asarray(pred, dtype=np.complex64)
            with torch.no_grad():
                bce = float(symbol_bce(torch.from_numpy(pred.copy()), y, x, mask, row['modulation'], nv))
            metrics.append({**row, 'method': method, 'mse': float(np.mean(abs(pred-a['h_true'])**2)),
                            'target_power': float(np.mean(abs(a['h_true'])**2)), 'bce': bce})
        if (i+1) % 30 == 0:
            print(f'Evaluated {i+1}/{len(splits["test"])}', flush=True)
    save_csv(artifacts/'per_record_metrics.csv', metrics)
    aggregate = summarize(metrics)
    by_snr_iot = summarize(metrics, ('snr_bin_low_db', 'iot_db'))
    by_channel = summarize(metrics, ('channel',))
    by_iot = summarize(metrics, ('iot_db',))
    save_csv(outdir/'aggregate.csv', aggregate)
    save_csv(outdir/'by_snr_iot.csv', by_snr_iot)
    save_csv(outdir/'by_channel.csv', by_channel)
    save_csv(outdir/'by_iot.csv', by_iot)
    write_json(artifacts/'analysis.json', dict(selection=selected, aggregate=aggregate,
               by_snr_iot=by_snr_iot, by_channel=by_channel, by_iot=by_iot))
    plot(outdir, by_snr_iot, by_channel)
    for r in aggregate:
        print(f"{r['method']:22s} MSE={r['mse']:.5g} NMSE={r['nmse_db']:.2f} dB BCE={r['bce']:.4f}", flush=True)


def plot(outdir, by_snr_iot, by_channel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = {'ls_linear':'LS', 'ls_time_average':'LS + time averaging', 'dft_fixed_window':'Fixed DFT window',
               'ridge_input_only':'Train-only ridge', 'lmmse_known_inr':'LMMSE, known I+N'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, iot in zip(axes, (0, 10, 20)):
        for method, label in methods.items():
            part = sorted((r for r in by_snr_iot if r['method']==method and r['iot_db']==iot), key=lambda r:r['snr_bin_low_db'])
            x = [r['snr_bin_low_db']+5 for r in part]
            ax.plot(x, [r['nmse_db'] for r in part], 'o-', label=label)
        ax.axhline(0, color='gray', ls=':', label='Zero estimate')
        ax.set(title='No interference' if iot==0 else f'IoT = {iot} dB', xlabel='SNR bin centre (dB)', xticks=[-15,-5,5])
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Pooled channel NMSE (dB)')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle('16 PRB / 4 RX / independent held-out channels')
    fig.tight_layout(rect=(0, .13, 1, .95))
    fig.savefig(outdir/'nmse_vs_snr.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4))
    for method, label in methods.items():
        part = sorted((r for r in by_channel if r['method']==method), key=lambda r:r['channel'])
        ax.plot([r['channel'].upper() for r in part], [r['nmse_db'] for r in part], 'o-', label=label)
    ax.axhline(0, color='gray', ls=':')
    ax.set(ylabel='Pooled channel NMSE (dB)', title='Held-out performance by channel family; all SNR/IoT levels')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir/'nmse_by_channel.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--outdir', type=Path, default=Path('experiments/ce_denoising/reports'))
    a = p.parse_args()
    evaluate(a.dataset, a.outdir)
