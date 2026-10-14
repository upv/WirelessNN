"""Aggregate sampling/calibration facts without exporting individual channel records."""
import argparse
import json
from pathlib import Path
import numpy as np
from .analyze import load_rows, save_csv


def audit(root, outdir):
    root, outdir = Path(root), Path(outdir)
    meta = [json.loads((root/'records'/r['id']/'meta.json').read_text()) for r in load_rows(root)]
    summary = []
    def add(name, values, unit):
        a = np.asarray(values, dtype=float)
        summary.append(dict(quantity=name, n=len(a), unit=unit, minimum=float(a.min()),
                            mean=float(a.mean()), median=float(np.median(a)), maximum=float(a.max())))
    add('snr_requested', [m['snr_db'] for m in meta], 'dB')
    add('measured_minus_requested_snr', [m['measured_snr_db']-m['snr_db'] for m in meta], 'dB')
    add('measured_minus_requested_inr', [m['measured_inr_db']-m['iot_db'] for m in meta if m['iot_db']], 'dB')
    add('measured_noise_over_nominal', [m['measured_power']['noise']/m['noise_variance'] for m in meta], 'linear')
    add('active_band_channel_power', [m['measured_power']['h_true'] for m in meta], 'linear')
    add('target_speed', [m['target_links'][0]['speed_kmh'] for m in meta], 'km/h')
    add('interferer_speed', [v['speed_kmh'] for m in meta for v in m['interferer_links']], 'km/h')
    add('interferer_count_configured', [m['num_interferers_configured'] for m in meta], 'count')
    for ch in ('cdl-b','cdl-c','cdl-d','uma','umi'):
        part = [m for m in meta if m['channel'] == ch]
        add(ch+'_ds_parameter', [m['target_links'][0].get('ds_ns',m['target_links'][0].get('lsp_ds_ns')) for m in part], 'ns')
    coverage = []
    for key in ('split', 'channel', 'iot_db', 'modulation', 'num_interferers_configured', 'num_interferers_active'):
        for value in sorted({m[key] for m in meta}):
            coverage.append(dict(parameter=key, value=value, count=sum(m[key]==value for m in meta)))
    outdir.mkdir(parents=True, exist_ok=True)
    save_csv(outdir/'sampling.csv', summary)
    save_csv(outdir/'coverage.csv', coverage)
    components = []
    for iot in (0, 10, 20):
        values = []
        for m in meta:
            if m['split'] != 'test' or m['iot_db'] != iot:
                continue
            with np.load(root/'records'/m['id']/'sample.npz', allow_pickle=False) as z:
                terms = [float(np.mean(abs(z[k])**2)) for k in ('h_ls_noise','h_ls_interference')]
                terms.append(float(np.mean(abs(z['h_ls_clean']-z['h_true'])**2)))
                total = float(np.mean(abs(z['h_ls']-z['h_true'])**2))
                values.append([*terms, total-sum(terms), total])
        mean = np.mean(values, axis=0)
        components.append(dict(iot_db=iot, n=len(values), **dict(zip(
            ('noise_mse','interference_mse','interpolation_mse','cross_terms','total_mse'), mean))))
    save_csv(outdir/'error_components.csv', components)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--outdir', type=Path, default=Path('experiments/ce_denoising/reports'))
    a=p.parse_args()
    audit(a.dataset, a.outdir)
