"""Paired, fixed-slot comparison of measured-covariance IRC and soft windows.

    .venv/bin/python scripts/campaigns/benchmark_irc_real.py --outdir results/irc_real_comparison

CDL-C 1 UE and UMa 2 UE at 68 PRB, IoT 0 / 10 / 20 dB, L-MMSE / IRC / IRC-real with
the soft-window estimator in both noise modes, 64 slots per point with paired seeds.
Writes benchmark.json (raw error counts) and comparison.png.
"""
import argparse
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import sionna.phy
from nr_ul_sim import NRUplinkSimulator, SimConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--outdir", default="results/irc_real_comparison")
args = parser.parse_args()

rows=[]
t0=time.time()
for channel, nue in [('cdl-c', 1), ('uma', 2)]:
    for mode in ['thermal', 'outside']:
        cfg=SimConfig(channel=channel, num_ue=nue, num_prb=68,
                      receivers=('lmmse','irc','irc_real'),
                      channel_estimators=('ls_soft_window',),
                      ce_soft_noise_mode=mode, num_interferers=1,
                      batch_size=8, max_mc_iter=8, num_target_bit_errors=10**12,
                      num_target_block_errors=10**12, seed=121)
        sim=NRUplinkSimulator(cfg)
        for iot in [0.0,10.0,20.0]:
            for snr in [-8.0,-4.0,0.0,4.0,8.0,12.0,16.0]:
                # Same random input sequence for old/new, not new channels
                # just because another estimator was evaluated first.
                seed=121+int(iot)*100+int(snr)
                sionna.phy.config.seed=seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                stats=sim.measure_point(snr,iot)
                rec=dict(channel=channel,num_ue=nue,mode=mode,iot_db=iot,snr_db=snr,
                         seed=seed,stats={k:v.as_dict() for k,v in stats.items()})
                rows.append(rec)
                print(channel,mode,iot,snr,{k:round(v.ber,6) for k,v in stats.items()},flush=True)
path=Path(args.outdir)
path.mkdir(parents=True,exist_ok=True)
(path/'benchmark.json').write_text(json.dumps(dict(rows=rows,duration_s=time.time()-t0,
    note='68 PRB, rank 1, one interferer, 8 slots/batch, fixed 8 batches, seeds paired; short validation not a production campaign'),indent=2))
print('Saved',path/'benchmark.json',flush=True)


# Render the saved paired comparison.
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path=Path(args.outdir)
data=json.loads((path/'benchmark.json').read_text())
rows=data['rows']
fig, axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
series=[('outside','lmmse','L-MMSE / new soft','#888888','-'),
        ('thermal','irc','IRC wideband true cov / old soft','#c44e52','--'),
        ('outside','irc','IRC wideband true cov / new soft','#c44e52','-'),
        ('thermal','irc_real','IRC DMRS + OAS / old soft','#4c72b0','--'),
        ('outside','irc_real','IRC DMRS + OAS / new soft','#4c72b0','-')]
for ri,ch in enumerate(['cdl-c','uma']):
    for ci,iot in enumerate([0.,10.,20.]):
        ax=axes[ri,ci]
        for mode,rx,label,color,ls in series:
            rs=sorted([r for r in rows if r['channel']==ch and r['mode']==mode and r['iot_db']==iot],key=lambda r:r['snr_db'])
            snrs=np.array([r['snr_db'] for r in rs])
            ber=np.array([r['stats'][rx]['ber'] for r in rs])
            shown=np.array([max(r['stats'][rx]['ber'],0.5/r['stats'][rx]['num_bits']) for r in rs])
            ax.semilogy(snrs,shown,ls=ls,color=color,marker='o',markersize=3,label=label)
            if (ber==0).any():
                ax.scatter(snrs[ber==0],shown[ber==0],marker='v',facecolors='none',edgecolors=color,s=32)
        ax.axhline(.01,color='black',ls=':',lw=1)
        ax.set_title(f'{ch.upper()} | {1 if ch=="cdl-c" else 2} UE | IoT {iot:g} dB',fontsize=11)
        ax.set_xlabel('Per-antenna SNR (dB)')
        ax.set_ylabel('Decoded BER (errors / bits)')
        ax.set_ylim(5e-7,.7)
        ax.grid(True,which='both',alpha=.2)
fig.suptitle('WirelessNN: measured-covariance IRC and empirical soft-window floor',fontsize=15)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,.025))
fig.text(.5,.005,'Source: GPU run 2026-10-06 | 68 PRB, QPSK, rank 1, 4 RX, 1 interferer | 64 slots / point, paired seeds. Zero errors shown at 0.5/N bits (v); this is not a confidence bound.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.115,1,.955])
fig.savefig(path/'comparison.png',dpi=150)
print('rows',len(rows),'duration_s',round(data['duration_s'],2))
print('num_blocks',sorted({r['stats']['irc']['num_blocks'] for r in rows}))
print(path/'comparison.png')
