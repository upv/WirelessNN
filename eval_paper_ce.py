#!/usr/bin/env python3
"""NMSE of every channel estimator vs SNR inside the link simulator (68 PRB).

    .venv/bin/python eval_paper_ce.py --out ~/reports/paper_ce/nmse

The paper estimators are trained on UMa (EqDeepRx protocol); UMi, CDL-B and
CDL-C are out of the training distribution. Writes nmse.json and nmse.png.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sionna.phy
import torch

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.plotting import RECEIVER_STYLE

ESTIMATORS = ("perfect", "ls_nn", "ls_lin", "ls_lin_time_avg", "lmmse_exp", "ls_hard_window",
              "ls_soft_window", "ls_fir", "denoise_nn", "lmmse_data", "lmmse_data_1d", "a_mmse", "ra_a_mmse")
SCENARIOS = [
    # (label, channel, delay spread ns, speed km/h, #UE, rank)
    ("UMa 1UE r1 30km/h (train dist.)", "uma", None, 30.0, 1, 1),
    ("UMa 2UE r2 30km/h (train dist.)", "uma", None, 30.0, 2, 2),
    ("UMi 1UE r1 3km/h", "umi", None, 3.0, 1, 1),
    ("CDL-B 100ns 1UE r1 3km/h", "cdl-b", 100.0, 3.0, 1, 1),
    ("CDL-C 300ns 1UE r1 3km/h", "cdl-c", 300.0, 3.0, 1, 1),
    ("CDL-C 300ns 2UE r1 60km/h", "cdl-c", 300.0, 60.0, 2, 1),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snr", default="-10,-5,0,5,10,15,20,25,30")
    ap.add_argument("--slots", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    snrs = [float(s) for s in args.snr.split(",")]
    res = {}
    t0 = time.time()
    for label, ch, ds, speed, nue, rank in SCENARIOS:
        cfg = SimConfig(channel=ch, delay_spread_ns=ds, speed_kmh=speed, num_ue=nue, num_layers=rank,
                        num_prb=68, receivers=("irc",), channel_estimators=ESTIMATORS, snr_db=snrs,
                        iot_db=[0.0], batch_size=args.batch, seed=11)
        sim = NRUplinkSimulator(cfg)
        cur = {e: [] for e in ESTIMATORS if e != "perfect"}
        calib = {e: [] for e in cur}
        for snr in snrs:
            # same channel realisations at every SNR point (common random numbers)
            sionna.phy.config.seed = cfg.seed
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            err = {e: 0.0 for e in cur}
            rep = {e: 0.0 for e in cur}
            pw = 0.0
            for _ in range(args.slots // args.batch):
                with torch.no_grad():
                    slot = sim.generate_slot(args.batch, snr, 0.0)
                    sim.estimate_csi(slot)
                h = slot["csi"]["perfect"][0]
                pw += float(h.abs().pow(2).mean())
                for e in cur:
                    hh, ev = slot["csi"][e]
                    err[e] += float((hh - h).abs().pow(2).mean())
                    rep[e] += float(torch.broadcast_to(ev, hh.shape).mean())
            for e in cur:
                cur[e].append(10 * math.log10(err[e] / pw))
                calib[e].append(rep[e] / err[e])
        res[label] = {"snr_db": snrs, "nmse_db": cur, "errvar_reported_over_measured": calib}
        i = snrs.index(10.0) if 10.0 in snrs else len(snrs) // 2
        print(f"{time.strftime('%H:%M:%S')} {label} @ {snrs[i]:g} dB: " + ", ".join(
            f"{e} {v[i]:.1f}" for e, v in cur.items()), flush=True)
        del sim
        torch.cuda.empty_cache()
    (args.out / "nmse.json").write_text(json.dumps(res, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), sharey=True)
    for ax, (label, r) in zip(axes.ravel(), res.items()):
        for e, v in r["nmse_db"].items():
            st = RECEIVER_STYLE.get(e, {})
            ax.plot(r["snr_db"], v, marker=st.get("marker", "."), color=st.get("color"), ms=4,
                    lw=1.8 if e in ("denoise_nn", "lmmse_data", "a_mmse", "ra_a_mmse", "ls_fir", "lmmse_data_1d") else 0.9,
                    label=st.get("label", e))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("SNR per antenna [dB]")
        ax.grid(True, ls=":", alpha=0.5)
    axes[0, 0].set_ylabel("NMSE [dB]")
    axes[1, 0].set_ylabel("NMSE [dB]")
    axes[0, 0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Channel-estimation NMSE, 68 PRB, DMRS type-1 add-pos 1 (paper estimators trained on UMa)")
    fig.tight_layout()
    fig.savefig(args.out / "nmse.png", dpi=130)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
