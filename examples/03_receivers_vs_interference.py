#!/usr/bin/env python3
"""Receivers under other-cell interference:  python examples/03_receivers_vs_interference.py

Two co-scheduled UEs, 4 gNB antennas, one interferer. All five receivers are run on
the same slots, without interference (IoT 0) and with 10 dB of it. Expect:

  * IoT 0:   IRC = L-MMSE exactly (nothing coloured to reject); MR is far behind
             because it does not separate the two UEs;
  * IoT 10:  ZF and L-MMSE lose the 10 dB and more, IRC clearly less: one interferer
             occupies one spatial direction and the gNB has antennas to spare.

A minute or two on a CPU (8 PRB). IOT_COV chooses how IRC learns the interference
covariance: "perfect" (genie) or "residual" (measured on the DMRS, realistic).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from nr_ul_sim import NRUplinkSimulator, SimConfig  # noqa: E402
from nr_ul_sim.cli import save_campaign  # noqa: E402

IOT_COV = "perfect"        # perfect, residual, incm_oas, estimated
OUTDIR = Path("results/examples")

cfg = SimConfig(
    channel="cdl-c",
    num_ue=2,
    num_layers=1,
    num_rx_ant=4,
    modulation="qpsk",
    snr_db=list(range(-8, 21, 4)),
    iot_db=[0.0, 10.0],
    num_interferers=1,
    receivers=("mr", "zf", "lmmse", "ideal_mmse", "irc"),
    channel_estimators=("ls_lin",),
    iot_cov=IOT_COV,
    num_prb=8,
    batch_size=4,
    max_mc_iter=6,
    seed=3,
)

campaign = NRUplinkSimulator(cfg).run()

iots = list(campaign["iot"])
print(f"\nWorking point [dB] at BER = {cfg.target_ber:g}  (IRC covariance: {IOT_COV})")
print(f"  {'receiver':12s}" + "".join(f"{'IoT ' + i + ' dB':>14s}" for i in iots) + f"{'loss':>10s}")
for name in cfg.receivers:
    wps = [campaign["iot"][i][name]["working_point_db"] for i in iots]
    cells = "".join(f"{wp:14.2f}" if wp is not None else f"{'not reached':>14s}" for wp in wps)
    loss = f"{wps[-1] - wps[0]:10.2f}" if None not in wps else f"{'-':>10s}"
    print(f"  {name:12s}{cells}{loss}")

json_path, plots = save_campaign(campaign, OUTDIR)
print(f"\nWrote {json_path}")
for png_path in plots:
    print(f"Wrote {png_path}")
