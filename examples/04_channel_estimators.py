#!/usr/bin/env python3
"""Channel estimators side by side:  python examples/04_channel_estimators.py

Part 1 measures the channel-estimation error (NMSE) of every estimator on the same
slots, at a few SNRs. Part 2 runs the IRC receiver with each of them and reports the
working-point loss against perfect CSI — the number that matters at link level.

A couple of minutes on a CPU (8 PRB). The trained estimators (denoise_nn, lmmse_data,
a_mmse, ...) are added automatically when models/paper_ce is present.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import torch  # noqa: E402

from nr_ul_sim import NRUplinkSimulator, SimConfig  # noqa: E402
from nr_ul_sim.cli import save_campaign  # noqa: E402
from nr_ul_sim.paper_ce import DEFAULT_MODEL_DIR  # noqa: E402

ESTIMATORS = ["perfect", "ls_nn", "ls_lin", "ls_lin_time_avg", "lmmse_ce", "lmmse_exp",
              "ls_hard_window", "ls_soft_window", "ls_fir"]
if (DEFAULT_MODEL_DIR / "a_mmse.pt").exists():
    ESTIMATORS += ["denoise_nn", "lmmse_data", "a_mmse"]
NMSE_SNR_DB = [0.0, 10.0, 20.0]
OUTDIR = Path("results/examples")

cfg = SimConfig(
    channel="cdl-c",
    delay_spread_ns=300.0,
    num_ue=1,
    modulation="qam16",
    snr_db=list(range(-2, 19, 2)),
    iot_db=[0.0],
    receivers=("irc",),
    channel_estimators=tuple(ESTIMATORS),
    num_prb=8,
    batch_size=4,
    max_mc_iter=6,
    seed=4,
)
sim = NRUplinkSimulator(cfg)

# --- 1) NMSE of the estimate itself ----------------------------------------------
print("Channel-estimation NMSE [dB]")
print(f"  {'estimator':16s}" + "".join(f"{f'SNR {s:g} dB':>12s}" for s in NMSE_SNR_DB))
nmse = {name: [] for name in ESTIMATORS if name != "perfect"}
for snr in NMSE_SNR_DB:
    slot = sim.generate_slot(16, snr, 0.0)
    sim.estimate_csi(slot)
    h = slot["h_perf"]
    for name in nmse:
        h_hat, _err_var = slot["csi"][name]
        ratio = (h_hat - h).abs().pow(2).mean() / h.abs().pow(2).mean()
        nmse[name].append(10 * torch.log10(ratio).item())
for name, values in nmse.items():
    print(f"  {name:16s}" + "".join(f"{v:12.1f}" for v in values))

# --- 2) what it costs at link level ----------------------------------------------
campaign = sim.run()
curves = campaign["iot"]["0.0"]          # one receiver, many estimators -> keys are estimator names
ref = curves["perfect"]["working_point_db"]
print(f"\nIRC working point at BER = {cfg.target_ber:g} and loss against perfect CSI [dB]")
for name in ESTIMATORS:
    wp = curves[name]["working_point_db"]
    if wp is None:
        print(f"  {name:16s} not reached")
    else:
        print(f"  {name:16s} {wp:6.2f}   {wp - ref:+5.2f}" if ref is not None else f"  {name:16s} {wp:6.2f}")

json_path, plots = save_campaign(campaign, OUTDIR)
print(f"\nWrote {json_path}")
for png_path in plots:
    print(f"Wrote {png_path}")
