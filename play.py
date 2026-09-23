#!/usr/bin/env python3
"""Edit knobs, then:  .venv/bin/python play.py"""

from pathlib import Path

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.cli import save_campaign

CHANNEL = "cdl-c"          # cdl-b, cdl-c, umi, uma
DELAY_SPREAD_NS = 300.0    # CDL only
NUM_UE = 1                 # 1-4
NUM_RX_ANT = 4
RANK = 1                   # 1 or 2
SPEED_KMH = 3.0
MODULATION = "qam16"       # qpsk, qam16, qam64
SNR_DB = list(range(-20, 31, 5))
IOT_DB = [0.0, 10.0]       # 0 = no other-cell interference
RECEIVERS = ("irc",)
CHANNEL_ESTIMATORS = ("perfect", "ls_nn", "ls_lin", "ls_lin_time_avg", "lmmse_ce")
NUM_PRB = 8                # spec: 68
FFT_SIZE = 1024
PERFECT_CSI = False        # False = DMRS estimators below (perfect is a CE option)
BATCH_SIZE = 2
MAX_MC_ITER = 5
TARGET_BER = 0.01
SEED = 42
OUTDIR = Path("results")

cfg = SimConfig(
    channel=CHANNEL,
    delay_spread_ns=DELAY_SPREAD_NS,
    num_ue=NUM_UE,
    num_rx_ant=NUM_RX_ANT,
    num_layers=RANK,
    speed_kmh=SPEED_KMH,
    modulation=MODULATION,
    snr_db=SNR_DB,
    iot_db=IOT_DB,
    receivers=RECEIVERS,
    channel_estimators=CHANNEL_ESTIMATORS,
    num_prb=NUM_PRB,
    fft_size=FFT_SIZE,
    perfect_csi=PERFECT_CSI,
    batch_size=BATCH_SIZE,
    max_mc_iter=MAX_MC_ITER,
    target_ber=TARGET_BER,
    seed=SEED,
)

print("NR PUSCH uplink")
for key, value in cfg.summary().items():
    print(f"  {key}: {value}")

campaign = NRUplinkSimulator(cfg).run(verbose=True)

print(f"\nWorking points (BER = {TARGET_BER:g})")
for iot, curves in campaign["iot"].items():
    print(f"  IoT = {iot} dB")
    for name, res in curves.items():
        wp = res["working_point_db"]
        print(f"    {name:12s}  {wp:.2f} dB" if wp is not None else f"    {name:12s}  not reached")

json_path, plots = save_campaign(campaign, OUTDIR)
print(f"\nWrote {json_path}")
for png_path in plots:
    print(f"Wrote {png_path}")
