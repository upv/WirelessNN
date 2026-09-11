#!/usr/bin/env python3
"""Playground for the NR PUSCH uplink platform.

Edit the knobs below, then:

    .venv/bin/python play.py

CPU-friendly defaults (8 PRB, a few SNR points). For the spec allocation
set NUM_PRB = 68 and widen SNR_DB / MAX_MC_ITER.
"""

from pathlib import Path

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.cli import save_campaign

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

# Channel: "cdl-b", "cdl-c", "umi", "uma"
CHANNEL = "cdl-c"
DELAY_SPREAD_NS = 300.0  # CDL only; UMi/UMa ignore this

NUM_UE = 1  # 1–4 co-scheduled UEs
NUM_RX_ANT = 4  # gNB antennas
RANK = 1  # layers per UE: 1 or 2
SPEED_KMH = 3.0  # 3–10 typical

MODULATION = "qpsk"  # "qpsk" or "qam16"

# Per-antenna SNR [dB] and other-cell IoT / INR [dB]
# IoT = 0 means no neighbouring-cell interference
SNR_DB = [-10.0, -5.0, 0.0, 5.0, 10.0]
IOT_DB = [0.0, 10.0]

RECEIVERS = ("mr", "lmmse", "zf", "irc")

# Waveform. Spec: 68 PRB, 816 used subcarriers, FFT 1024, 17 RBG.
NUM_PRB = 8
FFT_SIZE = 1024

PERFECT_CSI = False  # False → DMRS Type-1 LS channel estimation
BATCH_SIZE = 2
MAX_MC_ITER = 5  # increase for smoother BER curves
TARGET_BER = 0.01
SEED = 42
OUTDIR = Path("results")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

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
    num_prb=NUM_PRB,
    fft_size=FFT_SIZE,
    perfect_csi=PERFECT_CSI,
    batch_size=BATCH_SIZE,
    max_mc_iter=MAX_MC_ITER,
    target_ber=TARGET_BER,
    seed=SEED,
)

print("NR PUSCH uplink playground")
for key, value in cfg.summary().items():
    print(f"  {key}: {value}")

sim = NRUplinkSimulator(cfg)
campaign = sim.run(verbose=True)

print("\nWorking points (SNR where BER = {:.3g})".format(TARGET_BER))
for iot, curves in campaign["iot"].items():
    print(f"  IoT = {iot} dB")
    for name, res in curves.items():
        wp = res["working_point_db"]
        txt = f"{wp:.2f} dB" if wp is not None else "not reached"
        print(f"    {name:6s}  {txt}")

json_path, png_path = save_campaign(campaign, OUTDIR)
print(f"\nWrote {json_path}")
print(f"Wrote {png_path}")
