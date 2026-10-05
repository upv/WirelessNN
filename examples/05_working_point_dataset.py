#!/usr/bin/env python3
"""Working-point dataset:  python examples/05_working_point_dataset.py

Each scenario freezes one channel realization, then searches the SNR where the coded
BER of IRC crosses TARGET_BER for every estimator and modulation. Output goes to OUTDIR:
meta.jsonl / dataset.csv, channels/<id>/{H,H_int,R_iot}.npy and the packed dataset.npz.

The knobs below are a small demo (4 scenarios, 8 PRB, about three minutes on a CPU). The
values of the tracked dataset/irc_ce100 run are given in the comments marked "full:".
The same builder from the command line: python -m nr_ul_sim.dataset --help
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np  # noqa: E402

from nr_ul_sim.dataset import DatasetConfig, generate_dataset, load_dataset  # noqa: E402

NUM_SAMPLES = 4            # random scenarios; each yields one record per modulation (full: 100)
START_INDEX = 0            # shard offset, scenario i is reproducible from (SEED, i)
OUTDIR = Path("dataset/example")
SEED = 0
RESUME = False             # True: append to an existing OUTDIR (False refuses to overwrite one)

CHANNELS = ("cdl-b", "cdl-c", "umi", "uma")
MODULATIONS = ("qpsk", "qam16", "qam64")
RECEIVERS = ("irc",)
CHANNEL_ESTIMATORS = ("perfect", "ls_nn", "ls_lin", "ls_lin_time_avg", "lmmse_ce")
NUM_UE = (1, 2, 3, 4)
RANKS = (1, 2)
MAX_TOTAL_LAYERS = 4       # up to 8 (4 UE x rank 2) with DMRS length 2
NUM_RX_ANT = (4,)
SPEED_KMH = (3.0, 10.0)    # uniform
DELAY_SPREAD_NS = (30.0, 400.0)   # log-uniform, CDL only (UMi/UMa draw their own)
IOT_DB = (0.0, 20.0)       # uniform INR
P_NO_IOT = 0.25            # share of scenarios with no other-cell interference
NUM_INTERFERERS = (1, 2, 3)

NUM_PRB = 8                # full: 68 (816 subcarriers)
FFT_SIZE = 1024
CHANNEL_MODE = "frozen"    # frozen = label belongs to the stored tensor

TARGET_BER = 0.01
SNR_RANGE = (-20.0, 40.0)  # 64QAM needs extra headroom
COARSE_STEP = 4.0          # bracketing grid, then bisection down to REFINE_DB
REFINE_DB = 0.5
BATCH_SIZE = 2
MAX_MC_ITER = 4            # full: 10
NUM_TARGET_BIT_ERRORS = 200

NUM_FREQ_BINS = 64         # stored tensor: [rx ant, streams, symbols, subcarriers]
NUM_SYMBOL_BINS = 4

dcfg = DatasetConfig(
    num_samples=NUM_SAMPLES,
    start_index=START_INDEX,
    outdir=OUTDIR,
    seed=SEED,
    resume=RESUME,
    channels=CHANNELS,
    modulations=MODULATIONS,
    receivers=RECEIVERS,
    channel_estimators=CHANNEL_ESTIMATORS,
    num_ue_choices=NUM_UE,
    rank_choices=RANKS,
    max_total_layers=MAX_TOTAL_LAYERS,
    num_rx_ant_choices=NUM_RX_ANT,
    speed_kmh_range=SPEED_KMH,
    delay_spread_ns_range=DELAY_SPREAD_NS,
    iot_db_range=IOT_DB,
    p_no_iot=P_NO_IOT,
    num_interferer_choices=NUM_INTERFERERS,
    num_prb=NUM_PRB,
    fft_size=FFT_SIZE,
    channel_mode=CHANNEL_MODE,
    target_ber=TARGET_BER,
    snr_min=SNR_RANGE[0],
    snr_max=SNR_RANGE[1],
    coarse_step=COARSE_STEP,
    refine_db=REFINE_DB,
    batch_size=BATCH_SIZE,
    max_mc_iter=MAX_MC_ITER,
    num_target_bit_errors=NUM_TARGET_BIT_ERRORS,
    num_freq_bins=NUM_FREQ_BINS,
    num_symbol_bins=NUM_SYMBOL_BINS,
)

print("Working-point dataset (IRC × channel estimation)")
for key, value in dcfg.summary().items():
    print(f"  {key}: {value}")
print()

npz_path = generate_dataset(dcfg)
data = load_dataset(npz_path)

print(f"\nWrote {npz_path}")
print(f"  H      {data['H'].shape} {data['H'].dtype}   (samples, rx ant, streams, sym, sc)")
print(f"  R_iot  {data['R_iot'].shape}")
print(f"  features {[str(n) for n in data['feature_names']]}")
for receiver in data["receivers"]:
    wp = data[f"wp_{receiver}"]
    ok = wp[np.isfinite(wp)]
    if ok.size:
        print(f"  wp_{receiver:16s} {ok.size}/{wp.size} reached, "
              f"{ok.min():6.1f} .. {ok.max():6.1f} dB")
    else:
        print(f"  wp_{receiver:16s} never reached")
