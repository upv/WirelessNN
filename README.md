# WirelessNN — 5G NR PUSCH uplink simulator

Link-level uplink simulator for 5G NR PUSCH, built on [NVIDIA Sionna](https://nvlabs.github.io/sionna/) PHY (v2, PyTorch). It sweeps **per-antenna SNR**, compares **MR / L-MMSE / Ideal MMSE / ZF / IRC**, injects **other-cell IoT**, and reports the **working point** (SNR where BER = 0.01).

## What is modelled

| Item | Value |
| --- | --- |
| Channel | CDL-B, CDL-C, 3GPP UMi, 3GPP UMa |
| RMS delay spread | Configurable for CDL (defaults: CDL-B 100 ns, CDL-C 300 ns). UMi/UMa draw DS from TR 38.901 |
| Users | 1–4 co-scheduled UEs |
| gNB antennas | 4 (configurable) |
| Rank | 1–2 layers per UE |
| Speed | 3–10 km/h (any positive value) |
| SNR | −20…30 dB, **per receive antenna / resource element** (channel is energy-normalized) |
| IoT | 0 / 10 / 20 dB. **0 dB = no neighbouring-cell interference**; 10 and 20 dB are INR = I/N |
| Waveform | NR PUSCH, DMRS **Type-1**, mapping type A |
| Numerology | 30 kHz SCS, **68 PRB = 816 used subcarriers**, **FFT 1024** (104 + 104 guards), 17 RBG (size 4) |
| Modulation | QPSK (MCS 4), 16QAM (MCS 14), 64QAM (MCS 20), all table 1 |
| Receivers | MR (matched filter), L-MMSE (DMRS-LS), **Ideal MMSE** (perfect CSI), ZF, IRC |
| CSI | DMRS-LS (NN / linear / linear+time-avg), LMMSE-CE (TDL prior), or perfect; Ideal MMSE always uses the true channel; `--perfect-csi` applies perfect CSI to every receiver |
| Metrics | Coded BER and BLER after LDPC TB decoding; working point at BER = 0.01 |

DMRS Type-1 provides 4 ports with `length=1` and 8 ports with `length=2`. The simulator selects length 2 automatically when the total number of layers exceeds 4 (for example 4 UEs × rank 2).

## Receivers and IoT

Serving users are always inside the MIMO channel matrix `H`. Thermal noise is white.

- **MR** — maximum-ratio / matched filter; ignores spatial interference.
- **ZF** — zero-forcing among co-scheduled layers.
- **L-MMSE** — MMSE with `S = N0 I` (other-cell IoT treated as extra white noise) and DMRS-LS channel estimates.
- **Ideal MMSE** — same L-MMSE combiner, but with **perfect CSI** (genie-aided lower bound).
- **IRC** — MMSE with `S = R_iot + N0 I`, where `R_iot` is the spatial covariance of neighbouring-cell interference. That is the only receiver that can reject *coloured* inter-cell interference.

IoT is generated as extra UEs through the same channel family, transmitting unit-power symbols on the used subcarriers. Their power is scaled so that the interference-to-noise ratio equals the requested IoT value. IRC can use the true interferer covariance (`--iot-cov perfect`, default) or a sample covariance of the received grid (`--iot-cov estimated`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sionna 2.x requires PyTorch. A GPU is recommended for 68-PRB campaigns; CPU is fine for `--quick` checks.

## Quick start

Smoke test (8 PRB, three SNR points, no IoT):

```bash
python -m nr_ul_sim --quick --channel cdl-c --modulation qpsk --receivers mr,lmmse,ideal_mmse,zf,irc
```

Example campaign matching the default numerology:

```bash
python -m nr_ul_sim \
  --channel cdl-c \
  --delay-spread-ns 300 \
  --num-ue 2 \
  --rank 1 \
  --num-rx-ant 4 \
  --speed-kmh 3 \
  --modulation qpsk \
  --snr-db -20:30:2 \
  --iot-db 0,10,20 \
  --receivers mr,lmmse,ideal_mmse,zf,irc \
  --batch-size 2 \
  --max-mc-iter 30 \
  --outdir results
```

JSON results and a BER plot are written under `results/`. The plot marks the interpolated SNR where BER crosses 0.01.

## Python API

```python
from nr_ul_sim import NRUplinkSimulator, SimConfig

cfg = SimConfig(
    channel="umi",
    num_ue=4,
    num_layers=1,
    num_rx_ant=4,
    speed_kmh=10,
    modulation="qam16",
    snr_db=list(range(-20, 31, 4)),
    iot_db=[0, 10, 20],
    receivers=("mr", "lmmse", "ideal_mmse", "zf", "irc"),
)
sim = NRUplinkSimulator(cfg)
campaign = sim.run()
print(campaign["iot"]["10.0"]["irc"]["working_point_db"])
```

## Dataset: channel → working point

`nr_ul_sim/dataset.py` builds a supervised dataset for learning the map **channel tensor → working point**. Each scenario draws random link parameters, **freezes one channel realization**, and searches the SNR where the coded BER of the chosen receiver(s) crosses `target_ber` for every channel estimator and modulation. The default sweep is **IRC** × {perfect, LS-NN, LS-linear, LS-lin+time-avg, LMMSE-CE} × {QPSK, 16QAM, 64QAM}. Freezing is what makes the label well posed: only data, noise and interferer symbols are redrawn between Monte-Carlo iterations, so the working point belongs to the stored tensor rather than to a channel ensemble (`--channel-mode ensemble` restores the usual per-slot redraw). Scalar features stored with each tensor include RMS, numerical/effective rank of `H`, and rank / condition / dominant-fraction of `R_uu`.

```bash
python -m nr_ul_sim.dataset --num-samples 500 --outdir dataset/run1 --num-prb 8
```

or edit the knobs at the top of `make_dataset.py` and run `python make_dataset.py`.

Randomized per scenario: channel family, RMS delay spread (log-uniform, CDL only), 1–4 UEs × rank 1–2, speed, IoT level (`--p-no-iot` of the scenarios get none), and the number of interferers. Everything else — carrier, numerology, PRB count, antenna count — stays fixed so the tensors share one shape.

### Working-point search

Instead of a full SNR sweep, the crossing is **bracketed** on a `--coarse-step` grid and then **bisected** down to `--refine-db`, reusing one BER measurement for all receivers and warm-starting from the previous result. A scenario costs ~15 SNR points instead of 26. Receivers that never reach the target within `[--snr-min, --snr-max]` are labelled `not_reached` (`NaN` in the packed arrays), those already below the range `below_range`.

Every SNR point is measured with a random stream seeded from `(scenario seed, SNR)`, so the label does not depend on which points the search happened to visit — a resumed or resharded run reproduces the same numbers.

### Output

| File | Content |
| --- | --- |
| `meta.jsonl` | One JSON record per (scenario, modulation): all parameters, per-receiver working point and status, the full BER/BLER curve that was measured, and channel features |
| `channels/<id>/H.npy` | Serving channel tensor `[rx, streams, symbols, subcarriers]` complex64 |
| `channels/<id>/H_int.npy` | Interferer channel, same layout (only if IoT > 0) |
| `channels/<id>/R_iot.npy` | Normalized interference spatial covariance `[rx, rx]` |
| `channels/<id>/meta.json` | Simulation knobs for that drop: IoT, channel family, UE/rank/speed, plus per-modulation working points |
| `meta.jsonl` / `dataset.csv` | One row per (scenario, modulation): all parameters, labels, paths to the npy files |
| `dataset.npz` | Same records stacked and zero-padded, for batch training |

```python
import json
import numpy as np
from nr_ul_sim.dataset import load_dataset

H = np.load("dataset/run1/channels/000000/H.npy")          # [rx, streams, sym, sc]
info = json.loads(open("dataset/run1/channels/000000/meta.json").read())
info["iot_db"], info["channel"], info["modulations"]["qpsk"]["working_point_db"]

d = load_dataset("dataset/run1")   # stacked view of the same files
d["H"]            # complex64 [N, rx ant, streams, symbols, subcarriers]
d["iot_db"]
d["channel_code"], d["modulation_code"]
d["wp_lmmse"]     # NaN where BER = 0.01 was never reached
```

The stored tensor is the true channel, subsampled to `--num-symbol-bins` OFDM symbols × `--num-freq-bins` subcarriers spread evenly over the used band (not the DMRS estimate — receivers still estimate their own CSI during the search). Streams are ordered UE-major, `num_layers_total` of them are valid and the rest are zero padding.

Runs are resumable and shardable: scenario `i` depends only on `(--seed, i)`, so `--start-index` splits the work across processes and `--resume` appends to an existing directory. `--pack-only` rebuilds `dataset.npz` / `dataset.csv` from `meta.jsonl`.

### Analysis

```bash
python analyze_dataset.py dataset/run1
```

Prints label coverage per receiver, the physical gaps that validate the run (16QAM − QPSK, L-MMSE − ideal MMSE, IRC gain per IoT bin) and a ridge regression on the scalar features alone — the bar a model trained on the raw tensors has to beat. Writes `analysis.png` next to the dataset.

## SNR definition

The serving channel is normalized to unit average energy per resource element. Noise variance is

```text
N0 = 10^(-SNR_dB / 10)
```

so `SNR_dB` is the **per-antenna, per-RE** SNR of the serving link, independent of MCS. Other-cell IoT is set relative to this `N0`.

## Notes

- Simulations run in the **frequency domain** (one tap per subcarrier). Time-domain CP/ISI modelling is not included.
- UMi/UMa delay spread is a random variable of the 3GPP drop; `--delay-spread-ns` applies to CDL-B/C.
- 68 PRB × 14 symbols is a full NR 25 MHz / 30 kHz allocation. Monte-Carlo at this bandwidth is computationally heavy; start with `--quick` or `--num-prb 16` while debugging.
- Rank-2 with 4 UEs uses DMRS Type-1 length-2 (8 ports). Rank-1 with up to 4 UEs uses length-1.

## Tests

```bash
python -m pytest tests
```
