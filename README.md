# WirelessNN — 5G NR PUSCH uplink simulator

Link-level uplink simulator for 5G NR PUSCH, built on [NVIDIA Sionna](https://nvlabs.github.io/sionna/) PHY (v2, PyTorch). It sweeps **per-antenna SNR**, compares **MR / L-MMSE / ZF / IRC**, injects **other-cell IoT**, and reports the **working point** (SNR where BER = 0.01).

## What is modelled

| Item | Value |
| --- | --- |
| Channel | CDL-B, CDL-C, 3GPP UMi, 3GPP UMa |
| RMS delay spread | Configurable for CDL (defaults: CDL-B 100 ns, CDL-C 300 ns). UMi/UMa draw DS from TR 38.901 |
| Users | 1–4 co-scheduled UEs |
| gNB antennas | 4 (configurable) |
| Rank | 1–2 layers per UE |
| Speed | 3–10 km/h (any positive value) |
| SNR | −20…20 dB, **per receive antenna / resource element** (channel is energy-normalized) |
| IoT | 0 / 10 / 20 dB. **0 dB = no neighbouring-cell interference**; 10 and 20 dB are INR = I/N |
| Waveform | NR PUSCH, DMRS **Type-1**, mapping type A |
| Numerology | 30 kHz SCS, **68 PRB = 816 used subcarriers**, **FFT 1024** (104 + 104 guards), 17 RBG (size 4) |
| Modulation | QPSK (MCS table 1, index 4) and 16QAM (MCS table 1, index 14) |
| Receivers | MR (matched filter), L-MMSE, ZF, IRC |
| CSI | DMRS least-squares by default; `--perfect-csi` optional |
| Metrics | Coded BER and BLER after LDPC TB decoding; working point at BER = 0.01 |

DMRS Type-1 provides 4 ports with `length=1` and 8 ports with `length=2`. The simulator selects length 2 automatically when the total number of layers exceeds 4 (for example 4 UEs × rank 2).

## Receivers and IoT

Serving users are always inside the MIMO channel matrix `H`. Thermal noise is white.

- **MR** — maximum-ratio / matched filter; ignores spatial interference.
- **ZF** — zero-forcing among co-scheduled layers.
- **L-MMSE** — MMSE with `S = N0 I` (other-cell IoT treated as extra white noise).
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
python -m nr_ul_sim --quick --channel cdl-c --modulation qpsk --receivers mr,lmmse,zf,irc
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
  --snr-db -20:20:2 \
  --iot-db 0,10,20 \
  --receivers mr,lmmse,zf,irc \
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
    snr_db=list(range(-20, 21, 4)),
    iot_db=[0, 10, 20],
    receivers=("mr", "lmmse", "zf", "irc"),
)
sim = NRUplinkSimulator(cfg)
campaign = sim.run()
print(campaign["iot"]["10.0"]["irc"]["working_point_db"])
```

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
python -m pytest tests/test_metrics.py
```
