# WirelessNN: 5G NR uplink receiver and channel-estimation simulator

[![tests](https://github.com/upv/WirelessNN/actions/workflows/tests.yml/badge.svg)](https://github.com/upv/WirelessNN/actions/workflows/tests.yml)

A link-level simulator for **5G NR PUSCH receivers, channel estimation and
interference rejection**, built on [NVIDIA Sionna](https://nvlabs.github.io/sionna/)
PHY 2 (PyTorch). It turns an algorithm choice into coded BER/BLER and into the
**working point**, the SNR needed to reach BER = 0.01 or BLER = 0.1, so classical and
learned estimators and receivers can be compared on the same slots.

- 5 channel models (CDL-B/C/D, 3GPP UMi/UMa), up to 4 co-scheduled UEs with rank 1–2,
  1, 2, 4 or 8 gNB antennas, other-cell interference.
- 6 receivers, from matched filter to interference rejection combining (IRC) with a
  measured covariance.
- 14 built-in channel estimators (LS, LMMSE, delay-domain windows, EqDeepRx, A-MMSE)
  and a plugin hook for your own, with [Channelformer](experiments/channelformer/README.md)
  and five more recent neural estimators ([ce_zoo](experiments/ce_zoo/README.md)) as
  worked examples, tested in simulation and on over-the-air captures.
- Tools for large campaigns, datasets for learned estimators and working-point
  prediction, and a physics test suite.

## Results at a glance

**Over the air.** On a 5G stand with two commercial UEs and a 2-antenna gNB, the best
small neural channel estimators beat the gNB's own estimate by about 0.9 dB of fit SNR
(median per slot), Channelformer by 0.84 dB and a well-tuned LMMSE by 0.6 dB. The
42k-weight DenoiseNN matches the 300k-weight CHEA-XL, and both lead in simulation too.

| Estimator | Weights | Simulation NMSE, 4x2 [dB] | Simulation BLER, loss vs perfect CSI, 4x2 [dB] | Over the air, gain vs gNB estimate [dB] |
| --- | ---: | ---: | ---: | ---: |
| `zoo_chea_xl` (CHEA-XL) | 302k | -16.29 | 0.67 | +0.88 to +0.97 |
| `zoo_denoiserx` (EqDeepRx DenoiseNN) | 42k | **-16.60** | **0.65** | +0.87 to +0.97 |
| `channelformer` | 192k | -15.33 | 0.73 | +0.82 to +0.93 |
| `zoo_helena` (HELENA) | 133k | -15.16 | 0.71 | +0.76 to +0.85 |
| `zoo_swift` (SwiftChannel) | 9.6k | -14.56 | 0.79 | +0.71 to +0.78 |
| `zoo_chea` (CHEA) | 33k | -15.24 | 0.73 | +0.66 to +0.73 |
| `lmmse_exp` (LMMSE, exponential prior) | 0 | -13.07 | 1.57 | +0.58 to +0.78 |
| `ls_soft_window`, delay removed first | 0 | -12.25 | – | +0.35 to +0.76 |
| `ls_lin` (LS + linear interpolation) | 0 | -4.52 | – | -0.07 to +0.07 |

Simulation: validation NMSE on the 4x2 MIMO set (CDL-B/C/D, UMi, UMa, SNR -10 to 30 dB);
LMMSE there uses a 300 ns prior and estimates the noise level from the slice itself.
BLER: mean distance to perfect CSI of the SNR at 10 % BLER over 5 channels x 3 MCS at
30 km/h (quick sweep, about ±0.3 dB; the LMMSE row there is the simulator's `lmmse_exp`,
which knows the noise variance; – = not in that sweep;
[table](experiments/ce_zoo/README.md#results-bler-working-points)).
Over the air: models retrained for the stand's slot layout, range of the median gain
over six UE sets, 1 layer, uplink SNR 16 to 21 dB. Channelformer, LMMSE and the soft window
get the stand's mean delay (about -110 ns) removed first; the ce_zoo models were trained
with random timing offsets and handle it themselves. With one receive antenna the gains are
about half as large. Setup, method and the 1-RX numbers:
[experiments/ce_zoo](experiments/ce_zoo/README.md#over-the-air-replay).

![Mean SNR loss against perfect CSI, 4x2 MIMO campaign](docs/img/ce_loss_4x2.png)

**Link level.** In a 4x2 MIMO campaign (68 PRB, rank 2, 45 scenarios) the Channelformer
estimator loses 0.17 dB on average against perfect CSI on CDL channels, and the best
classical estimators lose about 0.5–0.7 dB. On UMi/UMa every estimator loses 2–4 dB,
most of it at 64QAM. A quick BLER sweep of the ce_zoo models on the same link (30 km/h,
1 dB grid) puts all of them within 0.15 dB of each other: mean loss against perfect CSI
0.65 dB for DenoiseNN and CHEA-XL, 0.73 dB for Channelformer, 1.57 dB for the simulator's
`lmmse_exp` ([table](experiments/ce_zoo/README.md#results-bler-working-points)).
Across 5000 random channel configurations the matched-prior LMMSE has the lowest median
loss (0.6 dB) but misses the target in 16 % of them. These and other results, with
setups and reproduction commands, are in [docs/RESULTS.md](docs/RESULTS.md).

## Quick start

Python 3.10 or newer. A CPU is enough for everything in this section; a GPU is
recommended for full 68-PRB campaigns.

```bash
git clone https://github.com/upv/WirelessNN.git
cd WirelessNN
scripts/setup_env.sh              # creates .venv, installs everything, runs an installation check
source .venv/bin/activate
python examples/01_quickstart.py  # first SNR sweep
```

`setup_env.sh` ends with `scripts/check_install.py`, which prints the library versions,
the device the simulator will use (GPU or CPU), whether the trained estimator models
are present, and runs a three-point smoke simulation (about 10 seconds on a CPU). On a GPU machine, if it reports
no GPU, install the PyTorch wheel that matches your CUDA driver into `.venv`
([selector](https://pytorch.org/get-started/locally/)) and run it again.

The quick-start example sweeps the SNR on a small 8-PRB allocation for the IRC
receiver with five channel estimators, at IoT 0 and 10 dB. It takes about 30 seconds
on a 4-core laptop CPU and prints one line per SNR point, then the working points:

```text
=== CDL-C | QAM16 | 1 UE x rank 1 | IoT=10.0 dB | CSI=perfect,ls_nn,ls_lin,ls_lin_time_avg,lmmse_ce ===
  SNR  -20.0 dB | perfect BER=4.813e-01 BLER=1.00e+00 | ls_nn BER=5.022e-01 BLER=1.00e+00 | ... | [1.2 s]
  ...
Working points (BER = 0.01)
  IoT = 0.0 dB
    perfect       5.08 dB
    ls_nn         5.16 dB
    ...
  IoT = 10.0 dB
    perfect       15.00 dB
    ...
```

The SNR grid is a coarse 5 dB here, so the working points are rough; refine
`SNR_DB` near the crossing for real comparisons. A JSON with every curve and the
BER/BLER and working-point figures land in `results/examples/`. Edit the knobs at the
top of the file (channel, modulation, UEs, receivers, estimators) and run it again.

The same kind of run from the command line:

```bash
python -m nr_ul_sim --list                       # every channel, receiver and estimator, with a description
python -m nr_ul_sim --quick --channel cdl-c --receivers lmmse,irc --estimators ls_lin,lmmse_exp
python -m nr_ul_sim --help                       # all options
```

`--quick` shrinks any command to an 8-PRB, few-second smoke run. Drop it and set
`--num-prb`, `--snr-db=-20:30:2`, `--batch-size` and `--max-mc-iter` for real runs.

## Next steps

| To | Read or run |
| --- | --- |
| understand one slot end to end | `python examples/02_one_slot_step_by_step.py` |
| compare receivers under interference | `python examples/03_receivers_vs_interference.py` |
| compare channel estimators (NMSE and working point) | `python examples/04_channel_estimators.py` |
| build a channel → working-point dataset | `python examples/05_working_point_dataset.py`, [docs/DATASET.md](docs/DATASET.md) |
| plug in your own channel estimator | [tutorial section 6](docs/TUTORIAL.md#6-channel-estimators), [experiments/channelformer](experiments/channelformer/README.md) |
| run full campaigns on a GPU | [tutorial sections 9–10](docs/TUTORIAL.md#9-random-channel-campaign) |
| check the physics | `python -m pytest tests`, `python scripts/physics_checks.py --outdir physics_out` |

## Documentation

| Document | Contents |
| --- | --- |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | guided tour: first run, configuration, receivers, estimators, results, datasets, campaigns, troubleshooting |
| [docs/MODEL.md](docs/MODEL.md) | what is simulated: scope, SNR and IoT definitions, receivers, estimators, known limitations |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | module map, data flow of one slot, tensor shapes, conventions |
| [docs/DATASET.md](docs/DATASET.md) | the channel → working-point dataset |
| [docs/RESULTS.md](docs/RESULTS.md) | campaign results |
| [CHANGELOG.md](CHANGELOG.md) | what changed and when |

## Repository layout

```text
nr_ul_sim/      the simulator package (python -m nr_ul_sim, python -m nr_ul_sim.dataset)
examples/       five numbered starter scripts, each runs on a CPU in minutes
scripts/        setup and installation check, physics report, campaigns, dataset tools, training
experiments/    self-contained studies: Channelformer 4x2 campaign, ce_zoo estimators, LS-denoising dataset
models/         trained estimator weights (paper_ce) and regression benchmarks
tests/          pytest suite (python -m pytest tests)
docs/           tutorial, model reference, architecture, dataset, results
dataset/        metadata of datasets built during development (tensors are not in Git)
```

## Python API

```python
from nr_ul_sim import NRUplinkSimulator, SimConfig

cfg = SimConfig(
    channel="umi",
    num_ue=2,
    num_layers=1,
    modulation="qam16",
    snr_db=list(range(-10, 31, 4)),
    iot_db=[0, 10],
    receivers=("lmmse", "irc"),
    channel_estimators=("ls_lin",),
    num_prb=8,
)
campaign = NRUplinkSimulator(cfg).run()
print(campaign["iot"]["10.0"]["irc"]["working_point_db"])
```

Every `SimConfig` field is documented in `nr_ul_sim/parameters.py`, and the
command-line options map onto them one to one.

## Scope

This is a frequency-domain simulator with ideal synchronisation, not a complete
hardware receiver. `irc` uses the perfect interference covariance by default; use
`irc_real` or `--iot-cov residual` for a measured one. Learned estimators in
`models/paper_ce` were trained on UMa only. See
[docs/MODEL.md](docs/MODEL.md#known-limitations) for the full list.

## Author

[Pavel Uydanov](https://github.com/upv) ·
[LinkedIn: project and R&D enquiries](https://www.linkedin.com/in/pavel-uydanov/)

Related work: [SVD Lab](https://github.com/upv/Pavel-s_SVD), a numerical algorithm and
GPU benchmarking laboratory.
