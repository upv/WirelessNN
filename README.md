# WirelessNN — 5G NR PUSCH uplink simulator

Link-level uplink simulator for 5G NR PUSCH, built on [NVIDIA Sionna](https://nvlabs.github.io/sionna/) PHY (v2, PyTorch). It sweeps **per-antenna SNR**, compares **MR / L-MMSE / Ideal MMSE / ZF / IRC**, injects **other-cell IoT**, and reports the **working point** (SNR where BER = 0.01).

## Start here

```bash
scripts/setup_env.sh                                  # .venv, dependencies, installation check
source .venv/bin/activate
python examples/01_quickstart.py                      # first SNR sweep, edit the knobs on top
python examples/02_one_slot_step_by_step.py           # what happens inside one slot
python -m nr_ul_sim --list                            # everything that can be simulated
```

| Where | What |
| --- | --- |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | guided tour: first run → receivers → channel estimators → datasets → GPU campaigns |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | module map, data flow of one slot, tensor shapes, conventions |
| `examples/` | five numbered starter files, each runs on a CPU in minutes |
| `nr_ul_sim/` | the simulator package |
| `scripts/` | environment setup, physics report, campaigns, dataset tools, training |
| `tests/` | `python -m pytest tests` |
| [CHANGELOG.md](CHANGELOG.md) | results of the campaigns so far and known issues |

## What is modelled

| Item | Value |
| --- | --- |
| Channel | CDL-B, CDL-C, CDL-D, 3GPP UMi, 3GPP UMa |
| RMS delay spread | Configurable for CDL (defaults: CDL-B 100 ns, CDL-C 300 ns, CDL-D 100 ns). UMi/UMa draw DS from TR 38.901 |
| Users | 1–4 co-scheduled UEs |
| gNB antennas | 4 (configurable) |
| Rank | 1–2 layers per UE |
| Speed | 3–10 km/h (any positive value) |
| SNR | −20…30 dB, **per receive antenna / resource element** (channel is energy-normalized) |
| IoT | 0 / 10 / 20 dB. **0 dB = no neighbouring-cell interference**; 10 and 20 dB are INR = I/N |
| Waveform | NR PUSCH, DMRS **Type-1**, mapping type A |
| Numerology | 30 kHz SCS, **68 PRB = 816 used subcarriers**, **FFT 1024** (104 + 104 guards), 17 RBG (size 4) |
| Modulation | QPSK (MCS 5, R≈0.37), 16QAM (MCS 14, R≈0.54), 64QAM (MCS 20, R≈0.55), all table 1. MCS 4 (R=0.30) would need LDPC BG1 repetition on a 68-PRB TB, which Sionna does not implement |
| Receivers | MR (matched filter), L-MMSE (DMRS-LS), **Ideal MMSE** (perfect CSI), ZF, IRC |
| CSI | DMRS-LS (NN / linear / linear+time-avg), LMMSE-CE (TDL prior), **LS + hard tap window**, **LS + soft (Wiener) tap window**, trained estimators (EqDeepRx DenoiseNN, data-covariance LMMSE, A-MMSE), or perfect; Ideal MMSE always uses the true channel; `--perfect-csi` applies perfect CSI to every receiver |
| Metrics | Coded BER and BLER after LDPC TB decoding; working points at BER = 0.01 and at BLER = 0.1 |

DMRS Type-1 provides 4 ports with `length=1` and 8 ports with `length=2`. The simulator selects length 2 automatically when the total number of layers exceeds 4 (for example 4 UEs × rank 2).

## Receivers and IoT

Serving users are always inside the MIMO channel matrix `H`. Thermal noise is white.

- **MR** — maximum-ratio / matched filter; ignores spatial interference.
- **ZF** — zero-forcing among co-scheduled layers.
- **L-MMSE** — MMSE with `S = N0 I` (other-cell IoT treated as extra white noise) and DMRS-LS channel estimates.
- **Ideal MMSE** — same L-MMSE combiner, but with **perfect CSI** (genie-aided lower bound).
- **IRC** — MMSE with `S = R_iot + N0 I`, where `R_iot` is the spatial covariance of neighbouring-cell interference. That is the only receiver that can reject *coloured* inter-cell interference.

IoT is generated as extra UEs through the same channel family, transmitting unit-power symbols on the used subcarriers. Their sum is scaled so that the **total** interference-to-noise ratio per receive antenna equals the requested IoT value, independent of the number of interferers and their antennas. IRC can use the true interferer covariance (`--iot-cov perfect`, default), the covariance of the DMRS residual `y − Ĥp` minus `N0 I` (`--iot-cov residual`, what a real receiver can measure), or the sample covariance of the whole received grid minus `N0 I` (`--iot-cov estimated`, contains the serving users' signal too).

## Channel estimation

| Name | Method |
| --- | --- |
| `perfect` | True channel (genie) |
| `ls_nn`, `ls_lin`, `ls_lin_time_avg` | Sionna PUSCH LS with CDM/OCC despreading, nearest / linear / linear + time-average interpolation |
| `lmmse_ce` | Sionna PUSCH LMMSE with a TDL time/frequency prior built from the scenario delay spread and speed. The TDL-B/C taps are the CDL-B/C cluster delays, so on CDL this is a **genie prior** (it knows the PDP); it collapses when the delay spread is 20 % off and is meaningless for UMi/UMa |
| `lmmse_exp` | Same LMMSE with a smooth exponential-PDP frequency prior (RMS delay spread of the scenario for CDL, 1 µs for UMi/UMa, `--ce-lmmse-prior-ds-ns` to override) — the robust prior a real receiver would use |
| `ls_hard_window` | LS at the DMRS, `M`-point IDFT of the 204 block estimates per DMRS symbol to the delay domain, taps outside `[-ce_window_neg_us, +ce_window_pos_us]` set to zero, `N`-point DFT back onto all 816 subcarriers, linear interpolation over OFDM symbols |
| `ls_soft_window` | Same, but each tap is weighted by the Wiener gain `max(P_k − a·σ²_tap, 0) / P_k` with the tap power `P_k` averaged over receive antennas and DMRS symbols (`--ce-soft-threshold a`, `--ce-soft-within-window` to combine with the hard window) |
| `ls_fir` | LS at the DMRS + a static 17-tap frequency FIR (EqDeepRx baseline, arXiv:2602.11834) |
| `denoise_nn` | EqDeepRx DenoiseNN on the despread LS estimates — trained |
| `lmmse_data`, `lmmse_data_1d` | 2D / 1D LMMSE with a covariance estimated from training channels — trained |
| `a_mmse`, `ra_a_mmse` | A-MMSE (arXiv:2506.00452): attention-learned fixed linear filters per 4-PRB block, and its rank-6 variant — trained |

The trained estimators load `models/paper_ce` (produced by `scripts/training/train_paper_ce.py` on UMa; UMi and CDL are outside the training distribution).

Both windowed estimators report an error variance that includes the channel energy removed by the window, so the LLR scaling stays calibrated. The rectangular 24.5 MHz band leaks every path as `1/(πk)` over the taps, which bounds the hard window near −24 dB NMSE with the default 1 µs negative window; the soft window adapts to the SNR and wins below ~20 dB.

## Install

```bash
scripts/setup_env.sh
```

creates `.venv`, installs the package in editable mode with the analysis and test
extras, and runs `scripts/check_install.py` (library versions, GPU, trained models,
a smoke simulation). By hand, simulator only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sionna 2.x requires PyTorch. A GPU is recommended for 68-PRB campaigns; a CPU is fine for the examples and `--quick` checks.

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
  --estimators ls_lin \
  --batch-size 8 \
  --max-mc-iter 10 \
  --outdir results
```

JSON results, BER/BLER plots and working-point bar charts are written under `results/`. The plots mark the interpolated SNR where BER crosses 0.01 and where BLER crosses 0.1. `python -m nr_ul_sim --help` documents every option. The full 68-PRB test matrix (every channel × modulation × receiver × IoT, plus the channel-estimator study) is `scripts/campaigns/run_campaigns.sh`; `scripts/campaigns/summarize_campaigns.py` turns its output into tables, and `scripts/physics_checks.py` measures every block of the simulator on its own (PDP and delay spread, time/frequency/spatial correlation, SNR and INR calibration, CE NMSE versus SNR).

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
python -m nr_ul_sim.dataset --num-samples 500 --outdir dataset/run1 --num-prb 68
```

or edit the knobs at the top of `examples/05_working_point_dataset.py` and run it (a small demo by default).

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

Runs are resumable and shardable: scenario `i` depends only on `(--seed, i)`, so `--start-index` splits the work across processes and `--resume` appends to an existing directory. `--pack-only` rebuilds `dataset.npz` / `dataset.csv` from `meta.jsonl`. `scripts/dataset/run_irc_ce.sh` launches a sharded run and `scripts/dataset/merge_shards.py` merges it.

### Analysis

```bash
python scripts/dataset/analyze_dataset.py dataset/run1
```

Prints label coverage per receiver, the physical gaps that validate the run (16QAM − QPSK, L-MMSE − ideal MMSE, IRC gain per IoT bin) and a ridge regression on the scalar features alone — the bar a model trained on the raw tensors has to beat. Writes `analysis.png` next to the dataset.

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/setup_env.sh`, `scripts/check_install.py` | set up and verify an installation |
| `scripts/physics_checks.py` | block-by-block physics report (figures + JSON) |
| `scripts/campaigns/run_campaigns.sh` | full 68-PRB receiver and estimator matrix |
| `scripts/campaigns/run_paper_ce_campaign.sh` | link-level campaign of the trained estimators |
| `scripts/campaigns/summarize_campaigns.py` | tables and figures from the campaign JSONs |
| `scripts/campaigns/run_random_campaign.py` | random-channel working-point campaign (restartable, multi-worker) |
| `scripts/campaigns/summarize_random_campaign.py`, `report_random_campaign.py` | its CSV tables and report |
| `scripts/campaigns/report_paper_ce.py` | report on the trained estimators |
| `scripts/dataset/run_irc_ce.sh`, `watch_run.sh`, `merge_shards.py` | sharded dataset generation |
| `scripts/dataset/analyze_dataset.py`, `analyze_ce.py` | dataset sanity checks and channel-estimation loss study |
| `scripts/training/train_paper_ce.py`, `eval_paper_ce.py` | train and evaluate the EqDeepRx / A-MMSE estimators (`models/paper_ce`) |
| `scripts/training/train_ridge.py`, `train_wp_models.py`, `train_wp_models2.py` | working-point regressors on the dataset |

Every script starts with a docstring or comment giving its command line.

## SNR definition

The serving channel is normalized to unit average energy per resource element. Noise variance is

```text
N0 = 10^(-SNR_dB / 10)
```

so `SNR_dB` is the **per-antenna, per-RE** SNR of one serving UE, independent of MCS. A UE radiates unit power in total: with rank 2 each layer carries half of it (`--tx-power-norm per_layer` restores unit power per layer). With several co-scheduled UEs every UE arrives at that SNR. Other-cell IoT is the **total** interference power per antenna relative to this `N0`.

## Notes

- Simulations run in the **frequency domain** (one tap per subcarrier). Time-domain CP/ISI modelling is not included.
- UMi/UMa delay spread is a random variable of the 3GPP drop; `--delay-spread-ns` applies to CDL-B/C.
- 68 PRB × 14 symbols is a full NR 25 MHz / 30 kHz allocation and is the default everywhere (simulator, dataset, tests). One SNR point with 8 slots of 68 PRB takes about a second on an RTX-class GPU; start with `--quick` while debugging.
- The gNB panel uses the 38.901 element pattern; it weights the CDL clusters by their angle of arrival, so the *effective* RMS delay spread at the receiver is smaller than the nominal CDL value (CDL-B 100 → ≈55 ns, CDL-C 300 → ≈70 ns). `scripts/physics_checks.py` reports both.
- DMRS Type-1 OCC despreading averages two pilots 2 subcarriers apart; it attenuates a tap at delay `d` by `cos(2πd/N)` and, with two ports in one CDM group, leaks `(h₁(k) − h₁(k+2))/2` between the ports on a frequency-selective channel. Every DMRS estimator shares this.
- Rank-2 with 4 UEs uses DMRS Type-1 length-2 (8 ports). Rank-1 with up to 4 UEs uses length-1.

## Tests

```bash
python -m pytest tests
```

`tests/test_physics.py` checks every block on its own at 68 PRB (TB size against 38.214, DMRS layout and power, channel energy, delay spread, Jakes time correlation, frequency and spatial correlation, noise and INR calibration, covariance estimators, receiver identities such as IRC ≡ L-MMSE without interference), `tests/test_windowed_ce.py` the delay-domain estimators, the rest the CLI, metrics, plots and the dataset builder.

## Measured-covariance IRC and interference-aware soft window

`irc_real` always estimates the interference-plus-noise covariance from the
received DMRS residual, using a dedicated `ls_soft_window` estimate and per-band
OAS shrinkage. It never uses true serving/interferer channels for covariance,
even with `--perfect-csi` (which still makes detector CSI genie-aided).
`irc` retains the covariance selected by `--iot-cov`; its default is perfect.
Compare both receivers on the same slots:

```bash
python -m nr_ul_sim --channel cdl-c --receivers lmmse,irc,irc_real \
    --estimators ls_soft_window --iot-db 0,10,20 --snr-db=-4:20:2
```

`--incm-band-sc 24` controls the covariance frequency bands. The realistic
receiver uses known thermal N0, ideal synchronization and the simulated DMRS
layout; residual fitting can suppress observed interference, while channel
estimation errors can inflate it. It is a practical covariance baseline,
not a complete hardware receiver model or a guaranteed ordering against IRC.

Soft window now estimates the tap disturbance floor by a median of the power
outside the configured delay window, corrected for averaging across antennas
and DMRS symbols, and bounded below by the thermal LS variance. The same floor
is used for Wiener gains and reported error variance. It does not read IoT or
true channels. Long channel tails and correlated disturbances can bias this
estimate; when fewer than eight outside taps exist it falls back to thermal.
`--ce-soft-noise-mode thermal` reproduces the previous algorithm for comparisons.

Reproduce the paired 68-PRB, fixed-64-slot comparison (CDL-C / UMa, IoT 0/10/20):

```bash
python scripts/campaigns/benchmark_irc_real.py --outdir results/irc_real_comparison
```

This writes raw BER/BLER counts and `comparison.png`. The coarse SNR grid is
for regression checks; refine near the target before quoting working-point gains.
