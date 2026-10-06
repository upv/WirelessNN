# Tutorial

A guided tour of the simulator, from the first run to a multi-hour GPU campaign.
Every command is run from the repository root. Steps 1–6 work on a laptop CPU.

Contents:
[1 Install](#1-install-and-check) ·
[2 First simulation](#2-first-simulation) ·
[3 Inside one slot](#3-inside-one-slot) ·
[4 Configuration](#4-configuration) ·
[5 Receivers and interference](#5-receivers-and-interference) ·
[6 Channel estimators](#6-channel-estimators) ·
[7 Reading the results](#7-reading-the-results) ·
[8 Working-point dataset](#8-working-point-dataset) ·
[9 Random-channel campaign](#9-random-channel-campaign) ·
[10 Full campaigns on a GPU](#10-full-campaigns-on-a-gpu) ·
[11 Verifying the simulator](#11-verifying-the-simulator) ·
[12 Extending](#12-extending-the-simulator) ·
[13 Troubleshooting](#13-troubleshooting)

---

## 1. Install and check

```bash
scripts/setup_env.sh
```

creates `.venv`, installs the package with its analysis and test dependencies and
finishes with `scripts/check_install.py`, which prints the library versions, the
device (GPU name and memory, or CPU), whether the trained channel-estimator models
are present, and runs a three-point simulation. Run the check again at any time:

```bash
.venv/bin/python scripts/check_install.py
```

The commands below write `python`; either activate the environment
(`source .venv/bin/activate`) or use `.venv/bin/python`.

The simulator picks the GPU automatically when CUDA works and falls back to the CPU
otherwise; the device is printed at start-up and stored in every result file.

## 2. First simulation

```bash
python examples/01_quickstart.py
```

The file is a list of knobs followed by three calls — build a `SimConfig`, run
`NRUplinkSimulator(cfg).run()`, save the result. It sweeps the SNR for IRC with five
channel estimators on a small 8-PRB allocation and prints one line per SNR point:

```text
=== CDL-C | QAM16 | 1 UE x rank 1 | IoT=0.0 dB | CSI=perfect,ls_nn,ls_lin,... ===
  SNR   -5.0 dB | perfect BER=1.6e-01 BLER=1.0e+00 | ls_nn BER=... | [0.4 s]
  ...
  perfect          working point BER=0.01: 3.12 dB      BLER=0.1: 3.40 dB
```

The **working point** is the SNR where the coded BER crosses 0.01 (and, separately,
where the BLER crosses 0.1), interpolated between the simulated SNR points. It is the
one number that summarises a curve, and the quantity every study in this repository
compares.

The same run from the command line, without editing a file:

```bash
python -m nr_ul_sim --channel cdl-c --modulation qam16 --num-prb 8 \
    --snr-db -20:30:5 --iot-db 0,10 --receivers irc \
    --estimators perfect,ls_nn,ls_lin,ls_lin_time_avg,lmmse_ce --max-mc-iter 5
```

`python -m nr_ul_sim --help` documents every option, `python -m nr_ul_sim --list`
prints the available channels, modulations, receivers and channel estimators with a
one-line description each, and `--quick` shrinks any command to a few-second smoke run.

## 3. Inside one slot

```bash
python examples/02_one_slot_step_by_step.py
```

runs a single batch of slots through the chain and prints what every block produces:

| Step | Call | Output |
| --- | --- | --- |
| 1 Transmit | `sim.transmit(batch)` | bits `[batch, UE, TB bits]`, grid `x [batch, UE, UE ant, symbol, FFT bin]` |
| 2 Channel | inside `sim.generate_slot` | `h [batch, 1, RX ant, UE, UE ant, symbol, FFT bin]`, unit mean energy |
| 3 Interference + noise | inside `sim.generate_slot` | `y [batch, 1, RX ant, symbol, FFT bin]` |
| 4 Channel estimation | `sim.estimate_csi(slot)` | `slot["csi"][name] = (h_hat, err_var)` on the used subcarriers |
| 5 IRC covariance | `irc_interference_covariance(...)` | `R_iot [batch, RX ant, RX ant]` |
| 6 Detection | `sim.detect(slot, receiver, estimator)` | decoded bits, same shape as the transmitted ones |

`sim.run()` is a loop around these calls: for every IoT value and SNR point,
`measure_point` repeats steps 1–6 until every curve has collected
`num_target_bit_errors` bit errors **and** `num_target_block_errors` block errors, or
`max_mc_iter` batches have been simulated. All receivers and estimators see the same
slots, so differences between curves are not Monte-Carlo noise between runs.

Two definitions to keep in mind:

- **SNR** is per receive antenna and per resource element, for *one* UE: the channel
  is normalised to unit energy and `N0 = 10^(-SNR/10)`. It does not depend on the MCS,
  the number of antennas or the number of co-scheduled UEs.
- **IoT** (interference over thermal) is the *total* other-cell interference power per
  receive antenna relative to `N0`, whatever the number of interferers. `0` means no
  interference at all, not 0 dB.

## 4. Configuration

Everything is set through one dataclass, `SimConfig` (`nr_ul_sim/parameters.py`); the
command-line options map onto its fields one to one. The most used ones:

| Field | CLI | Default | Meaning |
| --- | --- | --- | --- |
| `channel` | `--channel` | `cdl-c` | `cdl-b`, `cdl-c`, `cdl-d`, `umi`, `uma` |
| `delay_spread_ns` | `--delay-spread-ns` | per model | CDL RMS delay spread; UMi/UMa draw their own |
| `num_ue` | `--num-ue` | 1 | co-scheduled UEs, 1–4 |
| `num_layers` | `--rank` | 1 | layers per UE, 1–2 |
| `num_rx_ant` | `--num-rx-ant` | 4 | gNB antennas (1, 2, 4, 8) |
| `speed_kmh` | `--speed-kmh` | 3 | UE speed |
| `modulation` | `--modulation` | `qpsk` | `qpsk`, `qam16`, `qam64` with a preset MCS; override with `mcs_index` |
| `snr_db` | `--snr-db` | −20…30 step 2 | list, or `start:stop:step` on the command line |
| `iot_db` | `--iot-db` | `[0]` (CLI: 0,10,20) | interference levels, one sweep each |
| `num_interferers` | `--num-interferers` | 2 | interfering UEs sharing the IoT power |
| `receivers` | `--receivers` | all six | `mr`, `zf`, `lmmse`, `ideal_mmse`, `irc`, `irc_real` |
| `channel_estimators` | `--estimators` | `ls_lin` | see [section 6](#6-channel-estimators) |
| `iot_cov` | `--iot-cov` | `perfect` | how IRC obtains the interference covariance |
| `num_prb` / `fft_size` | `--num-prb` / `--fft-size` | 68 / 1024 | allocation and FFT size |
| `batch_size` | `--batch-size` | 4 (CLI: 2) | slots per Monte-Carlo iteration |
| `max_mc_iter` | `--max-mc-iter` | 50 (CLI: 20) | iteration cap per SNR point |
| `seed` | `--seed` | 42 | everything random derives from it |

`cfg.summary()` returns the resolved configuration (MCS, DMRS length, guard carriers,
…) and is stored inside every result file, so a result always documents how it was
produced. Invalid combinations are rejected when the config is created, with a message
saying what is allowed.

## 5. Receivers and interference

```bash
python examples/03_receivers_vs_interference.py
```

compares the five classic receivers at IoT 0 and 10 dB with two co-scheduled UEs.

| Receiver | Combiner | Notes |
| --- | --- | --- |
| `mr` | matched filter | ignores every kind of interference |
| `zf` | zero forcing | removes the co-scheduled layers, amplifies noise |
| `lmmse` | MMSE with `S = N0·I` | other-cell interference is treated as white noise |
| `ideal_mmse` | same, with the true channel | genie bound for `lmmse` |
| `irc` | MMSE with `S = R_iot + N0·I` | the only one that rejects *coloured* interference; `R_iot` from `iot_cov` |
| `irc_real` | the same combiner | `R_iot` always measured on the received DMRS (soft-window CSI of its own, per-band OAS shrinkage); never sees the true channels, even with `--perfect-csi` |

Without other-cell interference IRC and L-MMSE are the same receiver. With it, IRC
gains as long as the interference occupies fewer spatial directions than there are
spare antennas. How IRC gets `R_iot` is chosen with `iot_cov`:

| `iot_cov` | Source | Realistic? |
| --- | --- | --- |
| `perfect` | true interferer channels | genie |
| `residual` | covariance of `y − Ĥp` on the DMRS symbols, minus `N0·I` | yes |
| `incm_oas` | the same residual per 2 PRB, shrunk towards a scaled identity (EqDeepRx) | yes, frequency selective |
| `estimated` | covariance of the whole received grid minus `N0·I` | contains the serving signal; kept for comparison |

`irc_real` ignores `iot_cov`: it is the `incm_oas` path with its own channel estimate,
so `--receivers lmmse,irc,irc_real` compares the genie, the chosen method and the
fully measured receiver on the same slots.

## 6. Channel estimators

```bash
python examples/04_channel_estimators.py
```

measures the channel-estimation NMSE and the working-point loss against perfect CSI.

| Name | Method |
| --- | --- |
| `perfect` | true channel (genie) |
| `ls_nn`, `ls_lin`, `ls_lin_time_avg` | DMRS least squares, nearest / linear / linear + time-average interpolation |
| `lmmse_ce` | LMMSE with a TDL prior matched to the scenario — a genie prior on CDL |
| `lmmse_exp` | LMMSE with an exponential-PDP prior — what a real receiver can assume |
| `ls_hard_window`, `ls_soft_window` | LS denoised in the delay domain: rectangular window, or per-tap Wiener weights. The soft window takes its disturbance floor from the taps outside the window (`--ce-soft-noise-mode outside`, default: follows interference) or from the thermal noise only (`thermal`, the pre-2026-10-06 behaviour) |
| `ls_fir` | LS + a static 17-tap frequency FIR (EqDeepRx baseline) |
| `denoise_nn` | EqDeepRx DenoiseNN — trained |
| `lmmse_data`, `lmmse_data_1d` | LMMSE with a covariance measured on training channels — trained |
| `a_mmse`, `ra_a_mmse` | attention-learned fixed linear filters (A-MMSE) and the rank-6 variant — trained |

The five *trained* estimators load their weights from `models/paper_ce`
(`--ce-model-dir` to point elsewhere). They were trained on UMa by
`scripts/training/train_paper_ce.py`; UMi and CDL are outside the training
distribution. `scripts/training/eval_paper_ce.py` plots NMSE versus SNR for all of them.

With several estimators in one run, `perfect` is simply one more entry: the loss of an
estimator is its working point minus the `perfect` one.

## 7. Reading the results

`sim.run()` returns, and the CLI saves as `results/<name>_<timestamp>.json`:

```python
campaign["config"]                       # cfg.summary()
campaign["device"], campaign["duration_s"]
campaign["iot"]["10.0"]["irc"]           # one curve per IoT value and result key
    ["snr_db"], ["ber"], ["bler"]        # lists over the SNR points
    ["stats"]                            # bit / block error counts per point
    ["working_point_db"]                 # SNR at BER = target_ber, or None
    ["working_point_bler_db"]            # SNR at BLER = target_bler, or None
```

The **result key** depends on what was swept:

| Swept | Key | Example |
| --- | --- | --- |
| several receivers, one estimator | receiver name | `"irc"` |
| one receiver, several estimators | estimator name | `"ls_lin"` |
| several of both | `receiver_estimator` | `"irc_ls_lin"` |

Next to the JSON the CLI writes three figures: BER and BLER curves with the working
points marked, and bar charts of the BER and BLER working points per IoT value.
A working point of `None` means the curve never reached the target inside the SNR range
— widen `snr_db`, or the receiver has an error floor.

## 8. Working-point dataset

```bash
python examples/05_working_point_dataset.py        # small demo, edit the knobs on top
python -m nr_ul_sim.dataset --num-samples 500 --outdir dataset/run1
```

builds a supervised dataset *channel tensor → working point*. Every scenario draws
random link parameters, **freezes one channel realisation** and searches the working
point by bracketing and bisection (about 15 SNR points instead of a full sweep), for
every receiver, estimator and modulation. Output in `--outdir`:

| File | Content |
| --- | --- |
| `meta.jsonl`, `dataset.csv` | one row per (scenario, modulation): parameters, labels, BER curve, features |
| `channels/<id>/H.npy`, `H_int.npy`, `R_iot.npy` | stored tensors of the scenario |
| `dataset.npz` | everything stacked and padded; read it with `nr_ul_sim.dataset.load_dataset` |

Runs are resumable (`--resume`) and shardable (`--start-index`):
`scripts/dataset/run_irc_ce.sh` launches 20 shards and
`scripts/dataset/merge_shards.py` merges them. Analysis:
`scripts/dataset/analyze_dataset.py` (coverage, physical sanity gaps, ridge baseline)
and `scripts/dataset/analyze_ce.py` (channel-estimation loss study).

## 9. Random-channel campaign

The dataset of section 8 randomises a handful of scalar parameters. The random-channel
campaign goes further: every configuration draws its own cluster angles, delay spread,
K-factor, UE geometry and speed (`nr_ul_sim/random_channels.py`), runs IRC with every
channel estimator on shared slots and finds each working point with an adaptive,
censoring-aware search (`nr_ul_sim/wp_search.py`).

```bash
python scripts/campaigns/run_random_campaign.py --out ~/reports/my_campaign \
    --per-model 1000 --workers 3
python scripts/campaigns/summarize_random_campaign.py ~/reports/my_campaign   # CSV tables, any time
python scripts/campaigns/report_random_campaign.py ~/reports/my_campaign      # REPORT.md + figures
```

The campaign is restartable: run the same command again and finished configurations
are skipped, failed ones retried. `--limit 5 --per-model 10` is a five-minute rehearsal.

## 10. Full campaigns on a GPU

| Script | What it runs |
| --- | --- |
| `scripts/campaigns/run_campaigns.sh` | full 68-PRB matrix: channel × modulation × receiver × IoT, plus the estimator study |
| `scripts/campaigns/run_paper_ce_campaign.sh` | the trained estimators at link level |
| `scripts/campaigns/summarize_campaigns.py` | tables and figures from the JSONs of the two above |

All are configured through environment variables, e.g.
`OUT=~/reports/x/campaigns WORKERS=3 scripts/campaigns/run_campaigns.sh`; finished
jobs are skipped on restart and each job logs to `$OUT/logs`.

Sizing rules of thumb:

- One SNR point with 8 slots of 68 PRB takes about a second on an RTX-class GPU.
- GPU memory is dominated by the LMMSE estimators: a 2-UE job with them needs about
  21 GB at 68 PRB and 1 GB at 12 PRB. Reduce `--batch-size` or the number of parallel
  `WORKERS` before reducing the PRB count.
- Raise `--max-mc-iter` and the error targets for smooth curves; lower them to iterate.

## 11. Verifying the simulator

```bash
python -m pytest tests                                    # about a minute on a CPU
python scripts/physics_checks.py --outdir physics_out     # figures + JSON report
```

`tests/test_physics.py` checks every block on its own at 68 PRB — transport-block size
against 38.214, DMRS layout and power, channel energy, delay spread, Jakes time
correlation, frequency and spatial correlation, noise and INR calibration, the
covariance estimators and receiver identities such as IRC ≡ L-MMSE without
interference. The other test files cover the estimators, the working-point searches,
the dataset builder, the CLI, metrics and plots.

Results are reproducible: with the same `seed`, device and library versions a run
returns the same numbers. Results differ between CPU and GPU and between GPU models,
because the random streams and floating-point kernels differ — compare statistics,
not individual error counts, across machines.

## 12. Extending the simulator

**A channel estimator** is a callable `(y, no) -> (h_hat, err_var)`:

1. implement it (subclass Sionna's `PUSCHLSChannelEstimator` like
   `nr_ul_sim/windowed_ce.py` does, to reuse the DMRS despreading);
2. add its name to `CHANNEL_ESTIMATORS` in `nr_ul_sim/parameters.py` (and to
   `ESTIMATOR_INFO` for `--list`);
3. construct it in `build_estimator` in `nr_ul_sim/channel_estimation.py`;
4. give it a colour and label in `RECEIVER_STYLE` in `nr_ul_sim/plotting.py`.

**A receiver** needs a name in `RECEIVERS` (and `RECEIVER_INFO`), an equaliser in
`EQUALIZER_BY_NAME` (`nr_ul_sim/receivers.py`) — a Sionna equaliser name or a callable
like `ExtraCovarianceLMMSE` — and a plot style. A receiver that needs per-slot side
information gets a branch in `NRUplinkSimulator.detect`, as `irc` and `irc_real` do.

**A channel model**: return a Sionna `ChannelModel` from
`NRUplinkSimulator._make_model`, or subclass the simulator and override it, as
`RandomChannelSimulator` (`nr_ul_sim/random_campaign.py`) does.

[ARCHITECTURE.md](ARCHITECTURE.md) has the module map and the tensor conventions.

## 13. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ModuleNotFoundError: nr_ul_sim` | run from the repository root, or `pip install -e .` once |
| `MCS table … needs LDPC repetition` | the code rate is too low for that TB size in Sionna; raise `--mcs-index` or lower `--num-prb` |
| `Total layers … exceed DMRS Type-1 ports` | more than 8 layers in total (`num_ue × rank`) |
| CUDA out of memory | lower `--batch-size`, run fewer workers, or drop `lmmse_ce` / `lmmse_exp` from the estimators |
| `FileNotFoundError … models/paper_ce/…pt` | a trained estimator was requested without its weights; run `scripts/training/train_paper_ce.py` or pass `--ce-model-dir` |
| working point is `None` / "not reached" | the curve never crossed the target: widen `--snr-db`, or the receiver has an error floor |
| simulation runs on the CPU on a GPU machine | `scripts/check_install.py` shows what PyTorch sees; usually a CPU-only PyTorch wheel |
| `-20:30:2` read as an option | write `--snr-db=-20:30:2` in scripts (the CLI itself accepts both forms) |

Known limitations of the models are listed under *Known issues* in
[CHANGELOG.md](../CHANGELOG.md).
