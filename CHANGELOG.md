# Changelog

What changed in the code and the documentation, newest first. Campaign results are
collected in [docs/RESULTS.md](docs/RESULTS.md); the current model limitations are in
[docs/MODEL.md](docs/MODEL.md#known-limitations).

## 2026-11-18 — documentation for publication

- README rewritten around a short quick start and a results summary; the detailed
  reference moved to `docs/MODEL.md` (simulation model), `docs/DATASET.md`
  (working-point dataset) and `docs/RESULTS.md` (campaign results), so each topic is
  described once. The tutorial links to them instead of repeating their tables.
- `experiments/ce_denoising` documentation translated into English; its pilot and
  20k reports merged into one `REPORT.md`. The two campaign report generators
  (`report_paper_ce.py`, `report_random_campaign.py`) now write English reports.

## 2026-11-17 — Channelformer plugin and 4x2 MIMO campaign

- `experiments/channelformer`: Channelformer (Luan & Thompson, IEEE TWC 2023) as an
  `--ce-plugin` estimator, training-data generation, training, campaign and summary
  scripts, trained weights, and the results of a 45-scenario 4x2 rank-2 campaign
  against all 14 built-in estimators ([results](docs/RESULTS.md#1-4x2-mimo-14-estimators-and-channelformer)).

## 2026-10-27 — channel estimators from plugin files

- `--ce-plugin path.py` (repeatable, `SimConfig.ce_plugins`) imports a file that calls
  `register_estimator(name, factory, description)`; the estimator can then be used
  like a built-in one, without changing the package. Built-in names cannot be
  replaced. `tests/test_estimator_plugins.py`.

## 2026-10-25 — continuous integration

- The test suite and `scripts/check_install.py` run on GitHub Actions for every push
  to `main` and every pull request (CPU-only PyTorch).

## 2026-10-17 — dataset version 2 and per-UE LMMSE prior

- `DatasetConfig.dataset_version` (default 2): frozen-channel slots of the
  working-point dataset now use the simulator's power convention (unit power per UE,
  total INR per antenna). Version 1 keeps the legacy convention for extending old
  datasets; records store their version and resume / pack refuse to mix versions.
- `random_campaign.run_config` sets the `lmmse_ce` / `lmmse_exp` delay-spread prior to
  the maximum over the UEs (`prior_delay_spread_ns`) instead of UE 0's, as already done
  for speed. Sionna's LMMSE estimator shares one covariance across streams, so UE 0's
  value over-smoothed UE 1 whenever UE 1 had the longer channel. 1-UE results are
  unchanged.

## 2026-10-14 — dataset for denoising LS channel estimates

- `experiments/ce_denoising`: replayable 16-PRB dataset of LS estimate → true channel
  pairs with received grid, noise, interference and transmitted bits; PyTorch loader,
  channel-MSE and bit-level BCE losses, linear baselines (ridge, LMMSE), audit and
  tests ([report](experiments/ce_denoising/REPORT.md)).

## 2026-10-07 — complex BCE loss

- `nr_ul_sim/losses.py`: `complex_bce` and `complex_bce_batched`, exact bit-wise BCE
  between estimated and transmitted complex symbols for QPSK, 16QAM and 64QAM (mixed
  modulations in one batch), using separable per-axis LLRs.

## 2026-10-06 — measured-covariance IRC and empirical soft-window floor

- New receiver `irc_real`: IRC whose interference-plus-noise covariance is measured on
  the received DMRS only (own soft-window CSI, per-band OAS shrinkage), cached per
  slot and independent of the detector CSI and `--iot-cov`.
- `ls_soft_window` estimates the per-tap disturbance floor from the taps outside the
  delay window; `--ce-soft-noise-mode thermal` keeps the previous algorithm. With
  interference this gains 5–12 dB NMSE ([results](docs/RESULTS.md#4-interference-aware-soft-window)).
  `ls_soft_window` results from before this change reproduce only with
  `--ce-soft-noise-mode thermal`.
- `SimConfig.receivers` and `--receivers` default to all receivers, now including
  `irc_real`; the dataset builder and the random-channel campaign keep their explicit
  lists. With `thermal` mode and `irc_real` left out, a fixed-seed snapshot of every
  entry point is identical to the previous commit.

## 2026-10-05 — project layout, starter files, tutorial (no change in results)

Refactoring only; a fixed-seed snapshot of every entry point is bit-identical before
and after.

- Root scripts moved: `play.py` → `examples/01_quickstart.py`, `make_dataset.py` →
  `examples/05_working_point_dataset.py`, campaign runners and reports →
  `scripts/campaigns/`, dataset tools → `scripts/dataset/`, training and evaluation →
  `scripts/training/`, `physics_checks.py` → `scripts/`, the notebook → `notebooks/`.
  All scripts run from any directory. `python -m nr_ul_sim` and
  `python -m nr_ul_sim.dataset` are unchanged.
- New: `scripts/setup_env.sh`, `scripts/check_install.py`, `examples/02`–`04`,
  `docs/TUTORIAL.md`, `docs/ARCHITECTURE.md`.
- CLI options grouped and documented in `--help`; `--list` prints every channel,
  modulation, receiver, estimator and covariance method with a description.
- `NRUplinkSimulator._draw_topology` hook; `RandomChannelSimulator` overrides it.
- `pyproject.toml`: optional dependency groups `analysis` and `dev`.

## 2026-10-05 — channel estimators from EqDeepRx and A-MMSE

- `nr_ul_sim/paper_ce.py`: `ls_fir` (EqDeepRx static FIR baseline), `denoise_nn`
  (EqDeepRx DenoiseNN), `lmmse_data` / `lmmse_data_1d` (LMMSE with a covariance
  estimated from training channels), `a_mmse` (attention transformer that learns one
  fixed linear filter per 4-PRB block) and `ra_a_mmse` (rank-6 variant).
- `--iot-cov incm_oas`: EqDeepRx interference-plus-noise covariance from the DMRS
  residual over 2 PRB with shrinkage towards a scaled identity.
- `scripts/training/train_paper_ce.py` + `models/paper_ce/` (trained on UMa, 0–35 m/s,
  1–2 UE × rank 1–2, 68 PRB); `eval_paper_ce.py` (NMSE vs SNR, the same channel
  realisations at every SNR point); `run_paper_ce_campaign.sh`, `report_paper_ce.py`
  ([results](docs/RESULTS.md#2-trained-estimators-at-68-prb)).
- `run_random_campaign.py --estimators` (default: all); `report_random_campaign.py`
  adds a censored-loss section, `--compare` with an earlier campaign and
  `<campaign>/notes.md` for hand-written conclusions.

## 2026-10-02 — random-channel campaign report

- `report_random_campaign.py`: report generator for `run_random_campaign.py` output
  (setup, verification, censoring, channel-estimation loss tables and figures)
  ([results](docs/RESULTS.md#3-random-channel-campaign-5000-configurations)).
- `models/wp_bench`, `models/wp_bench2`: working-point regression benchmarks on
  `dataset/irc_ce10k`.

## 2026-09-30 — 68 PRB, windowed and robust channel estimation, physics fixes

- 68 PRB by default (simulator, dataset, tests, scripts).
- Interference scaled to the total INR per antenna; per-UE transmit power (rank 2
  splits it over the layers); IRC sample covariance no longer conjugated;
  `--iot-cov residual`.
- QPSK preset MCS 5 (MCS 4 needs LDPC BG1 repetition on a 68-PRB TB).
- `ls_hard_window`, `ls_soft_window` (delay-domain windowing of the LS estimate),
  `lmmse_exp` (LMMSE with an exponential-PDP prior).
- BLER working point, block-error stop criterion, negative `--snr-db` ranges.
- Per-block physics tests, `physics_checks.py`, `run_campaigns.sh`,
  `summarize_campaigns.py`.
- Random-channel campaign: `random_channels`, `random_campaign`, `wp_search`,
  `run_random_campaign.py`, `summarize_random_campaign.py`.

## 2026-09 — first version

- PUSCH link-level simulator on Sionna 2 with CDL/UMi/UMa channels, MR / ZF / L-MMSE /
  IRC receivers and DMRS-LS / LMMSE channel estimation; working-point dataset builder
  and a ridge baseline (8 PRB).
