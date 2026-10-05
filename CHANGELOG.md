# Changelog

## 2026-10-05 — project layout, starter files, tutorial (no change in results)

Refactoring only: the simulator produces bit-identical numbers. Checked with the test
suite (106 tests) and a fixed-seed snapshot of every entry point before and after —
five link-level sweeps covering all receivers, all 14 channel estimators and all
interference-covariance methods, the CLI, the dataset builder (frozen and ensemble) and
four random-channel configurations — compared byte for byte on the CPU.

- Root scripts moved: `play.py` → `examples/01_quickstart.py`, `make_dataset.py` →
  `examples/05_working_point_dataset.py` (now a small demo writing to `dataset/example`),
  campaign runners and reports → `scripts/campaigns/`, dataset tools →
  `scripts/dataset/`, training and evaluation → `scripts/training/`,
  `physics_checks.py` → `scripts/`, the notebook → `notebooks/`. Scripts find the
  repository root themselves and the shell scripts `cd` to it, so all of them run from
  any directory. The package `nr_ul_sim`, its module names, `python -m nr_ul_sim` and
  `python -m nr_ul_sim.dataset` are unchanged.
- New: `scripts/setup_env.sh` and `scripts/check_install.py` (environment, GPU, trained
  models, smoke run); `examples/02`–`04` (one slot step by step, receivers under
  interference, channel estimators); `docs/TUTORIAL.md`, `docs/ARCHITECTURE.md`.
- CLI: options grouped and documented in `--help`, `--list` prints every channel,
  modulation, receiver, estimator and covariance method with a description. Defaults
  are unchanged.
- `parameters.py`: `SimConfig` fields documented and grouped; `CHANNEL_INFO`,
  `RECEIVER_INFO`, `ESTIMATOR_INFO`, `IOT_COV_INFO`. Docstrings for every core module.
- `NRUplinkSimulator._draw_topology` hook; `RandomChannelSimulator` overrides it instead
  of carrying a copy of `generate_slot`.
- `pyproject.toml`: optional dependency groups `analysis` and `dev`.

### Found during the review (not fixed: fixing changes dataset labels)

- `dataset.FrozenChannelSimulator.generate_slot` was not updated with the 2026-09-30
  power fixes of the main simulator. It transmits with unit power per *layer* (rank 2:
  twice the power of `NRUplinkSimulator`, while the perfect-CSI channel is scaled for
  unit power per UE), and it scales the interference per interferer *stream*, so the
  total INR is `iot_db + 10 log10(num_interferers x num_ue_ant)`. Working-point datasets
  built with `channel_mode="frozen"` (the default) therefore use a different SNR and IoT
  definition from the sweeps and the random-channel campaign for rank 2 and for more
  than one interferer stream. `channel_mode="ensemble"` is not affected.

## 2026-10-05 — random-channel campaign with A-MMSE (5000 configurations, 16 PRB)

Report: `~/reports/random_ch_ammse_2026-10-05/report/REPORT.md` on the new GPU server
(RTX PRO 6000 Blackwell, outside the repository). Same plan and seeds as the 2026-09-30
campaign, all 14 channel estimators, 3 workers, 9 h 16 min, no runtime errors.

- `run_random_campaign.py --estimators` (default: all of `CHANNEL_ESTIMATORS`); the GPU
  name goes into `campaign_config.json`.
- `report_random_campaign.py`: estimators taken from the campaign, numbers in the key
  conclusions computed from the data, an A-MMSE section (loss with censoring, A-MMSE
  against every other estimator on the same configurations, breakdowns, scatter against
  LMMSE with data covariance), `--compare` with an earlier campaign of the same plan,
  `<campaign>/notes.md` as hand-written conclusions.
- `summarize_random_campaign.py`: estimators taken from `CHANNEL_ESTIMATORS`.

### Results

Loss of the working point against perfect CSI, IRC, BER = 1e-2. "With censoring" counts a
working point that was not found as infinite loss.

| estimator | WP found | median, found only | median with censoring | p90 with censoring |
|---|---|---|---|---|
| LMMSE, TDL prior | 84 % | 0.63 dB | 0.83 dB | not found |
| DenoiseNN | 90 % | 0.68 | 0.77 | not found |
| LMMSE, exp prior | 92 % | 1.02 | 1.13 | 4.26 |
| A-MMSE | 93 % | 1.19 | 1.29 | 5.98 |
| LMMSE, data covariance | 96 % | 1.32 | 1.37 | 3.95 |
| RA-A-MMSE r=6 | 86 % | 1.33 | 1.56 | not found |
| LS-linear | 93 % | 2.59 | 2.75 | 5.57 |

- A-MMSE matches its closed-form limit, LMMSE with data covariance: within ±0.25 dB in
  50 % of configurations, median difference −0.03 dB, as the paper says.
- Its tail is heavier: no working point in 7.1 % of configurations against 3.9 %, and a
  median loss of 15.4 dB for delay spread above 1000 ns. The filter is learned on UMa;
  the filter bank needs a delay-spread dimension or channels with larger spread.
- A-MMSE is 1.2 dB better than LS-linear and 0.55 dB better than the soft window, on par
  with LMMSE exp prior, 0.3 dB behind DenoiseNN and the matched TDL prior.
- RA-A-MMSE with rank 6 is not enough at high SNR: 14 % without a working point, 5.6 dB
  median loss at 2 UE × rank 2.
- Common estimators reproduce the 2026-09-30 campaign: median |ΔWP| 0.05–0.08 dB, no bias.

## 2026-10-05 — channel estimators from EqDeepRx and A-MMSE

Papers: EqDeepRx (arXiv:2602.11834) and A-MMSE (arXiv:2506.00452). Report with all tables:
`~/reports/paper_ce/REPORT.md` (outside the repository).

- `nr_ul_sim/paper_ce.py`: new channel estimators `ls_fir` (LS + static 17-tap frequency FIR,
  EqDeepRx baseline), `denoise_nn` (EqDeepRx DenoiseNN), `lmmse_data` / `lmmse_data_1d`
  (2D / 1D LMMSE with a covariance estimated from training channels), `a_mmse` (attention
  transformer that learns one fixed linear filter per 4-PRB block, bank over SNR and DMRS
  comb) and `ra_a_mmse` (rank-6 variant). Registered in `channel_estimation.py`, `cli.py`,
  `parameters.py`, `plotting.py`, `receivers.py`.
- `nr_ul_sim/interference.py`: `--iot-cov incm_oas`, the EqDeepRx interference-plus-noise
  covariance from the DMRS residual over 2 PRB with shrinkage towards a scaled identity.
- `train_paper_ce.py` + `models/paper_ce/`: trained on UMa, 0–35 m/s, 1–2 UE × rank 1–2,
  68 PRB; UMi, CDL-B and CDL-C are out of the training distribution.
- `eval_paper_ce.py`: NMSE vs SNR for every estimator, 6 scenarios. Defaults are 256 slots per
  point, and every SNR point uses the same channel realisations. With the previous
  16 slots and new channels per point, UMa/UMi curves were off by up to 15 dB at
  high SNR.
- `run_paper_ce_campaign.sh`, `report_paper_ce.py`, `tests/test_paper_ce.py` (103 tests pass).

### Results (48 link-level campaigns, IRC, 68 PRB)

Median loss of the working point against perfect CSI, BER = 1e-2: DenoiseNN 0.63 dB,
LMMSE exp. prior 0.64, LMMSE data covariance 0.99, A-MMSE 1.05, RA-A-MMSE 1.07,
LS + soft window 1.16, LS + FIR 1.31, LS-linear 2.84. DenoiseNN carries over to CDL
outside its training set (CDL-B 0.18, CDL-C 0.34 dB; LMMSE exp. prior 0.15 / 0.57) but
misses the working point in 9/72 cases. A-MMSE converges to the best fixed linear filter,
as the paper says, and does not beat `lmmse_data`. RA-A-MMSE with rank 6 loses
2–4.5 dB NMSE above 10 dB SNR on UMa validation.

## 2026-10-02 — random-channel campaign report

- `report_random_campaign.py`: report generator for `run_random_campaign.py` output
  (setup, verification, censoring, channel-estimation loss tables and figures, known issues).
- `models/wp_bench`, `models/wp_bench2`: results of the working-point regression benchmarks
  on `dataset/irc_ce10k` (metrics, reports, figures, out-of-fold predictions).

### Random-channel campaign, 5000 configurations (16 PRB, IRC, BER = 1e-2)

Results in `/root/reports/random_ch_2026-09-30/report/REPORT.md` (outside the repository).

- All 5000 configurations finished, no runtime errors. Every working point is reproduced
  exactly from the stored BER curves; a dense re-simulation of 16 configurations gives a
  label accuracy of ±0.3 dB (bias −0.14 dB). Perfect-CSI working points follow the
  theoretical post-IRC SINR (correlation 0.93–0.97; gap to Shannon 1.2 / 2.6 / 3.4 dB for
  QPSK / 16QAM / 64QAM).
- Median channel-estimation loss vs perfect CSI: LMMSE (TDL prior) 0.6 dB, LMMSE
  (exponential prior) 1.0, LS + soft window 1.9, LS + hard window 1.9, LS-lin + time
  average 2.3, LS-linear 2.6, LS-NN 3.6 dB. Share of configurations where the working point
  is found: hard window 94 %, LS-linear 93 %, soft window 93 %, LMMSE exp 92 %, LS-NN 86 %,
  LMMSE TDL 85 %, LS-lin + time average 80 %.

### Known issues (not fixed yet)

- `random_campaign.run_config` gives `lmmse_ce` the delay spread of UE 0 only; Sionna's
  LMMSE estimator uses one covariance for all streams, so with 2 UEs the prior is wrong
  for UE 1 (working point not reached in 41 % of configurations where UE 1 has more than
  twice the delay spread of UE 0, 6 % with 1 UE). Use the maximum over the UEs.
- `ls_soft_window` sets its Wiener threshold from N0; other-cell interference raises the
  per-tap noise floor, so the loss grows from 0.9 dB (no IoT) to 2.9 dB (IoT 15–20 dB),
  against 1.6 → 2.4 dB for the hard window. Estimate the per-tap noise from the taps
  outside the window instead.
- `sinr_ref_eff_db` uses the wideband interference covariance and is optimistic under
  strong frequency-selective interference (working point offset +2.6 dB at IoT 15–20 dB);
  a working-point model needs `iot_db` as a separate input.
- UMi/UMa interferers are drawn inside the serving sector, not in neighbouring cells.

## 2026-09-30 — 68 PRB, windowed / robust channel estimation, physics fixes (77e67ad)

- 68 PRB by default (simulator, dataset, tests, scripts).
- Interference scaled to the total INR per antenna; per-UE transmit power (rank 2 splits
  it over the layers); IRC sample covariance no longer conjugated; `--iot-cov residual`.
- QPSK preset MCS 5 (MCS 4 needs LDPC BG1 repetition on a 68-PRB TB).
- `ls_hard_window`, `ls_soft_window` (delay-domain windowing of the LS estimate),
  `lmmse_exp` (LMMSE with an exponential-PDP prior).
- BLER working point, block-error stop criterion, negative `--snr-db` ranges.
- Per-block physics tests, `physics_checks.py`, `run_campaigns.sh`, `summarize_campaigns.py`.
- Random-channel campaign: `random_channels`, `random_campaign`, `wp_search`,
  `run_random_campaign.py`, `summarize_random_campaign.py`.
