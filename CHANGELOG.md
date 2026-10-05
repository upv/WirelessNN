# Changelog

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
