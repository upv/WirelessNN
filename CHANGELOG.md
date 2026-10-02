# Changelog

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
