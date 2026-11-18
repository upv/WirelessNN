# Simulation model

What the simulator models, the conventions every result depends on, and the known
limitations. For a hands-on introduction read [TUTORIAL.md](TUTORIAL.md); for the code
structure read [ARCHITECTURE.md](ARCHITECTURE.md).

## Scope

| Item | Value |
| --- | --- |
| Channel | CDL-B, CDL-C, CDL-D, 3GPP UMi, 3GPP UMa (TR 38.901) |
| RMS delay spread | Configurable for CDL (defaults CDL-B 100 ns, CDL-C 300 ns, CDL-D 100 ns); UMi/UMa draw it from TR 38.901 |
| Users | 1–4 co-scheduled UEs, rank 1–2 each |
| Antennas | gNB 1, 2, 4 (default) or 8 with the 38.901 element pattern; UE 1, 2 or 4 ports |
| Speed | any positive value (3 km/h by default) |
| SNR | per receive antenna and resource element (channel energy-normalised), typically −20…30 dB |
| Interference (IoT) | other-cell interferers through the same channel family; `0` means none, otherwise the total INR in dB |
| Waveform | NR PUSCH, DMRS Type 1, mapping type A |
| Numerology | 30 kHz SCS, 14 symbols; 68 PRB = 816 used subcarriers in an FFT of 1024 (104 + 104 guards) by default, any PRB count configurable |
| Modulation | QPSK (MCS 5, R≈0.37), 16QAM (MCS 14, R≈0.54), 64QAM (MCS 20, R≈0.55), MCS table 1; `--mcs-index` overrides |
| Receivers | MR, ZF, L-MMSE, ideal MMSE (perfect CSI), IRC, measured-covariance IRC |
| Channel estimators | 14 built-in, plus your own through `--ce-plugin` |
| Metrics | coded BER and BLER after LDPC TB decoding; working points at BER = 0.01 and BLER = 0.1 |

MCS 4 (R = 0.30) is not offered for QPSK because on a 68-PRB TB it would need LDPC
BG1 repetition, which Sionna does not implement. DMRS Type 1 provides 4 ports with
`length=1` and 8 with `length=2`; the simulator selects length 2 automatically when
the total number of layers exceeds 4 (for example 4 UEs × rank 2).

## SNR and IoT

The serving channel is normalised to unit average energy per resource element and the
noise variance is

```text
N0 = 10^(-SNR_dB / 10)
```

so `SNR_dB` is the **per-antenna, per-RE** SNR of one serving UE, independent of the
MCS, the number of antennas and the number of co-scheduled UEs. A UE radiates unit
power in total: with rank 2 each layer carries half of it (`--tx-power-norm per_layer`
restores unit power per layer). With several co-scheduled UEs every UE arrives at this
SNR.

**IoT** (interference over thermal) is the **total** other-cell interference power per
receive antenna relative to `N0`, whatever the number of interferers and their
antennas. Interferers are extra UEs drawn from the same channel family, transmitting
unit-power Gaussian symbols on the used subcarriers; their sum is scaled to the
requested INR. IoT `0` disables interference rather than meaning 0 dB.

## Receivers

Serving users are always inside the MIMO channel matrix `H`; thermal noise is white.

| Receiver | Combiner | Notes |
| --- | --- | --- |
| `mr` | matched filter | ignores every kind of interference |
| `zf` | zero forcing | removes the co-scheduled layers, amplifies noise |
| `lmmse` | MMSE with `S = N0·I` | other-cell interference is treated as white noise |
| `ideal_mmse` | same, with the true channel | genie bound for `lmmse` |
| `irc` | MMSE with `S = R_iot + N0·I` | the only receiver that rejects *coloured* interference; `R_iot` from `--iot-cov` |
| `irc_real` | same combiner | `R_iot` always measured on the received DMRS; never sees the true channels, even with `--perfect-csi` |

Without other-cell interference IRC and L-MMSE are the same receiver. With it, IRC
gains as long as the interference occupies fewer spatial directions than there are
spare antennas. How `irc` obtains `R_iot`:

| `--iot-cov` | Source | Realistic? |
| --- | --- | --- |
| `perfect` (default) | true interferer channels | genie |
| `residual` | covariance of `y − Ĥp` on the DMRS symbols, minus `N0·I` | yes |
| `incm_oas` | the same residual per 2 PRB, shrunk towards a scaled identity (EqDeepRx) | yes, frequency selective |
| `estimated` | sample covariance of the whole received grid minus `N0·I` | contains the serving signal; kept for comparison |

`irc_real` ignores `--iot-cov`. It estimates the interference-plus-noise covariance
from the received DMRS residual, using its own `ls_soft_window` channel estimate and
per-band OAS shrinkage (`--incm-band-sc`, 24 subcarriers by default), with known
thermal `N0`, ideal synchronisation and the simulated DMRS layout. It is a practical
covariance baseline, not a complete hardware receiver, and is not guaranteed to rank
below genie IRC: residual fitting can suppress observed interference, while
channel-estimation errors can inflate it. `--receivers lmmse,irc,irc_real` compares
all three on the same slots; `scripts/campaigns/benchmark_irc_real.py` runs the paired
68-PRB comparison (CDL-C / UMa, IoT 0/10/20) and writes raw counts and a figure.

`--perfect-csi` gives every receiver the true channel; `ideal_mmse` always uses it.

## Channel estimators

| Name | Method |
| --- | --- |
| `perfect` | true channel (genie) |
| `ls_nn`, `ls_lin`, `ls_lin_time_avg` | Sionna PUSCH LS with CDM/OCC despreading; nearest / linear / linear + time-average interpolation |
| `lmmse_ce` | Sionna PUSCH LMMSE with a TDL time/frequency prior from the scenario delay spread and speed. The TDL-B/C taps are the CDL-B/C cluster delays, so on CDL this is a **genie prior**; it collapses when the delay spread is 20 % off and is meaningless for UMi/UMa |
| `lmmse_exp` | the same LMMSE with a smooth exponential-PDP prior (scenario RMS delay spread for CDL, 1 µs for UMi/UMa, `--ce-lmmse-prior-ds-ns` to override); the robust prior a real receiver would use |
| `ls_hard_window` | LS at the DMRS, IDFT of the 204 block estimates per DMRS symbol to the delay domain, taps outside `[-ce_window_neg_us, +ce_window_pos_us]` zeroed, DFT back onto all 816 subcarriers, linear interpolation over symbols |
| `ls_soft_window` | the same, with each tap weighted by the Wiener gain `max(P_k − a·σ²_tap, 0) / P_k`, tap power averaged over antennas and DMRS symbols (`--ce-soft-threshold a`, `--ce-soft-within-window` to combine with the hard window) |
| `ls_fir` | LS + a static 17-tap frequency FIR (EqDeepRx baseline, arXiv:2602.11834) |
| `denoise_nn` | EqDeepRx DenoiseNN on the despread LS estimates — trained |
| `lmmse_data`, `lmmse_data_1d` | 2D / 1D LMMSE with a covariance measured on training channels — trained |
| `a_mmse`, `ra_a_mmse` | A-MMSE (arXiv:2506.00452): attention-learned fixed linear filters per 4-PRB block, and its rank-6 variant — trained |

The five trained estimators load `models/paper_ce` (`--ce-model-dir` to point
elsewhere), produced by `scripts/training/train_paper_ce.py` on UMa; UMi and CDL are
outside their training distribution.

**Soft-window disturbance floor.** By default (`--ce-soft-noise-mode outside`) the
per-tap disturbance floor is the median power of the taps outside the configured delay
window, corrected for averaging across antennas and DMRS symbols and bounded below by
the thermal LS variance. It follows interference without reading IoT or the true
channels; long channel tails and correlated disturbances can bias it, and with fewer
than eight outside taps it falls back to thermal. `--ce-soft-noise-mode thermal` uses
the thermal noise only (the behaviour before 2026-10-06).

Both windowed estimators report an error variance that includes the channel energy
removed by the window, so the LLR scaling stays calibrated. The rectangular 24.5 MHz
band leaks every path as `1/(πk)` over the taps, which bounds the hard window near
−24 dB NMSE with the default 1 µs negative window; the soft window adapts to the SNR
and wins below about 20 dB.

**Your own estimator** plugs in through a Python file that registers it, without
changing the package; see [tutorial section 6](TUTORIAL.md#6-channel-estimators).
[Channelformer](../experiments/channelformer/README.md) is a complete example.

## Modelling notes

- Simulations run in the **frequency domain** (one tap per subcarrier) with ideal
  synchronisation. Time-domain CP/ISI, CFO and timing errors are not modelled.
- The gNB panel uses the 38.901 element pattern, which weights the CDL clusters by
  their angle of arrival, so the *effective* RMS delay spread at the receiver is
  smaller than the nominal CDL value (CDL-B 100 → ≈55 ns, CDL-C 300 → ≈70 ns).
  `scripts/physics_checks.py` reports both.
- DMRS Type 1 OCC despreading averages two pilots 2 subcarriers apart. It attenuates a
  tap at delay `d` by `cos(2πd/N)` and, with two ports in one CDM group, leaks
  `(h₁(k) − h₁(k+2))/2` between the ports on a frequency-selective channel. Every DMRS
  estimator shares this.
- One SNR point with 8 slots of 68 PRB takes about a second on an RTX-class GPU. GPU
  memory is dominated by the LMMSE estimators (a 2-UE job with them needs about 21 GB
  at 68 PRB).

## Known limitations

- **UMi/UMa interferers** are drawn as UEs inside the serving sector, not in
  neighbouring cells, so their angles of arrival and antenna gain resemble those of the
  served UEs. For IRC this is somewhat optimistic.
- **`irc` uses a perfect, wideband interference covariance by default.** Use
  `irc_real`, `--iot-cov residual` or `--iot-cov incm_oas` for a measured one.
- **`sinr_ref_eff_db`** (a feature of the random-channel campaign) uses the wideband
  interference covariance and is optimistic under strong frequency-selective
  interference (working point offset +2.6 dB at IoT 15–20 dB); a working-point model
  needs `iot_db` as a separate input.
- **Legacy working-point datasets.** Datasets built by `nr_ul_sim.dataset` before
  `dataset_version` existed (2026-10-25), including the ones tracked under `dataset/`,
  use the legacy frozen-channel power convention: unit power per *layer* and
  interference scaled per interferer *stream*. For rank 2 or more than one interferer
  stream their SNR and IoT differ from the sweeps. New datasets use the simulator's
  convention (`dataset_version = 2`); see [DATASET.md](DATASET.md).
- Results differ between CPU and GPU and between GPU models, because random streams
  and floating-point kernels differ. Compare statistics, not individual error counts,
  across machines.
