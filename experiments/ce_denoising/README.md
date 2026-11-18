# Dataset for denoising the LS channel estimate

This experiment builds **LS estimate → true channel** pairs for training channel
estimators. Next to each pair it stores the received grid, thermal noise,
interference and the transmitted QAM symbols, so channel MSE and bit-level losses
can both be computed. Band: **16 PRB** (192 used subcarriers, FFT 256), 14 OFDM
symbols, 30 kHz SCS, 3.5 GHz carrier. One target UE, rank 1, one TX antenna, four
RX ports. All signals are frequency-domain complex64.

Results of the linear baselines on 900 and 20 250 realisations are in
[REPORT.md](REPORT.md).

## Running

From the repository root, in the WirelessNN environment:

```bash
# 900 independent realisations (540 / 180 / 180); GPU is picked automatically, CPU works too.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m experiments.ce_denoising.generate \
  --outdir experiments/ce_denoising/data/pilot_16prb --counts 540 180 180

# Fit the linear filters on train, tune on validation, evaluate on test.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -u -m experiments.ce_denoising.analyze \
  --dataset experiments/ce_denoising/data/pilot_16prb --outdir experiments/ce_denoising/reports

# File, identity and label audit of a generated dataset; unit tests.
python -m experiments.ce_denoising.audit --dataset experiments/ce_denoising/data/pilot_16prb
python -m pytest experiments/ce_denoising/tests -q
```

Each split size must be a positive multiple of 45 (5 channels × 3 IoT × 3 SNR
bands). `--counts 45 45 45 --limit 1` writes a single example for a quick check.
`--limit` caps the number of records without changing the sampling plan.
`--resume` continues an existing dataset only if the recipe, the generator and
simulator sources and the library versions match; the SHA-256 of every record
already written is verified. Without `--resume` an existing directory is never
overwritten.

Each record is written to `.pending` and renamed atomically. After a crash,
`.pending` and `.writer.lock` must be inspected and removed by hand; parallel
writers into one directory are refused. Generation stops when less than 2 GiB of
disk is left. The generator does not daemonise; wrap long runs in `timeout`.

## Random parameters

Each split shuffles strata of five channels × three IoT levels × three SNR bands.
SNR is uniform inside its band, so the whole set covers **−20…10 dB uniformly**;
channel family and IoT have equal shares. Modulation is drawn from QPSK, 16QAM and
64QAM, and the number of configured interferers uniformly from **2…6**. All seeds
are fixed before generation, so adding noise augmentations later does not change
the channel split.

**IoT = 0 means no interference.** The channels of the 2…6 potential interferers
and their parameters are still stored, but their amplitude is zero and
`num_interferers_active = 0`. IoT = 10 / 20 dB means a total I/N of 10 / 100,
split equally in power among the interferers. SNR refers to the normalised
channel and one RE per receive antenna, not Eb/N0.

The target UE and every interferer get an independent speed of **3…30 km/h**,
direction of motion and multipath realisation:

- CDL-B/C/D: delay-spread scale log-uniform 20…1000 ns, angular scales 0.5…2,
  random mean angles and angle/delay jitter from `nr_ul_sim/random_channels.py`;
  CDL-D also gets a K-factor of 0…15 dB. These are perturbed CDL profiles, not the
  tabulated ones, and the DS scale is not the RMS DS measured after perturbation,
  K-factor and antenna patterns.
- UMa/UMi: random sector geometry, indoor/LoS state, correlated large-scale
  parameters, ray delays, powers, angles and XPR from the Sionna model. DS follows
  the joint UMa/UMi distribution. The 3…30 km/h range also applies to indoor UEs.
- All links of one record use the same channel family. Topologies, rays and speeds
  of the target and the interferers are independent; this is not a common
  multi-cell layout.

The target waveform is coded NR PUSCH with DMRS Type 1, MCS table 1: QPSK index 5,
16QAM index 14, 64QAM index 20. Interferers transmit independent complex Gaussian
OFDM symbols on the used subcarriers, as in the main simulator. This is spatially
coloured interference, **not pilot contamination by synchronous DMRS of
neighbouring PUSCH**; that needs a separate dataset with coded neighbouring UEs.

Channels vary within the slot with Doppler. Each link is normalised over the full
FFT grid, as `GenerateOFDMChannel` does; the actual power in the used band is
stored. Path loss and shadow fading are off and SNR is set separately. The
frequency-domain model has no ISI/ICI, synchronisation errors, CFO or delays
beyond the cyclic prefix; check very long channel tails with a time-domain model.

## What is stored

`data/.../config.json` holds the recipe, library versions, Git revision and the
SHA-256 of the sources. `index.jsonl` lists the train/validation/test split. Each
`records/<id>/` contains `sample.npz`, `parameters.npz` and `meta.json`; read them
with `np.load(..., allow_pickle=False)`.

Shapes use R = 4 RX ports, T = 14 symbols, F = 192 subcarriers, D = 2 DMRS
symbols, M = 48 CDM blocks, K = 2…6 interferers. Only singleton batch / RX-site /
UE / layer axes are dropped; scale and axis order are the same for all
modulations.

| Key in `sample.npz` | Shape | Purpose |
| --- | --- | --- |
| `h_ls` | R,T,F | Main input: Sionna PUSCH LS with CDM despreading and linear interpolation |
| `h_true` | R,T,F | Target for complex MSE, with the same normalisation as the transmitted signal |
| `ls_pilots` | R,D,M | LS after averaging the two pilots of each 4-subcarrier CDM block; compact input |
| `h_ls_clean`, `h_ls_noise`, `h_ls_interference` | R,T,F | Decomposition of the LS input into noiseless interpolated estimate, noise and interference |
| `ls_pilots_clean`, `ls_pilots_noise`, `ls_pilots_interference` | R,D,M | The same decomposition before interpolation |
| `y`, `y_target`, `noise`, `interference` | R,T,F | Received grid and its components |
| `interference_per_link`, `h_interferers` | K,R,T,F | Received contribution and normalised channel of each interferer |
| `x_interferers_unit`, `x_interferers_scaled` | K,T,F | Interferer symbols before / after the power factor |
| `x_tx`, `pilot_grid`, `pilot_reserved_mask`, `data_mask` | T,F | Transmitted grid, DMRS and masks; `data_mask` excludes whole DMRS symbols |
| `tx_data_symbols` | 2304 | Transmitted QAM symbols, row-major `x_tx[data_mask]` |
| `tx_coded_bits` | 2304,Qm | Bit labels of those symbols, after coding and scrambling |
| `tx_tb_bits` | TB size | Original transport block, for checking a decoder |
| `dmrs_symbols`, `pilot_block_centres_sc` | D; M | DMRS symbol indices and LS block centres, zero-based |
| `cir_target_a`, `cir_interferer_a` | Sionna CIR layout | Complex path coefficients before frequency normalisation |
| `cir_target_tau_s`, `cir_interferer_tau_s` | Sionna delay layout | Path delays in seconds |

CIR layout: `[batch, RX-site, RX-ant, TX, TX-ant, path, time]`; delays:
`[batch, RX-site, TX, path]`. Exact shapes and dtypes of every array are in
`meta.json`. To rebuild the normalised H from the CIR, call
`cir_to_ofdm_channel(..., normalize=True)` on the full FFT grid and strip the
guards; the CIR coefficients alone do not have unit energy.

`parameters.npz` stores the CDL cluster tables or the UMa/UMi ray and topology
arrays, plus the numeric Sionna model / scenario / LSP / ray parameters (speeds,
orientations, LoS, delays, powers, angles, XPR). Together with the seed and the CIR
this makes every realisation reproducible. `meta.json` holds the sampled and
resolved configuration, per-link parameters, N0, nominal interference power,
measured SNR/INR, scaling and checksums.

Before a record is written these identities are checked:

```text
y = h_true * x_tx + noise + interference
interference = sum_k(h_interferers[k] * x_interferers_scaled[k])
h_ls = h_ls_clean + h_ls_noise + h_ls_interference
```

`h_ls_clean` is the noiseless LS + interpolation result, not `h_true`; the
difference is the interpolation and CDM-averaging bias. Every `*_clean` array,
the separate noise / interference and the true channels are labels and
diagnostics: never feed them to a trained network as input.

## Training with it

```python
import torch
from experiments.ce_denoising.dataset import CEDataset
from experiments.ce_denoising.losses import channel_mse, symbol_bce

sample = CEDataset("experiments/ce_denoising/data/pilot_16prb",
                   split="train", include_bce=True)[0]
x = torch.view_as_real(sample["input"])      # [4, 14, 192, 2]; the network layout is up to you
# h_pred = model(...)                        # complex [4, 14, 192]
# loss_mse = channel_mse(h_pred, sample["target"])
# loss_bce = symbol_bce(h_pred, sample["y"], sample["x_tx"], sample["data_mask"],
#                       sample["modulation"], sample["disturbance_variance"])
```

`channel_mse` is mean(|Ĥ − H|²); averaging real and imaginary parts separately
would give half that number. NMSE in the report is sum(|Ĥ − H|²) / sum(|H|²) in dB.

BCE cannot be computed between complex Ĥ and H directly. `symbol_bce` runs a
differentiable SIMO combiner, exact QAM bit likelihoods, and BCE against the
transmitted **coded QAM bits** (not the TB bits). It uses the known nominal N0 + I0
and a spatially white Gaussian disturbance model, ignoring the interference
covariance and the uncertainty of Ĥ. It is a training surrogate, not coded BER/BLER
after LDPC. In the report BCE is averaged over the bits of a record, then over
records.

All methods get the same I + N power when BCE is evaluated. A trained network can
receive it as an auxiliary input or estimate it from the observation. With mixed
modulations, weight the per-modulation BCE by the number of bits; a plain mean of
the three groups overweights QPSK. MSE can be computed in one batch.

## Linear baselines and protocol

Covariances and regressions are fitted on train, regularisation is chosen by MSE on
validation, and test is used only for the final numbers. Compared on the same
realisations:

1. LS with linear interpolation; LS averaged over the two DMRS symbols.
2. A fixed DFT window of −1…+3 µs plus time interpolation.
3. `ridge_input_only`: one shared affine map from the 96 complex LS pilots to the
   2688 REs, the same for every antenna and condition. Uses only LS.
4. `lmmse_known_inr`: train-set covariance and cross-covariance of the clean
   channel, plus a diagonal pilot-error model `(N0 + I0) / 4` with a power factor
   tuned on validation. It is linear for given N0 / IoT but uses that extra known
   information, so it is not an LS-only method.
5. Zero estimate as a sanity baseline, noiseless LS as the interpolation floor,
   true H as the BCE reference.

None of the filters uses the spatial covariance across RX ports: one map is fitted
on the rows of all antennas. Antennas of one channel are correlated, so the 95 %
bootstrap intervals in the report resample **whole realisations**, not antennas or
REs. The mix of SNR / IoT conditions drives the average MSE; look at the per-SNR and
per-IoT breakdowns as well.

## What is in Git

Generator, loader, losses, analysis, tests, this README, the report, aggregated
CSVs and figures. `data/`, `runs/`, `artifacts/`, channel arrays, fitted filters,
per-record metrics and logs are ignored by the local `.gitignore`.

## API references

- [Sionna PUSCH LS](https://nvlabs.github.io/sionna/_modules/sionna/phy/nr/pusch_channel_estimation.html): CDM-aware LS and interpolation.
- [Sionna 2.1 CDL tutorial](https://nvlabs.github.io/sionna/v2.1.0/phy/tutorials/notebooks/MIMO_OFDM_Transmissions_over_CDL.html): CIR → normalised OFDM channel → estimation → equalisation.
- Project code: `nr_ul_sim/random_channels.py`, `pusch.py`, `channel_estimation.py`, `losses.py`.
