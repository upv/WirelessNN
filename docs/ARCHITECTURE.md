# Architecture

How the repository is organised, how one slot flows through the code, and the
conventions the modules share. For a hands-on introduction read
[TUTORIAL.md](TUTORIAL.md) first.

## Repository layout

```text
nr_ul_sim/            the simulator package (everything importable lives here)
examples/             starter files: numbered, self-contained, CPU-friendly
scripts/
  setup_env.sh        create .venv, install, check
  check_install.py    versions, GPU, trained models, smoke simulation
  physics_checks.py   block-by-block physics report (figures + JSON)
  campaigns/          large runs and their reports
  dataset/            sharded dataset runs, merging, analysis
  training/           trained estimators and working-point regressors
tests/                pytest suite (physics, estimators, searches, dataset, CLI)
docs/                 TUTORIAL.md, ARCHITECTURE.md
notebooks/            scratch notebooks
dataset/              tracked metadata of generated datasets (tensors are not in git)
models/               trained estimators (paper_ce), ridge baseline, regressor benchmarks
results/              figures of reference runs
```

Scripts and examples add the repository root to `sys.path` themselves, so they run
from a plain checkout; `pip install -e .` makes `nr_ul_sim` importable from anywhere.

## The package

```text
                         SimConfig  (parameters.py)
                             |
        +--------------------+---------------------+
        v                    v                     v
   pusch.py             channels.py          receivers.py
   PUSCH transmitter    CDL / UMi / UMa      PuschRx: estimators, detectors, decoder
        |                    |                     ^
        |                    |              channel_estimation.py  (factory)
        |                    |                windowed_ce.py   paper_ce.py
        v                    v                     |
   +---------------------------------------------------------+
   |  simulator.py   NRUplinkSimulator                        |
   |    generate_slot -> estimate_csi -> detect               |
   |    measure_point (Monte Carlo)  ->  run (SNR x IoT)      |
   +---------------------------------------------------------+
        ^                    |
   interference.py           v
   interferers, noise,   metrics.py   error counts, working point
   IRC covariance        plotting.py  figures
                         cli.py       python -m nr_ul_sim
```

| Module | Responsibility |
| --- | --- |
| `parameters.py` | `SimConfig`; the tuples of allowed channels, modulations, receivers, estimators and their one-line descriptions; name aliases; `detection_keys` |
| `pusch.py` | Sionna `PUSCHTransmitter` for all UEs, re-gridded onto the configured FFT with guard carriers |
| `channels.py` | antenna arrays, independent CDL links per transmitter, UMi/UMa models and their per-batch topology |
| `interference.py` | interferer symbols, `y = y_s + scaled y_i + noise`, the four `R_iot` estimators |
| `channel_estimation.py` | `build_estimator(name, …)`: Sionna LS / LMMSE with their priors, or one of the two modules below |
| `windowed_ce.py` | LS + hard / soft window in the delay domain |
| `paper_ce.py` | EqDeepRx (`ls_fir`, `denoise_nn`), data-covariance LMMSE, A-MMSE / RA-A-MMSE; loads `models/paper_ce` |
| `receivers.py` | `PuschRx`: estimators, one `LinearDetector` per receiver (IRC = `ExtraCovarianceLMMSE`), layer demapper, TB decoder |
| `simulator.py` | `NRUplinkSimulator`: the slot stages, the Monte-Carlo loop, the sweep |
| `metrics.py` | `ErrorStats`, `SweepResult`, `working_point_snr` |
| `plotting.py` | campaign figures and the shared `RECEIVER_STYLE` |
| `cli.py` | argument parser ⇄ `SimConfig`, `--list`, saving results |

Built on the simulator:

| Module | Responsibility |
| --- | --- |
| `dataset.py` | *channel tensor → working point* dataset: random scenarios, frozen channel (`FrozenChannelSimulator`), bracketing + bisection search, storage, packing, `python -m nr_ul_sim.dataset` |
| `ridge.py` | scalar-feature ridge baseline for that dataset |
| `random_channels.py` | per-configuration random CDL tables and frozen UMi/UMa drops |
| `random_campaign.py` | one configuration of the random-channel campaign (`RandomChannelSimulator`, features, record) |
| `wp_search.py` | adaptive working-point search with censoring for all estimators at once |

## One slot

`NRUplinkSimulator.measure_point(snr_db, iot_db)` repeats, up to `max_mc_iter` times:

1. `_draw_topology` — new UMi/UMa drop (nothing for CDL).
2. `transmit` — bits, LDPC, QAM, layer mapping, DMRS; scaled so that every UE radiates
   unit power in total (`tx_power_norm="per_ue"`).
3. `gen_serving` / `gen_int` — frequency-domain channels, normalised to unit mean
   energy per resource element; `apply_channel`.
4. `serving_plus_interference` — interferers scaled to the total INR, white noise
   `N0 = 10^(-SNR/10)` added.
5. `estimate_csi` — every configured estimator on the same received grid.
6. `detect` — for every `(result key, receiver, estimator)`: set the IRC covariance,
   equalise, demap (max-log), layer-demap, LDPC-decode, count errors.

It stops early once every curve has `num_target_bit_errors` bit errors and
`num_target_block_errors` block errors.

The **slot dict** passed between the stages:

| Key | Set by | Content |
| --- | --- | --- |
| `b` | `generate_slot` | transmitted bits `[B, UE, TB bits]` |
| `y` | `generate_slot` | received grid `[B, 1, RX ant, symbol, FFT bin]` |
| `h`, `h_int` | `generate_slot` | true serving / interferer channel `[B, 1, RX ant, TX, TX ant, symbol, FFT bin]`; `h_int` is `None` without interference |
| `no`, `iot_db` | `generate_slot` | noise variance, interference level |
| `csi` | `estimate_csi` | `{estimator: (h_hat, err_var)}`, `h_hat [B, 1, RX ant, UE, stream, symbol, used subcarrier]` |
| `h_perf` | `estimate_csi` | true channel in the layout of `h_hat` (precoding and power scale applied) |
| `h_hat`, `err_var` | `estimate_csi` | the first non-perfect estimate (kept for older callers) |

## Conventions

- **Units.** SNR and IoT in dB; delay spread in ns in the config, seconds internally;
  speed in km/h in the config, m/s internally; carrier in Hz (`--carrier-ghz` on the CLI).
- **SNR** is per receive antenna and resource element for one UE. **IoT** is the total
  interference-to-noise ratio per antenna; `0` disables interference.
- **Grid.** 30 kHz SCS, 14 symbols, `num_prb × 12` used subcarriers centred in the FFT;
  `cfg.guard_carriers` gives the unused bins on each side. Channels and `y` live on the
  FFT grid, estimates on the used subcarriers.
- **Result keys.** One estimator → receiver names; one receiver → estimator names; both
  swept → `receiver_estimator` (`parameters.detection_keys`).
- **Names.** Receiver and estimator names are canonicalised (lower case, `-` → `_`,
  aliases resolved) when the config is created; the rest of the code only sees the
  canonical names in `RECEIVERS` / `CHANNEL_ESTIMATORS`.
- **Randomness.** `SimConfig.seed` seeds Sionna, PyTorch and NumPy when the simulator
  is built. The dataset builder re-seeds per SNR point and the campaign per
  configuration, so their labels do not depend on the order of evaluation.
- **Device.** CUDA when available, otherwise CPU; chosen once in `simulator._select_device`.

## Three simulators, one base class

| Class | Channel per Monte-Carlo iteration | Used by |
| --- | --- | --- |
| `NRUplinkSimulator` | new realisation (and new UMi/UMa drop) every batch | sweeps, CLI |
| `FrozenChannelSimulator` (`dataset.py`) | one stored realisation reused in every slot | working-point dataset |
| `RandomChannelSimulator` (`random_campaign.py`) | new fading on a fixed, per-configuration random large-scale channel | random-channel campaign |

Two working-point searches exist for the same reason: `dataset.search_working_point`
(bracket + bisect one curve, labels `ok` / `not_reached` / `below_range`) and
`wp_search.search_working_point` (all estimators at once, interval labels with
censoring reasons). They are independent implementations with different outputs.
