# Working-point dataset

`nr_ul_sim/dataset.py` builds a supervised dataset for learning the map
**channel tensor → working point**. Each scenario draws random link parameters,
**freezes one channel realisation**, and searches the SNR where the coded BER of the
chosen receiver(s) crosses `target_ber`, for every channel estimator and modulation.
The default sweep is IRC × {perfect, LS-NN, LS-linear, LS-lin + time average,
LMMSE-CE} × {QPSK, 16QAM, 64QAM}.

Freezing is what makes the label well posed: only data, noise and interferer symbols
are redrawn between Monte-Carlo iterations, so the working point belongs to the stored
tensor rather than to a channel ensemble (`--channel-mode ensemble` restores the usual
per-slot redraw). Scalar features stored with each tensor include the RMS delay
spread, the numerical and effective rank of `H`, and the rank, condition number and
dominant fraction of `R_uu`.

## Build

```bash
python -m nr_ul_sim.dataset --num-samples 500 --outdir dataset/run1 --num-prb 68
```

or edit the knobs at the top of `examples/05_working_point_dataset.py` and run it (a
small demo by default). `python -m nr_ul_sim.dataset --help` lists every option.

Randomised per scenario: channel family, RMS delay spread (log-uniform, CDL only),
1–4 UEs × rank 1–2, speed, IoT level (`--p-no-iot` of the scenarios get none) and the
number of interferers. Carrier, numerology, PRB count and antenna count stay fixed so
the tensors share one shape.

Runs are resumable and shardable: scenario `i` depends only on `(--seed, i)`, so
`--start-index` splits the work across processes and `--resume` appends to an existing
directory. `--pack-only` rebuilds `dataset.npz` / `dataset.csv` from `meta.jsonl`.
`scripts/dataset/run_irc_ce.sh` launches a sharded run and
`scripts/dataset/merge_shards.py` merges it.

**Power convention.** `dataset_version = 2` (default) uses the same SNR and IoT
definition as the simulator ([MODEL.md](MODEL.md#snr-and-iot)). Version 1 is the
legacy frozen-channel convention (unit power per layer, interference scaled per
interferer stream), kept only to extend old datasets; records store their version and
resume / pack refuse to mix versions. The datasets tracked under `dataset/` predate
the version flag and are version 1.

## Working-point search

Instead of a full SNR sweep, the crossing is **bracketed** on a `--coarse-step` grid
and then **bisected** down to `--refine-db`, reusing one BER measurement for all
receivers and warm-starting from the previous result. A scenario costs about 15 SNR
points instead of 26. Receivers that never reach the target within
`[--snr-min, --snr-max]` are labelled `not_reached` (`NaN` in the packed arrays), those
already below the range `below_range`.

Every SNR point is measured with a random stream seeded from `(scenario seed, SNR)`,
so the label does not depend on which points the search happened to visit; a resumed
or resharded run reproduces the same numbers.

## Output

| File | Content |
| --- | --- |
| `meta.jsonl`, `dataset.csv` | one record per (scenario, modulation): all parameters, per-receiver working point and status, the measured BER/BLER curve, channel features, paths to the arrays |
| `channels/<id>/H.npy` | serving channel `[rx, streams, symbols, subcarriers]`, complex64 |
| `channels/<id>/H_int.npy` | interferer channel, same layout (only if IoT > 0) |
| `channels/<id>/R_iot.npy` | normalised interference spatial covariance `[rx, rx]` |
| `channels/<id>/meta.json` | simulation knobs of that drop: IoT, channel family, UE/rank/speed, per-modulation working points |
| `dataset.npz` | the same records stacked and zero-padded, for batch training |

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

The stored tensor is the true channel, subsampled to `--num-symbol-bins` OFDM symbols
× `--num-freq-bins` subcarriers spread evenly over the used band (not the DMRS
estimate; receivers still estimate their own CSI during the search). Streams are
ordered UE-major; `num_layers_total` of them are valid and the rest are zero padding.

## Analysis

```bash
python scripts/dataset/analyze_dataset.py dataset/run1
```

prints label coverage per receiver, the physical gaps that validate the run (16QAM −
QPSK, L-MMSE − ideal MMSE, IRC gain per IoT bin) and a ridge regression on the scalar
features alone, the bar a model trained on the raw tensors has to beat. It writes
`analysis.png` next to the dataset. `scripts/dataset/analyze_ce.py` studies the
channel-estimation loss, and `scripts/training/train_ridge.py`,
`train_wp_models.py`, `train_wp_models2.py` train working-point regressors (results in
`models/ridge`, `models/wp_bench`, `models/wp_bench2`).

## Tracked datasets

`dataset/` holds the metadata of datasets built during development (`meta.jsonl`,
`dataset.csv`, per-channel `meta.json`, figures); the channel tensors are not in Git.
All of them were built at **8 PRB**, before 68 PRB became the default, and with the
legacy power convention (version 1), so they are references for the regression
scripts rather than inputs for new studies.

| Directory | Scenarios | Receivers × estimators | Used by |
| --- | ---: | --- | --- |
| `demo`, `demo100` | 40, 100 | five receivers, DMRS-LS, QPSK/16QAM | demonstration |
| `main`, `run5000` | 5000 each | five receivers, DMRS-LS, QPSK/16QAM | `train_ridge.py` → `models/ridge` |
| `irc_ce100` | 100 | IRC × five estimators, three modulations | `analyze_ce.py` |
| `irc_ce10k` | 20 shards × 500 | IRC × five estimators, three modulations | `train_wp_models*.py` → `models/wp_bench*` |
