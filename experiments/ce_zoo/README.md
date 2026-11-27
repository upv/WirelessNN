# ce_zoo: recent neural channel estimators on the 4x2 PUSCH link

Five channel estimators from recent papers, adapted to `nr_ul_sim` as one `--ce-plugin`
and trained on the same data as [Channelformer](../channelformer/README.md) (4 gNB
antennas x 2 UE ports, rank 2, 68 PRB, DMRS type 1 with one additional position).

| estimator | paper | params |
|---|---|---|
| `zoo_chea` | CHEA, d=16: Abdollahpour et al., "Scalable Attention for 5G NR Channel Estimation", [arXiv:2607.16462](https://arxiv.org/abs/2607.16462) | 32.8k |
| `zoo_chea_xl` | CHEA-XL, d=64, same paper | 302k |
| `zoo_helena` | HELENA: Camelo Botero et al., [arXiv:2506.13408](https://arxiv.org/abs/2506.13408) | 133k |
| `zoo_swift` | SwiftChannel student network: Lyu et al., [arXiv:2605.01931](https://arxiv.org/abs/2605.01931) | 9.6k |
| `zoo_denoiserx` | EqDeepRx DenoiseNN + linear interpolation: Honkala et al., [arXiv:2602.11834](https://arxiv.org/abs/2602.11834) | 41.8k |
| `zoo_hard_window`, `zoo_soft_window` | baselines: the simulator's `ls_hard_window` / `ls_soft_window` on one slice | 0 |
| `zoo_lmmse_exp` | baseline: frequency LMMSE with an exponential delay prior on one slice | 0 |

These are re-implementations from the papers, not the authors' code. Every model works on
one (rx antenna, layer) slice: the OCC-despread DMRS LS values (one per 4 subcarriers,
both DMRS symbols) in, the 14-symbol slot out, for any number of PRB. The deviations from
each paper are listed in the model docstrings in `models.py`; the main ones:

- CHEA: one token per DMRS block (paper: per 3 subcarriers), so a 6-PRB window holds 18
  tokens; shifted windows in the second stage; output added to linear LS interpolation.
- HELENA: input is the zero-filled LS grid as in the paper; the global residual is the
  linear LS interpolation; conv widths (16, 8) are not given in the paper.
- SwiftChannel: the paper super-resolves a subcarrier x antenna SRS grid; here the network
  is 1D over frequency. Only the network, without distillation, re-parameterisation or QAT.
- DenoiseRx: depthwise-separable 13-tap convolutions (filter type not given in the paper);
  only the channel-estimation part of EqDeepRx, followed by fixed linear interpolation.
- Delay windows (`DelayWindow`): the same computation as `nr_ul_sim/windowed_ce.py`
  (window -1..+3 us, soft threshold 1.5), bit-identical with one receive antenna
  (`tests/test_ce_zoo_windows.py`). With several antennas the simulator's soft window
  averages tap power over them; here each slice is estimated alone. `train.py` evaluates
  them once and writes a checkpoint with the error-variance table, no training.
- LMMSE (`LmmseExp`): frequency interpolation with an exponential delay prior of fixed
  spread (default 300 ns), noise level from second differences of neighbouring blocks,
  linear in time. The simulator's `lmmse_exp` knows the noise variance and the channel
  family instead. `derotate=True` removes the mean delay first (used on air, 100 ns).

## Files

| file | what it does |
|---|---|
| `models.py` | the five networks and the shared input/output conventions |
| `train.py` | training on `experiments/channelformer/gen_data.py` output |
| `run_all.sh` | trains all five in sequence and writes the summary |
| `summarize.py` | markdown table against Channelformer and LS interpolation |
| `plugin.py` | registers the `zoo_*` estimators |
| `bler.py` | quick BLER sweep (working points) and the summary table |

## Reproduce

```bash
# data: see experiments/channelformer/README.md (12 000 + 1 200 slots, 4x2, 68 PRB)
bash experiments/ce_zoo/run_all.sh data/cf_4x2 runs/zoo
```

Training: random contiguous crops of 4 to 68 PRB (full band with probability 0.3),
Huber loss, AdamW 1e-3 with one-cycle schedule, batch 64 slices, 60 epochs. All five
take about 70 min in total on an RTX PRO 4000 (swift 7, denoiserx 11, helena 14,
chea 15, chea_xl 23 min).

## Other slot layouts and timing offsets

The models take any number of DMRS symbols and any slot length (`nd`, `dmrs`, `n_sym`;
`train.py` reads them from the data, `--dmrs` gives the DMRS symbol indices). `train.py`
accepts several data folders (concatenated) and `--tau-max-ns T`, which multiplies input
and target by the phase ramp of a random delay in [-T, T] per sample. That teaches the
models a receiver FFT-window (timing) offset; with it, the best checkpoint is chosen on
validation data with random offsets.

Example: a 2-RX stand layout (24 PRB, PUSCH mapping type B, 13 symbols, DMRS at 0/5/10),
rank-2 plus 1-layer data:

```bash
export PYTHONPATH=.:experiments/channelformer:experiments/ce_zoo
L="--rx-ant 2 --prb 24 --fft-size 512 --carrier-freq 3.6e9 --mapping-type B --num-symbols 13 --dmrs-addpos 2"
python experiments/channelformer/gen_data.py --out data/stand_r2 $L --ue-ant 2 --rank 2 --slots 24000 --val-slots 2400 --chunk-slots 32 --batch 4
python experiments/channelformer/gen_data.py --out data/stand_r1 $L --ue-ant 1 --rank 1 --slots 24000 --val-slots 2400 --chunk-slots 32 --batch 4
for m in denoiserx swift chea helena chea_xl; do
  python experiments/ce_zoo/train.py --model $m --data data/stand_r2 data/stand_r1 --out runs/stand/$m \
      --dmrs 0,5,10 --tau-max-ns 400 --min-prb 2 --epochs 40
done
```

Validation NMSE on that data (random offsets of up to ±400 ns; LS interpolation -6.28 dB):

| model | params | val NMSE [dB] |
|---|---|---|
| chea_xl | 297 784 | -17.70 |
| denoiserx | 43 074 | -17.56 |
| helena | 125 280 | -16.54 |
| swift | 9 404 | -15.62 |
| chea | 31 672 | -15.55 |
| soft_window | 0 | -12.79 |
| hard_window | 0 | -8.13 |
| lmmse_exp | 0 | -5.75 (-13.78 without offsets) |

### Over-the-air replay

These stand-layout checkpoints were replayed on uplink captures from an OAI-based 5G
stand with two commercial UEs (captures of 19 November 2026; 1 layer, 7 to 24 PRB,
mostly 24; uplink SNR reported by the gNB 16 to 21 dB). With 2 gNB receive antennas
there are six UE sets (each UE alone and four sets from a capture with both active),
2 567 to 2 768 decoded slots each; with 1 receive antenna, one UE and 2 762 slots.
There is no true channel on air, so each slot is scored by the fit SNR
|cHX|^2 / |Y - cHX|^2 on its data REs, with X rebuilt by re-encoding the decoded
transport block and one complex gain c per slot and receive antenna; with 2 antennas the
per-antenna fit SNRs are summed (maximum-ratio combining). Median per-slot gain over the
gNB's own estimate:

| estimator | 2 RX: range over the six UE sets [dB] | 1 RX [dB] |
|---|---|---|
| zoo_chea_xl | +0.88 to +0.97 | +0.38 |
| zoo_denoiserx | +0.87 to +0.97 | +0.36 |
| Channelformer (same layout, delay removed before the network) | +0.82 to +0.93 | +0.34 |
| zoo_helena | +0.76 to +0.85 | +0.26 |
| zoo_swift | +0.71 to +0.78 | +0.21 |
| zoo_chea | +0.66 to +0.73 | +0.17 |
| LMMSE, exponential delay profile | +0.58 to +0.78 | +0.21 |
| soft window, delay removed before the window | +0.35 to +0.76 | +0.07 |
| hard window, delay removed before the window | +0.09 to +0.33 | -0.21 |
| linear interpolation | -0.07 to +0.07 | -0.43 |
| hard window | -1.92 to -0.46 | -1.73 |
| soft window | -1.93 to -0.39 | -1.78 |

Every neural model beats the gNB's estimate in every UE set, and the top three beat
LMMSE in every set by 0.15 to 0.3 dB. The spread between UE sets is about 0.1 dB for the
neural models and 0.2 dB for LMMSE. At this SNR noise dominates the estimation error,
so a second receive antenna roughly doubles what a better estimate is worth.

The stand's receiver places its FFT window early, so the channel carries a delay of
about -110 ns. The offset-trained chea_xl, denoiserx and helena handle it themselves:
removing the mean delay before the network changes their result by at most 0.03 dB.
The delay windows suffer from that delay: over 24 PRB a -100 ns delay is a phase ramp
of about 0.9 cycle, so the DFT sees a jump at the band edges and the window cuts it
off (error -15 dB in the outer eighths of the band against -26 to -29 dB inside).
Removing the mean delay first fixes most of it, but the soft window still trails LMMSE
and every trained model.

Earlier 1-RX captures (18 November, four capture/UE sets, about half of the slots lost
to a tracing race in the receiver) gave the same order for the trained models and slightly smaller gains (zoo_chea_xl
+0.35 to +0.36 dB, LMMSE +0.16 to +0.21 dB).
The capture and replay tools are not part of this repository.

## Use the estimators

```bash
CE_ZOO_WEIGHTS=runs/zoo python -m nr_ul_sim --channel cdl-c --rank 2 --num-rx-ant 4 --num-ue-ant 2 \
    --receivers lmmse --estimators ls_lin,channelformer,zoo_denoiserx,zoo_chea \
    --ce-plugin experiments/channelformer/plugin.py --ce-plugin experiments/ce_zoo/plugin.py \
    --snr-db -4:24:2 --iot-db 0
```

## Results (validation NMSE)

Whole-band NMSE on the Channelformer validation split (1 200 slots, 9 600 slices; CDL-B/C/D,
UMi, UMa, 3 to 120 km/h, SNR uniform in -10..30 dB, no interference):

| model | params | val NMSE [dB] | -10 dB | 10 dB | 30 dB |
|---|---|---|---|---|---|
| zoo_denoiserx | 41 794 | **-16.60** | -10.1 | -22.5 | -24.6 |
| zoo_chea_xl | 302 288 | -16.29 | -9.7 | -22.4 | -24.8 |
| channelformer | 191 963 | -15.33 | -8.7 | -21.7 | -24.3 |
| zoo_chea | 32 816 | -15.24 | -8.5 | -21.3 | -24.4 |
| zoo_helena | 132 984 | -15.16 | -8.3 | -21.5 | -24.6 |
| zoo_swift | 9 628 | -14.56 | -7.7 | -20.9 | -24.3 |
| zoo_lmmse_exp | 0 | -13.07 | -6.3 | -19.2 | -21.9 |
| zoo_soft_window | 0 | -12.25 | | | |
| zoo_hard_window | 0 | -6.39 | | | |
| LS interpolation | 0 | -4.52 | | | |

The windows level off near -21 dB above about 15 dB SNR (band-edge leakage), below plain
LS at 30 dB. DenoiseRx was the best model in every channel type (CDL-B/C/D, UMi, UMa). All runs had
stopped improving well before epoch 60.

## Results (BLER working points)

SNR at 10 % BLER on the 4x2 link of the [Channelformer campaign](../channelformer/README.md)
(68 PRB, rank 2, LMMSE receiver + LDPC, MCS table 1 index 5/14/20 = QPSK/16QAM/64QAM),
averaged over the five channels (CDL-B, CDL-C, CDL-D, UMi, UMa) at 30 km/h. The last
column is the mean distance to perfect CSI over all 15 channel x modulation points.
All estimators ran on the same slots in each scenario.

| estimator | params | val NMSE [dB] | QPSK [dB] | 16QAM [dB] | 64QAM [dB] | loss vs perfect CSI [dB] |
|---|---|---|---|---|---|---|
| perfect CSI | | | -2.2 | 7.1 | 12.5 | |
| zoo_denoiserx | 41 794 | **-16.60** | -1.8 | 7.6 | 13.6 | **0.65** |
| zoo_chea_xl | 302 288 | -16.29 | -1.8 | 7.6 | 13.6 | 0.67 |
| zoo_helena | 132 984 | -15.16 | -1.7 | 7.6 | 13.6 | 0.71 |
| zoo_chea | 32 816 | -15.24 | -1.7 | 7.6 | 13.7 | 0.73 |
| channelformer | 191 963 | -15.33 | -1.7 | 7.6 | 13.7 | 0.73 |
| zoo_swift | 9 628 | -14.56 | -1.6 | 7.7 | 13.7 | 0.79 |
| lmmse_exp (simulator) | 0 | | -1.3 | 7.9 | 15.5 | 1.57 |

This is a quick sweep: 1 dB SNR grid (QPSK -5..3, 16QAM 3..14, 64QAM 8..24 dB), at most
96 slots or 50 block errors per point, one speed; the working points are good to about
±0.3 dB, so differences below that between the neural models are not significant. All
neural models are within 0.15 dB of each other on average and halve the loss of the
simulator's `lmmse_exp`. The gap to perfect CSI is small on CDL (0.0 to 0.6 dB, swift up to 0.8 dB) and grows
on UMi/UMa: 64QAM on UMi loses 1.7 dB (denoiserx, chea_xl) to 2.2 dB (channelformer), on
UMa about 3.2 dB, where `lmmse_exp` loses 8.9 dB. On CDL-D 64QAM `lmmse_exp` loses 2.6 dB
and every neural model none. Per-scenario curves: `results/bler_20261127/*.json`
(full `SimConfig` in each file; code at 862753e plus this script, Channelformer checkpoint
`channelformer_4x2.pt`, 62.6 min on an RTX PRO 4000).

```bash
export PYTHONPATH=.:experiments/channelformer:experiments/ce_zoo
CE_ZOO_WEIGHTS=runs/zoo python experiments/ce_zoo/bler.py --out results/zoo_bler
python experiments/ce_zoo/bler.py --out results/zoo_bler --summary runs/zoo   # the table above
```
