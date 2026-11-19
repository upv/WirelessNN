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

## Files

| file | what it does |
|---|---|
| `models.py` | the five networks and the shared input/output conventions |
| `train.py` | training on `experiments/channelformer/gen_data.py` output |
| `run_all.sh` | trains all five in sequence and writes the summary |
| `summarize.py` | markdown table against Channelformer and LS interpolation |
| `plugin.py` | registers the `zoo_*` estimators |

## Reproduce

```bash
# data: see experiments/channelformer/README.md (12 000 + 1 200 slots, 4x2, 68 PRB)
bash experiments/ce_zoo/run_all.sh data/cf_4x2 runs/zoo
```

Training: random contiguous crops of 4 to 68 PRB (full band with probability 0.3),
Huber loss, AdamW 1e-3 with one-cycle schedule, batch 64 slices, 60 epochs. All five
take about 70 min in total on an RTX PRO 4000 (swift 7, denoiserx 11, helena 14,
chea 15, chea_xl 23 min).

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
| LS interpolation | 0 | -4.52 | | | |

DenoiseRx was the best model in every channel type (CDL-B/C/D, UMi, UMa). All runs had
stopped improving well before epoch 60. These are channel-estimation errors only; BLER
with these estimators has not been simulated yet.
