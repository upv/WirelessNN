# Channelformer for nr_ul_sim and a 4x2 MIMO channel-estimation campaign

Channelformer (Luan & Thompson, "Channelformer: Attention based Neural Solution for
Wireless Channel Estimation and Effective Online Training", IEEE Trans. Wireless
Commun., 2023, [arXiv:2302.04368](https://arxiv.org/abs/2302.04368)), offline variant,
implemented as an `--ce-plugin` for `nr_ul_sim`, and a campaign that compares it with
the 14 built-in estimators on a 4 gNB antenna x 2 UE port (rank 2) PUSCH link.

## Files

| file | what it does |
|---|---|
| `model.py` | the network and the 68-PRB assembly from 12-PRB windows |
| `plugin.py` | registers the estimator `channelformer` |
| `gen_data.py` | training data from the simulator |
| `train.py` | training with the paper's hyper-parameters |
| `run_campaign.py` | BLER/throughput and CE NMSE of all estimators |
| `summarize.py` | report.md, summary.json and plots from a campaign folder |
| `weights/channelformer_4x2.pt` | trained checkpoint used for the results below |
| `results/mimo4x2_20261116/` | campaign output: one JSON per scenario, report, plots |

## Use the estimator

```bash
python -m nr_ul_sim --channel cdl-c --rank 2 --num-rx-ant 4 --num-ue-ant 2 \
    --receivers lmmse --estimators ls_lin,lmmse_exp,channelformer \
    --ce-plugin experiments/channelformer/plugin.py --snr-db -4:24:2 --iot-db 0
```

The checkpoint path can be overridden with `CHANNELFORMER_CKPT`. The model needs a
multiple of 4 PRB (at least 12), DMRS type 1 with one additional position, and was
trained for CDM group 0 (DMRS ports 0 and 1).

## Adaptation to NR PUSCH

The paper's network takes 72 LS pilot values (2 DMRS symbols x 36) and outputs a full
14 x 72 slot. NR DMRS type 1 with FD-OCC gives one despread LS value per 4 subcarriers,
so a 12-PRB window holds exactly 72 values. The network runs on 12-PRB windows of each
(rx antenna, layer) slice and keeps the central 4 PRB; 17 windows cover 68 PRB, clamped
at the band edges. Layers follow the paper: FC 72->216 into Q/K/V, 2 heads (one per DMRS
symbol), add & LayerNorm, a 2x2 convolutional pre-network, a decoder with three 5x5
residual blocks (12 channels), FC upsampling to 14 x 144 and a 5x5 output convolution;
Huber loss, Adam 2e-3. 192k parameters. Inputs are normalised by the RMS of each window.
The reported error variance comes from a validation MSE-vs-SNR table.

## Reproduce

```bash
export PYTHONPATH=.:experiments/channelformer
# 12 000 + 1 200 slots: CDL-B/C/D (30/100/300 ns), UMi, UMa, 3/30/60/120 km/h, SNR U(-10, 30)
python experiments/channelformer/gen_data.py --out data/cf_4x2 --slots 12000 --val-slots 1200 \
    --chunk-slots 32 --batch 4
python experiments/channelformer/train.py --data data/cf_4x2 --out runs/cf_4x2_w8 \
    --epochs 25 --drop-every 12 --windows-per-slice 8
CHANNELFORMER_CKPT=runs/cf_4x2_w8/channelformer.pt \
    python experiments/channelformer/run_campaign.py --out results/mimo4x2
python experiments/channelformer/summarize.py results/mimo4x2
```

Data generation takes about 6 min, training about 16 min, and the campaign 13.8 h on an
RTX PRO 4000 (24 GB; the campaign uses about 22 GB).

## Campaign configuration

One UE, 4 gNB antennas (one 38.901 panel 1x2 dual-pol), 2 UE ports, rank 2
non-codebook; 3.5 GHz, 30 kHz, 68 PRB, FFT 1024; DMRS type 1, additional position 1;
LMMSE detector, LDPC; no interference. Channels CDL-B 100 ns, CDL-C 300 ns, CDL-D 100 ns,
UMi, UMa at 3/30/120 km/h; MCS table 1 index 5/14/20 (TBS 14 600/42 016/65 576 bit);
BLER sweeps QPSK -10..16, 16QAM -4..24, 64QAM 2..30 dB in 2 dB steps, up to 200 slots
per point (stop at 100 block errors); NMSE -10..30 dB, 64 slots per point.

## Results

Mean SNR loss against perfect CSI at 10 % BLER, over the 45 scenarios where each
estimator reached 10 % BLER (number of scenarios where it did not in brackets):

| estimator | mean loss [dB] |
|---|---|
| channelformer | 0.91 (1) |
| denoise_nn* | 1.18 (1) |
| ls_soft_window | 1.23 (1) |
| lmmse_data* | 1.29 (0) |
| lmmse_data_1d* | 1.35 (1) |
| lmmse_exp | 1.43 (0) |
| a_mmse* | 1.58 (1) |
| ls_fir | 1.70 (1) |
| ls_hard_window | 2.07 (1) |
| ra_a_mmse* | 2.75 (5) |
| ls_lin | 2.85 (3) |
| ls_nn | 3.40 (9) |
| ls_lin_time_avg | 2.11 (12) |
| lmmse_ce | 0.56 (18) |

\* trained on UMa only (`models/paper_ce`). `lmmse_ce` uses a TDL prior that equals the
CDL power-delay profile (a genie on CDL) and fails on UMi/UMa. Each BLER point has at
most 200 blocks, so differences below about 0.3 dB are within Monte-Carlo noise. Full
tables and plots: `results/mimo4x2_20261116/report.md`.
