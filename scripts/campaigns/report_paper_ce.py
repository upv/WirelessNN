#!/usr/bin/env python3
"""Report on the paper channel estimators: training, NMSE, link-level campaign.

    .venv/bin/python scripts/campaigns/report_paper_ce.py reports/paper_ce [models/paper_ce]

Reads <dir>/nmse/nmse.json, <dir>/campaigns/{ce,incm}/*/*.json and
<models>/train_meta.json; writes <dir>/REPORT.md and figures.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nr_ul_sim.plotting import RECEIVER_STYLE  # noqa: E402

D = Path(sys.argv[1]).resolve()
M = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "models" / "paper_ce"
NEW = ["ls_fir", "denoise_nn", "lmmse_data", "lmmse_data_1d", "a_mmse", "ra_a_mmse"]
OLD = ["ls_nn", "ls_lin", "ls_lin_time_avg", "lmmse_exp", "ls_hard_window", "ls_soft_window"]
CAMP_EST = ["ls_lin", "ls_soft_window", "lmmse_exp", "ls_fir", "denoise_nn", "lmmse_data", "lmmse_data_1d", "a_mmse", "ra_a_mmse"]
CH = ["uma", "umi", "cdl-b", "cdl-c"]
MODS = ["qpsk", "qam16", "qam64"]
CFGS = [("1", "1"), ("2", "1"), ("2", "2")]


def label(e):
    return RECEIVER_STYLE.get(e, {}).get("label", e)


def f(v, nd=1):
    return "n/r" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


L: list[str] = []
add = L.append
add("# Channel estimators from EqDeepRx and A-MMSE in the WirelessNN simulator\n")
add("""Papers:
- **EqDeepRx** — M. Honkala, D. Korpi, E. Raninen, J. Huttunen, *EqDeepRx: Learning a Scalable MIMO Receiver*, arXiv:2602.11834 (Nokia Bell Labs).
- **A-MMSE** — T. Ha, C. Jung, H. Kim, J. Park, J. Park, *Learning MMSE Filters for OFDM Channel Estimation: Attention Transformer Gains at Linear Inference*, arXiv:2506.00452v5.

Code: `nr_ul_sim/paper_ce.py` (estimators), `nr_ul_sim/interference.py` (INCM), `scripts/training/train_paper_ce.py` (training), `scripts/training/eval_paper_ce.py` (NMSE), `scripts/campaigns/run_paper_ce_campaign.sh` (BER/BLER), `tests/test_paper_ce.py`.
""")

add("## 1. What was taken from the papers and how it was adapted to 68-PRB PUSCH\n")
add("""| name | paper | method | adaptation |
|---|---|---|---|
| `ls_fir` | EqDeepRx, Sec. II-B (baseline) | LS on the pilots → static frequency smoothing filter → linear interpolation | 17-tap FIR (Hann × sinc), delay band [−1, +3] µs, normalised at the band edges |
| `denoise_nn` | EqDeepRx, Sec. III-A | LS → DenoiseNN (1D convolutions over frequency with downsampling, 1×1 mixer over DMRS symbols, per RX antenna / layer pair) → linear interpolation | 4 subsampled residual blocks, 32 channels; trained on its own with channel MSE (the paper trains end to end through the receiver BCE) |
| INCM `--iot-cov incm_oas` | EqDeepRx, Sec. II-C | interference-plus-noise covariance from the DMRS residual `y − Ĥp`, 2-PRB bands, linear shrinkage towards tr(S)/p·I | shrinkage weight: oracle for the complex Gaussian case with unbiased moment estimates (OAS-like) |
| `lmmse_data` | A-MMSE, Sec. III-A ("LMMSE mismatch") | 2D LMMSE with a covariance estimated from training channels | R(t₁,t₂,Δf) under a WSS assumption from 18 432 training grids, biased (PSD) estimate |
| `lmmse_data_1d` | A-MMSE, eq. (9) | 1D LMMSE over frequency on each DMRS symbol + time interpolation | same covariance |
| `a_mmse` | A-MMSE, Sec. IV | two-stage attention encoder (frequency → time) + ResFC decoder learn **one fixed linear filter W** | one filter per 4-PRB block (48 subcarriers × 14 symbols from 2×12 pilot values), shared by all 17 blocks; filter bank over SNR (−10…35 dB, step 5) and CDM-group comb offset; selected by N0 |
| `ra_a_mmse` | A-MMSE, Sec. V | W·U·Vᵀ with trainable U, V of rank r | r = 6 of 24 (25 %) |

Common to all: the input is the CDM-despread LS estimate of one RX antenna and one layer (one value per 4-subcarrier DMRS block; 68 PRB → 2×204 values). Training (EqDeepRx protocol): UMa, speed U(0, 35 m/s), 1–2 UE × rank 1–2, 68 PRB, 4 RX antennas, SNR U(−10, 40) dB for DenoiseNN and a fixed SNR per A-MMSE filter. UMi, CDL-B and CDL-C are **outside the training distribution**.
""")

# ------------------------------------------------------------------ training
meta_p = M / "train_meta.json"
if meta_p.exists():
    meta = json.loads(meta_p.read_text())
    add("## 2. Training (validation on UMa)\n")
    add(f"Training samples (RX antenna × layer): {meta['train_samples']}, validation: {meta['val_samples']}, time {meta['duration_s'] / 60:.0f} min on an RTX PRO 4000.\n")
    v = meta["val_nmse_db_full_grid"]
    add("| SNR, dB | LS + lin. interp. | DenoiseNN | LMMSE (data) |")
    add("|---|---|---|---|")
    for snr in sorted(v, key=float):
        r = v[snr]
        add(f"| {snr} | {f(r.get('ls_lin_interp'))} | {f(r.get('denoise_nn'))} | {f(r.get('lmmse_data'))} |")
    a = meta["val_nmse_db_ammse_chunks"]
    add("\nA-MMSE on 4-PRB blocks (NMSE, dB; comb 0):\n")
    add("| SNR | LS lin. | A-MMSE fixed W | A-MMSE input-dependent W(x) | fixed W, fine-tuned directly | RA-A-MMSE r=6 |")
    add("|---|---|---|---|---|---|")
    for k, r in a.items():
        if k.startswith("c0_"):
            add(f"| {k.split('snr')[1]} | {f(r['ls_lin_interp'])} | {f(r['a_mmse_fixed'])} | {f(r['a_mmse_adaptive'])} | {f(r['fixed_filter_finetuned'])} | {f(r['ra_a_mmse'])} |")
    add("\nThe fixed W (mean of the network outputs, as in the paper: \"one final A-MMSE filter\") is no worse than the input-dependent W(x), confirming that A-MMSE converges to the best fixed linear filter; fine-tuning W directly gains a little more (that is the limit the paper refers to).\n")

# ------------------------------------------------------------------ NMSE in the simulator
nm_p = D / "nmse" / "nmse.json"
if nm_p.exists():
    nm = json.loads(nm_p.read_text())
    add("## 3. NMSE in the simulator (68 PRB, all estimators)\n")
    add("![NMSE](nmse/nmse.png)\n")
    add("`eval_paper_ce.py --slots 256 --batch 16`: 256 slots per point, the same channel realisations "
        "at every SNR (the seed is reset before each point). NMSE = Σ|Ĥ−H|² / Σ|H|² over all slots.\n")
    ests = list(next(iter(nm.values()))["nmse_db"])
    for snr in (0.0, 10.0, 20.0):
        add(f"**SNR {snr:g} dB**, NMSE in dB (bold: best realisable):\n")
        add("| scenario | " + " | ".join(label(e) for e in ests) + " |")
        add("|---|" + "---|" * len(ests))
        for lab, r in nm.items():
            i = r["snr_db"].index(snr)
            vals = [r["nmse_db"][e][i] for e in ests]
            real = [v for e, v in zip(ests, vals) if e != "perfect"]
            best = min(real)
            add(f"| {lab} | " + " | ".join(("**" + f(v) + "**") if v == best else f(v) for v in vals) + " |")
        add("")

# ------------------------------------------------------------------ campaign
camps = {}
for p in sorted((D / "campaigns").glob("*/*/*.json")):
    camps[p.parent.name] = json.loads(p.read_text())
if camps:
    add("## 4. Link level: IRC working point (68 PRB)\n")
    add(f"{len(camps)} campaigns. Working point: SNR at BER = 1e-2 (and BLER = 0.1); loss relative to perfect CSI on the same slots.\n")
    loss = collections.defaultdict(list)
    loss_b = collections.defaultdict(list)
    rows = []
    for ch in CH:
        for mod in MODS:
            for u, r in CFGS:
                c = camps.get(f"ce_{ch}_{mod}_ue{u}_r{r}")
                if not c:
                    continue
                for iot, cur in c["iot"].items():
                    p = cur["perfect"]["working_point_db"]
                    pb = cur["perfect"]["working_point_bler_db"]
                    vals = []
                    for e in CAMP_EST:
                        w = cur.get(e, {}).get("working_point_db")
                        wb = cur.get(e, {}).get("working_point_bler_db")
                        vals.append(w)
                        for key in (e, (e, ch), (e, mod), (e, f"{u}UE r{r}"), (e, f"IoT {float(iot):g}")):
                            if w is not None and p is not None:
                                loss[key].append(w - p)
                            if wb is not None and pb is not None:
                                loss_b[key].append(wb - pb)
                    rows.append((ch, mod, f"{u}UE r{r}", float(iot), p, vals))
    add("### 4.1 Median working-point loss, dB (BER = 1e-2 / BLER = 0.1)\n")
    groups = ["all"] + CH + MODS + [f"{u}UE r{r}" for u, r in CFGS] + ["IoT 0", "IoT 10"]
    add("| estimator | " + " | ".join(groups) + " | not reached |")
    add("|---|" + "---|" * (len(groups) + 1))
    for e in CAMP_EST:
        cells = []
        for g in groups:
            k = e if g == "all" else (e, g)
            a1, a2 = loss.get(k, []), loss_b.get(k, [])
            cells.append(f"{np.median(a1):.2f} / {np.median(a2):.2f}" if a1 and a2 else "—")
        miss = sum(1 for row in rows if row[5][CAMP_EST.index(e)] is None and row[4] is not None)
        add(f"| {label(e)} | " + " | ".join(cells) + f" | {miss}/{len(rows)} |")
    add("")
    # figure: loss boxplot
    fig, ax = plt.subplots(figsize=(10, 4.2))
    data = [loss[e] for e in CAMP_EST if loss[e]]
    labs = [label(e) for e in CAMP_EST if loss[e]]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    for patch, e in zip(bp["boxes"], [e for e in CAMP_EST if loss[e]]):
        patch.set_facecolor(RECEIVER_STYLE.get(e, {}).get("color", "grey"))
    ax.set_xticks(range(1, len(labs) + 1), labs, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("WP loss vs perfect CSI [dB]")
    ax.set_title("IRC working-point loss (BER = 1e-2), 68 PRB, all channels / modulations / UE configs / IoT")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(D / "ce_loss.png", dpi=130)
    plt.close(fig)
    add("![CE loss](ce_loss.png)\n")
    add("### 4.2 All working points (BER = 1e-2), dB\n")
    add("| channel | mod. | UE | IoT | perfect | " + " | ".join(label(e) for e in CAMP_EST) + " |")
    add("|---|---|---|---|---|" + "---|" * len(CAMP_EST))
    for ch, mod, cfg, iot, p, vals in rows:
        best = min([v for v in vals if v is not None], default=None)
        add(f"| {ch} | {mod} | {cfg} | {iot:g} | {f(p)} | " + " | ".join(("**" + f(v) + "**") if v is not None and v == best else f(v) for v in vals) + " |")
    add("")

    inc = [(k, v) for k, v in camps.items() if k.startswith("incm_")]
    if inc:
        add("### 4.3 Interference covariance for IRC (16QAM, 1 interferer)\n")
        add("Working point at BER = 1e-2, dB. `perfect`: true covariance, `residual`: wideband from the DMRS residual, `incm_oas`: EqDeepRx INCM (2 PRB + shrinkage).\n")
        add("| channel | UE | IoT | method | " + " | ".join(f"IRC/{label(e)}" for e in ("ls_lin", "denoise_nn", "lmmse_data")) + " | L-MMSE/LS-lin |")
        add("|---|---|---|---|---|---|---|---|")
        for ch in ("uma", "cdl-c"):
            for nue in ("1", "2"):
                for iot in ("5.0", "10.0", "20.0"):
                    for cov in ("perfect", "residual", "incm_oas"):
                        c = camps.get(f"incm_{ch}_ue{nue}_{cov}")
                        if not c or iot not in c["iot"]:
                            continue
                        cur = c["iot"][iot]
                        vals = [cur.get(f"irc_{e}", {}).get("working_point_db") for e in ("ls_lin", "denoise_nn", "lmmse_data")]
                        lm = cur.get("lmmse_ls_lin", {}).get("working_point_db")
                        add(f"| {ch} | {nue} | {float(iot):g} | {cov} | " + " | ".join(f(v) for v in vals) + f" | {f(lm)} |")
        add("")

(D / "REPORT.md").write_text("\n".join(L))
print(f"wrote {D / 'REPORT.md'}")
