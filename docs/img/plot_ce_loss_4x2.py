#!/usr/bin/env python3
"""README figure: mean SNR loss vs perfect CSI at 10 % BLER, 4x2 MIMO campaign.

    python docs/img/plot_ce_loss_4x2.py

Reads experiments/channelformer/results/mimo4x2_20261116/summary.json and writes
docs/img/ce_loss_4x2.png. Means are over the scenarios where both the estimator and
perfect CSI reached 10 % BLER (3 modulations x 3 speeds x channels).
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "experiments/channelformer/results/mimo4x2_20261116/summary.json"
OUT = Path(__file__).with_name("ce_loss_4x2.png")
ESTIMATORS = ["channelformer", "denoise_nn", "ls_soft_window", "lmmse_data", "lmmse_exp",
              "lmmse_data_1d", "a_mmse", "ls_fir", "ls_hard_window", "ls_lin"]
GROUPS = [("CDL-B/C/D", ("cdl-b", "cdl-c", "cdl-d"), "#2a78d6"),
          ("UMi/UMa", ("umi", "uma"), "#eb6834")]


def mean_loss(d, est, prefixes):
    loss = []
    for mod in d.values():
        for scen, p in mod["perfect"].items():
            v = mod[est].get(scen)
            if scen.startswith(prefixes) and p is not None and v is not None and not math.isnan(v):
                loss.append(v - p)
    return sum(loss) / len(loss)


d = json.loads(SUMMARY.read_text())
fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
h = 0.38
for g, (name, prefixes, color) in enumerate(GROUPS):
    vals = [mean_loss(d, e, prefixes) for e in ESTIMATORS]
    ys = [i + (g - 0.5) * (h + 0.02) for i in range(len(ESTIMATORS))]
    ax.barh(ys, vals, height=h, color=color, label=name, zorder=2)
    for y, v in zip(ys, vals):
        ax.text(v + 0.06, y, f"{v:.2f}", va="center", fontsize=7.5, color="#52514e")
ax.set_yticks(range(len(ESTIMATORS)), ESTIMATORS, fontsize=9, color="#0b0b0b")
ax.invert_yaxis()
ax.set_xlabel("mean SNR loss vs perfect CSI at 10 % BLER [dB]", fontsize=9, color="#52514e")
ax.set_title("4x2 MIMO PUSCH, 68 PRB, rank 2, LMMSE receiver", fontsize=10, color="#0b0b0b", loc="left")
ax.grid(True, axis="x", color="#e4e3df", lw=0.8, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#b9b8b2")
ax.tick_params(axis="both", colors="#52514e", length=0)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_xlim(0, 4.4)
fig.tight_layout()
fig.savefig(OUT, dpi=150, facecolor=fig.get_facecolor())
print(OUT)
