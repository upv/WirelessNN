"""BER curves and working-point markers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


RECEIVER_STYLE = {
    "mr": dict(color="#4c78a8", marker="o", label="MR"),
    "lmmse": dict(color="#f58518", marker="s", label="L-MMSE"),
    "zf": dict(color="#54a24b", marker="^", label="ZF"),
    "irc": dict(color="#e45756", marker="D", label="IRC"),
}


def plot_campaign(campaign: dict[str, Any], outfile: str | Path) -> Path:
    """Plot BER vs per-antenna SNR for every IoT level in a campaign."""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg = campaign["config"]
    iot_levels = list(campaign["iot"].keys())
    n = len(iot_levels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    target = cfg.get("target_ber", 0.01)

    for ax, iot in zip(axes[0], iot_levels):
        curves = campaign["iot"][iot]
        for name, style in RECEIVER_STYLE.items():
            if name not in curves:
                continue
            snr = np.asarray(curves[name]["snr_db"], dtype=float)
            ber = np.clip(np.asarray(curves[name]["ber"], dtype=float), 1e-12, 1)
            ax.semilogy(snr, ber, **style)
            wp = curves[name]["working_point_db"]
            if wp is not None:
                ax.axvline(wp, color=style["color"], ls="--", lw=0.8, alpha=0.7)
        ax.axhline(target, color="grey", ls=":", lw=0.9, label=f"BER={target:g}")
        ax.set_xlabel("Per-antenna SNR [dB]")
        ax.set_ylabel("BER")
        ax.set_title(f"IoT = {iot} dB")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8)

    title = (
        f"NR PUSCH UL  {cfg['channel'].upper()}  {cfg['modulation'].upper()}  "
        f"{cfg['num_ue']} UE × rank {cfg['rank']}  "
        f"{cfg['num_rx_ant']} RX ant  {cfg['speed_kmh']} km/h"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    return outfile
