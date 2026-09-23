from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


RECEIVER_STYLE = {
    "mr": dict(color="#4c78a8", marker="o", label="MR"),
    "lmmse": dict(color="#f58518", marker="s", label="L-MMSE"),
    "ideal_mmse": dict(color="#9d755d", marker="P", ls="--", label="Ideal MMSE"),
    "zf": dict(color="#54a24b", marker="^", label="ZF"),
    "irc": dict(color="#e45756", marker="D", label="IRC"),
    "perfect": dict(color="#9d755d", marker="P", ls="--", label="Perfect CSI"),
    "ls_nn": dict(color="#4c78a8", marker="o", label="LS-NN"),
    "ls_lin": dict(color="#f58518", marker="s", label="LS-linear"),
    "ls_lin_time_avg": dict(color="#54a24b", marker="^", label="LS-lin + time avg"),
    "lmmse_ce": dict(color="#b279a2", marker="D", label="LMMSE-CE"),
}


def _title(cfg: dict[str, Any]) -> str:
    return (
        f"NR PUSCH UL  {cfg['channel'].upper()}  {cfg['modulation'].upper()}  "
        f"{cfg['num_ue']} UE × rank {cfg['rank']}  "
        f"{cfg['num_rx_ant']} RX ant  {cfg['speed_kmh']} km/h"
    )


def _plot_metric(ax, curves: dict[str, Any], metric: str, mark_wp: bool) -> None:
    for name, style in RECEIVER_STYLE.items():
        if name not in curves or metric not in curves[name]:
            continue
        res = curves[name]
        snr = np.asarray(res["snr_db"], dtype=float)
        values = np.clip(np.asarray(res[metric], dtype=float), 1e-12, 1)
        ax.semilogy(snr, values, **style)
        if mark_wp and res.get("working_point_db") is not None:
            ax.axvline(res["working_point_db"], color=style["color"], ls="--", lw=0.8, alpha=0.7)


def _style_axis(ax, ylabel: str, iot: str, target: float | None) -> None:
    ax.set_xlabel("Per-antenna SNR [dB]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"IoT = {iot} dB")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    if target is not None:
        ax.axhline(target, color="grey", ls=":", lw=0.9, label=f"{ylabel}={target:g}")
    ax.legend(loc="best", fontsize=8)


def plot_campaign(campaign: dict[str, Any], outfile: str | Path) -> Path:
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg = campaign["config"]
    iot_levels = list(campaign["iot"])
    fig, axes = plt.subplots(2, len(iot_levels), figsize=(5.2 * len(iot_levels), 8.4), squeeze=False)
    target = cfg.get("target_ber", 0.01)
    for col, iot in enumerate(iot_levels):
        curves = campaign["iot"][iot]
        _plot_metric(axes[0, col], curves, "ber", mark_wp=True)
        _style_axis(axes[0, col], "BER", iot, target)
        _plot_metric(axes[1, col], curves, "bler", mark_wp=False)
        _style_axis(axes[1, col], "BLER", iot, None)
    fig.suptitle(_title(cfg), fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    return outfile


def plot_working_points(campaign: dict[str, Any], outfile: str | Path) -> Path:
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg = campaign["config"]
    iot_levels = list(campaign["iot"])
    names = [n for n in RECEIVER_STYLE if any(n in campaign["iot"][iot] for iot in iot_levels)]
    x = np.arange(len(iot_levels))
    n_rx = max(len(names), 1)
    width = 0.8 / n_rx
    fig, ax = plt.subplots(figsize=(max(6.4, 1.8 * len(iot_levels)), 4.8))
    for i, name in enumerate(names):
        style = RECEIVER_STYLE[name]
        values = []
        for iot in iot_levels:
            wp = campaign["iot"][iot].get(name, {}).get("working_point_db")
            values.append(float(wp) if wp is not None else np.nan)
        ax.bar(
            x + (i - (n_rx - 1) / 2) * width,
            values,
            width,
            color=style["color"],
            edgecolor="black",
            linewidth=0.4,
            label=style["label"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{iot} dB" for iot in iot_levels])
    ax.set_xlabel("IoT")
    ax.set_ylabel("Working-point SNR [dB]")
    ax.set_title(f"SNR where BER = {cfg.get('target_ber', 0.01):g}")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(_title(cfg), fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    return outfile


def save_campaign_plots(campaign: dict[str, Any], outdir: str | Path, stem: str) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return [
        plot_campaign(campaign, outdir / f"{stem}.png"),
        plot_working_points(campaign, outdir / f"{stem}_working_points.png"),
    ]
