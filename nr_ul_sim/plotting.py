"""Figures of a campaign: BER/BLER curves and working-point bar charts.

``RECEIVER_STYLE`` gives every receiver and channel estimator its colour,
marker and label; the report scripts reuse it so figures stay consistent.
"""

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
    "lmmse_ce": dict(color="#b279a2", marker="D", label="LMMSE-CE (TDL prior)"),
    "lmmse_exp": dict(color="#ff9da6", marker="d", label="LMMSE-CE (exp. prior)"),
    "ls_hard_window": dict(color="#e45756", marker="v", label="LS + hard window"),
    "ls_soft_window": dict(color="#72b7b2", marker="X", label="LS + soft window"),
    "ls_fir": dict(color="#bab0ac", marker="h", label="LS + freq. FIR (EqDeepRx base)"),
    "denoise_nn": dict(color="#2f4b7c", marker="*", label="DenoiseNN (EqDeepRx)"),
    "lmmse_data": dict(color="#665191", marker="p", label="LMMSE, data covariance"),
    "lmmse_data_1d": dict(color="#a05195", marker="<", label="1D-LMMSE, data covariance"),
    "a_mmse": dict(color="#d45087", marker=">", label="A-MMSE"),
    "ra_a_mmse": dict(color="#f95d6a", marker="8", label="RA-A-MMSE"),
}
WP_KEY = {"ber": "working_point_db", "bler": "working_point_bler_db"}
TARGET_KEY = {"ber": "target_ber", "bler": "target_bler"}
DEFAULT_TARGET = {"ber": 0.01, "bler": 0.1}


def _style_for(name: str) -> dict:
    if name in RECEIVER_STYLE:
        return RECEIVER_STYLE[name]
    # receiver_estimator keys of a joint sweep
    for rx, style in RECEIVER_STYLE.items():
        if name.startswith(rx + "_"):
            est = name[len(rx) + 1 :]
            est_style = RECEIVER_STYLE.get(est, {})
            return dict(style, label=f"{style['label']} / {est_style.get('label', est)}",
                        marker=est_style.get("marker", style["marker"]))
    return dict(color="black", marker=".", label=name)


def _title(cfg: dict[str, Any]) -> str:
    return (
        f"NR PUSCH UL  {cfg['channel'].upper()}  {cfg['modulation'].upper()}  "
        f"{cfg['num_ue']} UE × rank {cfg['rank']}  "
        f"{cfg['num_rx_ant']} RX ant  {cfg['speed_kmh']} km/h"
    )


def _curve_names(campaign: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for curves in campaign["iot"].values():
        for name in curves:
            if name not in seen:
                seen.append(name)
    return seen


def _plot_metric(ax, curves: dict[str, Any], metric: str, mark_wp: bool) -> None:
    for name in curves:
        res = curves[name]
        if metric not in res:
            continue
        style = _style_for(name)
        snr = np.asarray(res["snr_db"], dtype=float)
        values = np.clip(np.asarray(res[metric], dtype=float), 1e-12, 1)
        ax.semilogy(snr, values, **style)
        wp = res.get(WP_KEY[metric])
        if mark_wp and wp is not None:
            ax.axvline(wp, color=style["color"], ls="--", lw=0.8, alpha=0.7)


def _style_axis(ax, ylabel: str, iot: str, target: float | None) -> None:
    ax.set_xlabel("Per-antenna SNR [dB]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"IoT = {iot} dB")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    if target is not None:
        ax.axhline(target, color="grey", ls=":", lw=0.9, label=f"{ylabel}={target:g}")
    ax.legend(loc="best", fontsize=7)


def plot_campaign(campaign: dict[str, Any], outfile: str | Path) -> Path:
    """BER (top) and BLER (bottom) versus SNR, one column per IoT value."""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg = campaign["config"]
    iot_levels = list(campaign["iot"])
    fig, axes = plt.subplots(2, len(iot_levels), figsize=(5.2 * len(iot_levels), 8.4), squeeze=False)
    for col, iot in enumerate(iot_levels):
        curves = campaign["iot"][iot]
        for row, metric in enumerate(("ber", "bler")):
            target = cfg.get(TARGET_KEY[metric], DEFAULT_TARGET[metric])
            _plot_metric(axes[row, col], curves, metric, mark_wp=True)
            _style_axis(axes[row, col], metric.upper(), iot, target)
    fig.suptitle(_title(cfg), fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    return outfile


def plot_working_points(campaign: dict[str, Any], outfile: str | Path, metric: str = "ber") -> Path:
    """Bar chart of the working points of every curve, grouped by IoT value."""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg = campaign["config"]
    iot_levels = list(campaign["iot"])
    names = _curve_names(campaign)
    x = np.arange(len(iot_levels))
    n_rx = max(len(names), 1)
    width = 0.8 / n_rx
    fig, ax = plt.subplots(figsize=(max(6.4, 1.8 * len(iot_levels)), 4.8))
    for i, name in enumerate(names):
        style = _style_for(name)
        values = []
        for iot in iot_levels:
            wp = campaign["iot"][iot].get(name, {}).get(WP_KEY[metric])
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
    target = cfg.get(TARGET_KEY[metric], DEFAULT_TARGET[metric])
    ax.set_title(f"SNR where {metric.upper()} = {target:g}")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=7)
    fig.suptitle(_title(cfg), fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    return outfile


def save_campaign_plots(campaign: dict[str, Any], outdir: str | Path, stem: str) -> list[Path]:
    """Write the curve figure and both working-point figures; returns their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return [
        plot_campaign(campaign, outdir / f"{stem}.png"),
        plot_working_points(campaign, outdir / f"{stem}_working_points.png"),
        plot_working_points(campaign, outdir / f"{stem}_working_points_bler.png", metric="bler"),
    ]
