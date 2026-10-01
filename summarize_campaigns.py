#!/usr/bin/env python3
"""Collect the campaign JSONs written by run_campaigns.sh into tables and figures.

    .venv/bin/python summarize_campaigns.py /root/reports/x/campaigns
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nr_ul_sim.plotting import RECEIVER_STYLE

RECEIVERS = ["mr", "lmmse", "ideal_mmse", "zf", "irc"]
ESTIMATORS = ["perfect", "ls_nn", "ls_lin", "ls_lin_time_avg", "lmmse_ce", "lmmse_exp", "ls_hard_window", "ls_soft_window"]
CHANNELS = ["cdl-b", "cdl-c", "umi", "uma"]
MODS = ["qpsk", "qam16", "qam64"]


def load(root: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(root.glob("*/*/*.json")):
        out[path.parent.name] = json.loads(path.read_text())
    return out


def fmt(v) -> str:
    return "  n/r " if v is None else f"{v:6.1f}"


def receiver_tables(camps: dict, lines: list[str], summary: dict) -> None:
    lines.append("## Receivers: working point SNR [dB] (BER = 1e-2 / BLER = 0.1), 68 PRB, LS-linear CSI\n")
    for cfg_name, cfg_label in (("ue1_r1", "1 UE, rank 1"), ("ue2_r1", "2 UE, rank 1"), ("ue2_r2", "2 UE, rank 2")):
        lines.append(f"### {cfg_label}\n")
        for metric, key in (("BER=1e-2", "working_point_db"), ("BLER=0.1", "working_point_bler_db")):
            lines.append(f"**{metric}**\n")
            lines.append("| channel | mod | IoT | " + " | ".join(RECEIVERS) + " |")
            lines.append("|---|---|---|" + "---|" * len(RECEIVERS))
            for ch in CHANNELS:
                for mod in MODS:
                    camp = camps.get(f"rx_{ch}_{mod}_{cfg_name}")
                    if camp is None:
                        continue
                    for iot, curves in camp["iot"].items():
                        vals = [curves.get(r, {}).get(key) for r in RECEIVERS]
                        summary.setdefault("receivers", []).append(
                            {"config": cfg_name, "channel": ch, "modulation": mod, "iot_db": float(iot),
                             "metric": metric, **{r: v for r, v in zip(RECEIVERS, vals)}})
                        lines.append(f"| {ch} | {mod} | {float(iot):g} | " + " | ".join(fmt(v) for v in vals) + " |")
            lines.append("")

    # gains of IRC over L-MMSE and CE loss of L-MMSE vs Ideal MMSE
    gains = defaultdict(list)
    for row in summary.get("receivers", []):
        if row["metric"] != "BER=1e-2":
            continue
        if row["irc"] is not None and row["lmmse"] is not None:
            gains[("IRC gain over L-MMSE", row["iot_db"])].append(row["lmmse"] - row["irc"])
        if row["ideal_mmse"] is not None and row["lmmse"] is not None:
            gains[("L-MMSE loss vs Ideal MMSE (LS-linear CE)", row["iot_db"])].append(row["lmmse"] - row["ideal_mmse"])
        if row["zf"] is not None and row["lmmse"] is not None:
            gains[("ZF loss vs L-MMSE", row["iot_db"])].append(row["zf"] - row["lmmse"])
    lines.append("### Aggregates over channels, modulations and UE configs (BER = 1e-2)\n")
    lines.append("| quantity | IoT | median dB | mean dB | n |")
    lines.append("|---|---|---|---|---|")
    for (name, iot), vals in sorted(gains.items()):
        lines.append(f"| {name} | {iot:g} | {np.median(vals):.2f} | {np.mean(vals):.2f} | {len(vals)} |")
        summary.setdefault("receiver_aggregates", []).append(
            {"quantity": name, "iot_db": iot, "median_db": float(np.median(vals)), "mean_db": float(np.mean(vals)), "n": len(vals)})
    lines.append("")


def ce_tables(camps: dict, lines: list[str], summary: dict, outdir: Path) -> None:
    lines.append("## Channel estimation with IRC: working point SNR [dB] at BER = 1e-2, 68 PRB\n")
    lines.append("| channel | mod | UE | IoT | " + " | ".join(ESTIMATORS) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(ESTIMATORS))
    loss = defaultdict(list)
    for ch in CHANNELS:
        for mod in MODS:
            for nue in (1, 2):
                camp = camps.get(f"ce_{ch}_{mod}_ue{nue}")
                if camp is None:
                    continue
                for iot, curves in camp["iot"].items():
                    vals = [curves.get(e, {}).get("working_point_db") for e in ESTIMATORS]
                    lines.append(f"| {ch} | {mod} | {nue} | {float(iot):g} | " + " | ".join(fmt(v) for v in vals) + " |")
                    summary.setdefault("ce", []).append(
                        {"channel": ch, "modulation": mod, "num_ue": nue, "iot_db": float(iot),
                         **{e: v for e, v in zip(ESTIMATORS, vals)}})
                    perf = vals[0]
                    if perf is None:
                        continue
                    for e, v in zip(ESTIMATORS[1:], vals[1:]):
                        if v is not None:
                            loss[e].append(v - perf)
                            loss[(e, mod)].append(v - perf)
                            loss[(e, ch)].append(v - perf)
                            loss[(e, f"ue{nue}")].append(v - perf)
    lines.append("")
    lines.append("### CE loss = WP(estimator) - WP(perfect CSI), dB, IRC, 68 PRB\n")
    lines.append("| estimator | median | mean | p90 | n | qpsk | qam16 | qam64 | cdl-b | cdl-c | umi | uma | 1 UE | 2 UE |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in ESTIMATORS[1:]:
        v = np.asarray(loss[e])
        if v.size == 0:
            continue
        cells = [f"{np.median(loss[(e, k)]):.2f}" if loss[(e, k)] else "n/a"
                 for k in MODS + CHANNELS + ["ue1", "ue2"]]
        lines.append(f"| {e} | {np.median(v):.2f} | {v.mean():.2f} | {np.percentile(v, 90):.2f} | {v.size} | " + " | ".join(cells) + " |")
        summary.setdefault("ce_loss", {})[e] = {
            "median": float(np.median(v)), "mean": float(v.mean()), "p90": float(np.percentile(v, 90)), "n": int(v.size),
            **{k: float(np.median(loss[(e, k)])) for k in MODS + CHANNELS + ["ue1", "ue2"] if loss[(e, k)]}}
    lines.append("")

    # figure: CE loss box plot
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [loss[e] for e in ESTIMATORS[1:] if loss[e]]
    labels = [RECEIVER_STYLE[e]["label"] for e in ESTIMATORS[1:] if loss[e]]
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel("WP loss vs perfect CSI [dB]")
    ax.set_title("IRC working-point loss of each channel estimator (all channels / modulations / IoT)")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "ce_loss_boxplot.png", dpi=140)
    plt.close(fig)

    # speed study
    rows = []
    for sp in (3, 10, 30):
        camp = camps.get(f"speed_{sp}")
        if camp is None:
            continue
        curves = camp["iot"]["0.0"]
        rows.append({"speed_kmh": sp, **{e: curves.get(e, {}).get("working_point_db") for e in curves}})
    if rows:
        lines.append("### Speed (CDL-C, 16QAM, 1 UE, IRC), WP at BER = 1e-2\n")
        keys = [k for k in rows[0] if k != "speed_kmh"]
        lines.append("| speed | " + " | ".join(keys) + " |")
        lines.append("|---|" + "---|" * len(keys))
        for r in rows:
            lines.append(f"| {r['speed_kmh']} km/h | " + " | ".join(fmt(r[k]) for k in keys) + " |")
        lines.append("")
        summary["speed"] = rows


def cov_tables(camps: dict, lines: list[str], summary: dict) -> None:
    rows = []
    for cov in ("perfect", "residual", "estimated"):
        camp = camps.get(f"cov_{cov}")
        if camp is None:
            continue
        for iot, curves in camp["iot"].items():
            rows.append({"iot_cov": cov, "iot_db": float(iot),
                         "lmmse": curves["lmmse"]["working_point_db"], "irc": curves["irc"]["working_point_db"]})
    if rows:
        lines.append("### IRC interference-covariance estimators (CDL-C, 16QAM, 2 UE), WP at BER = 1e-2\n")
        lines.append("| R_iot method | IoT | L-MMSE | IRC |")
        lines.append("|---|---|---|---|")
        for r in rows:
            lines.append(f"| {r['iot_cov']} | {r['iot_db']:g} | {fmt(r['lmmse'])} | {fmt(r['irc'])} |")
        lines.append("")
        summary["iot_cov"] = rows


def overview_figure(camps: dict, outdir: Path) -> None:
    """BER curves of every receiver for 2 UE rank 1, all channels x modulations, IoT 10 dB."""
    fig, axes = plt.subplots(len(CHANNELS), len(MODS), figsize=(4.2 * len(MODS), 3.2 * len(CHANNELS)), sharey=True)
    for i, ch in enumerate(CHANNELS):
        for j, mod in enumerate(MODS):
            ax = axes[i, j]
            camp = camps.get(f"rx_{ch}_{mod}_ue2_r1")
            if camp is None:
                ax.set_visible(False)
                continue
            for iot, ls in (("0.0", "-"), ("10.0", "--"), ("20.0", ":")):
                curves = camp["iot"].get(iot, {})
                for r in RECEIVERS:
                    if r not in curves:
                        continue
                    st = RECEIVER_STYLE[r]
                    ax.semilogy(curves[r]["snr_db"], np.clip(curves[r]["ber"], 1e-6, 1), ls=ls, color=st["color"],
                                lw=1, label=f"{st['label']} IoT {float(iot):g}" if i == 0 and j == 0 else None)
            ax.axhline(1e-2, color="grey", ls=":", lw=0.8)
            ax.set_ylim(1e-5, 1)
            ax.set_title(f"{ch.upper()} {mod.upper()} 2 UE", fontsize=9)
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if i == len(CHANNELS) - 1:
                ax.set_xlabel("SNR per antenna [dB]")
    axes[0, 0].legend(fontsize=5, ncol=3)
    fig.suptitle("Coded BER, 68 PRB, 2 UE x rank 1, all receivers (solid IoT 0, dashed 10, dotted 20 dB)")
    fig.tight_layout()
    fig.savefig(outdir / "receivers_overview_ue2.png", dpi=130)
    plt.close(fig)


def main() -> int:
    root = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else root
    camps = load(root)
    print(f"{len(camps)} campaigns loaded")
    lines = [f"# Campaign summary ({len(camps)} campaigns)\n"]
    summary: dict = {"num_campaigns": len(camps)}
    receiver_tables(camps, lines, summary)
    ce_tables(camps, lines, summary, outdir)
    cov_tables(camps, lines, summary)
    overview_figure(camps, outdir)
    (outdir / "campaign_summary.md").write_text("\n".join(lines))
    (outdir / "campaign_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {outdir / 'campaign_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
