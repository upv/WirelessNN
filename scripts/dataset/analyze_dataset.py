#!/usr/bin/env python3
"""Summarize a working-point dataset:  .venv/bin/python scripts/dataset/analyze_dataset.py dataset/main

Prints label coverage and the physical gaps the simulator should reproduce (modulation,
channel estimation, IRC vs L-MMSE), fits a ridge baseline on the scalar features so a
model trained on the raw tensors has something to beat, and writes an overview figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from nr_ul_sim.dataset import load_dataset

TARGET_PRIORITY = ("ls_lin", "irc", "lmmse", "perfect")


def names(d) -> list[str]:
    return [str(r) for r in d["receivers"]]


def pick_target(d) -> str:
    have = set(names(d))
    for cand in TARGET_PRIORITY:
        if cand in have:
            return cand
    return names(d)[0]


def label(d, receiver: str) -> np.ndarray:
    return d[f"wp_{receiver}"]


def describe(d) -> list[str]:
    lines = [
        f"records                {len(d['scenario_id'])}",
        f"distinct channels      {len(np.unique(d['scenario_id']))}",
        f"channel tensor         {d['H'].shape[1:]} complex64",
    ]
    for code, name in enumerate(d["channel_names"]):
        n = int((d["channel_code"] == code).sum())
        lines.append(f"  {str(name):6s}               {n:5d} records")
    if "feature_names" in d:
        lines.append(f"features               {', '.join(str(n) for n in d['feature_names'])}")
    lines.append("")
    lines.append(f"{'label':16s} {'reached':>9s} {'mean':>8s} {'std':>7s} {'p5':>7s} {'p95':>7s}")
    for receiver in names(d):
        wp = label(d, receiver)
        ok = wp[np.isfinite(wp)]
        if not ok.size:
            lines.append(f"{receiver:16s}        0%")
            continue
        lines.append(
            f"{receiver:16s} {ok.size / wp.size:8.0%} {ok.mean():8.1f} {ok.std():7.1f} "
            f"{np.percentile(ok, 5):7.1f} {np.percentile(ok, 95):7.1f}"
        )
    return lines


def _mod_name(d, code: int) -> str:
    mods = [str(m) for m in d["modulation_names"]]
    return mods[code] if 0 <= code < len(mods) else f"mod{code}"


def _paired_gap(d, target: str, lo_mod: int, hi_mod: int) -> list[float]:
    gaps = []
    for scenario in np.unique(d["scenario_id"]):
        m = d["scenario_id"] == scenario
        lo = label(d, target)[m & (d["modulation_code"] == lo_mod)]
        hi = label(d, target)[m & (d["modulation_code"] == hi_mod)]
        if lo.size and hi.size and np.isfinite(lo[0]) and np.isfinite(hi[0]):
            gaps.append(hi[0] - lo[0])
    return gaps


def gaps(d, target: str) -> list[str]:
    """Physical sanity checks: each gap has a known sign and rough magnitude."""
    have = set(names(d))
    lines = []
    mods = sorted(int(c) for c in np.unique(d["modulation_code"]))
    for lo_mod, hi_mod in zip(mods, mods[1:]):
        paired = _paired_gap(d, target, lo_mod, hi_mod)
        if paired:
            lines.append(
                f"{_mod_name(d, hi_mod):6s} - {_mod_name(d, lo_mod):6s}  "
                f"{np.median(paired):+6.1f} dB (median over channels)"
            )

    if "perfect" in have:
        for name in names(d):
            if name == "perfect":
                continue
            cost = label(d, name) - label(d, "perfect")
            lines.append(
                f"{name:16s} - perfect   {np.nanmedian(cost):+6.1f} dB (cost of CSI)"
            )
    if "lmmse" in have and "ideal_mmse" in have:
        csi = label(d, "lmmse") - label(d, "ideal_mmse")
        lines.append(f"L-MMSE - ideal MMSE    {np.nanmedian(csi):+6.1f} dB (cost of DMRS-LS CSI)")
    if "lmmse" in have and "irc" in have:
        irc = label(d, "lmmse") - label(d, "irc")
        for lo, hi in ((0.0, 1.0), (1.0, 8.0), (8.0, 14.0), (14.0, 20.0)):
            m = (d["iot_db"] >= lo) & (d["iot_db"] < hi) & np.isfinite(irc)
            if m.sum():
                lines.append(
                    f"L-MMSE - IRC, IoT {lo:4.0f}-{hi:<4.0f} {np.median(irc[m]):+6.1f} dB "
                    f"({int(m.sum())} records)"
                )
    return lines


def design_matrix(d) -> tuple[np.ndarray, list[str]]:
    delay = d["delay_spread_ns"].copy()
    delay[~np.isfinite(delay)] = np.nanmedian(delay)
    columns = [d["features"][:, i] for i in range(d["features"].shape[1])]
    names = [str(n) for n in d["feature_names"]]
    for name, value in (
        ("iot_db", d["iot_db"]),
        ("iot_db^2", d["iot_db"] ** 2),
        ("num_layers_total", d["num_layers_total"]),
        ("num_ue", d["num_ue"]),
        ("rank", d["rank"]),
        ("speed_kmh", d["speed_kmh"]),
        ("delay_spread_ns", delay),
        ("num_interferers", d["num_interferers"]),
        ("bits_per_symbol", d["num_bits_per_symbol"]),
        ("target_coderate", d["target_coderate"]),
    ):
        columns.append(np.asarray(value, dtype=float))
        names.append(name)
    for code, name in enumerate(d["channel_names"]):
        columns.append((d["channel_code"] == code).astype(float))
        names.append(f"is_{name}")
    return np.column_stack(columns), names


def ridge_baseline(d, receiver: str, alpha: float = 1.0, seed: int = 0) -> dict:
    """Least-squares on the scalar features only, split by channel to avoid leakage."""
    x, names = design_matrix(d)
    y = label(d, receiver)
    keep = np.isfinite(y)
    x, y, scenario = x[keep], y[keep], d["scenario_id"][keep]

    channels = np.unique(scenario)
    rng = np.random.default_rng(seed)
    test_channels = set(rng.choice(channels, max(1, len(channels) // 5), replace=False).tolist())
    test = np.array([s in test_channels for s in scenario])

    mean, std = x[~test].mean(0), x[~test].std(0)
    std[std == 0] = 1.0
    xs = np.column_stack([(x - mean) / std, np.ones(len(x))])
    a = xs[~test].T @ xs[~test] + alpha * np.eye(xs.shape[1])
    w = np.linalg.solve(a, xs[~test].T @ y[~test])

    pred = xs[test] @ w
    resid = pred - y[test]
    var = ((y[test] - y[~test].mean()) ** 2).mean()
    order = np.argsort(-np.abs(w[:-1]))
    return {
        "receiver": receiver,
        "n_train": int((~test).sum()),
        "n_test": int(test.sum()),
        "rmse": float(np.sqrt((resid**2).mean())),
        "rmse_mean_predictor": float(np.sqrt(var)),
        "r2": float(1.0 - (resid**2).mean() / var),
        "top_weights": [(names[i], float(w[i])) for i in order[:6]],
        "y_test": y[test],
        "pred": pred,
    }


def figure(d, baseline: dict, path: Path, target: str) -> None:
    mods = [str(m) for m in d["modulation_names"]]
    chans = [str(c) for c in d["channel_names"]]
    feat = {str(n): d["features"][:, i] for i, n in enumerate(d["feature_names"])}
    have = set(names(d))
    wp = label(d, target)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    ax = axes[0, 0]
    for code, mod in enumerate(mods):
        values = wp[(d["modulation_code"] == code) & np.isfinite(wp)]
        if values.size:
            ax.hist(values, bins=40, alpha=0.6, label=f"{mod} (n={values.size})")
    ax.set_xlabel(f"working point, {target} [dB]")
    ax.set_ylabel("records")
    ax.set_title("Label distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if "perfect" in have and len(have) > 1:
        for name in names(d):
            if name == "perfect":
                continue
            cost = label(d, name) - label(d, "perfect")
            m = np.isfinite(cost)
            ax.scatter(d["iot_db"][m], cost[m], s=8, alpha=0.45, label=name)
        ax.set_xlabel("IoT [dB]")
        ax.set_ylabel("WP − perfect CSI [dB]")
        ax.set_title("Cost of channel estimation")
        ax.legend(fontsize=7)
    elif "lmmse" in have and "irc" in have:
        gain = label(d, "lmmse") - label(d, "irc")
        m = np.isfinite(gain)
        sc = ax.scatter(d["iot_db"][m], gain[m], c=d["num_layers_total"][m], cmap="viridis",
                        s=8, alpha=0.5)
        ax.set_xlabel("IoT [dB]")
        ax.set_ylabel("L-MMSE - IRC [dB]")
        ax.set_title("IRC gain vs other-cell interference")
        plt.colorbar(sc, ax=ax, label="total layers")
    else:
        ax.axis("off")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    data, labels = [], []
    for code, chan in enumerate(chans):
        for mcode, mod in enumerate(mods):
            values = wp[(d["channel_code"] == code) & (d["modulation_code"] == mcode)]
            values = values[np.isfinite(values)]
            if values.size:
                data.append(values)
                labels.append(f"{chan}\n{mod}")
    if data:
        ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel(f"working point, {target} [dB]")
    ax.set_title("Spread per channel family and modulation")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    x, cols = design_matrix(d)
    corr = []
    for i in range(x.shape[1]):
        m = np.isfinite(wp) & np.isfinite(x[:, i])
        c = np.corrcoef(x[m, i], wp[m])[0, 1] if x[m, i].std() > 0 else 0.0
        corr.append(0.0 if np.isnan(c) else c)
    order = np.argsort(-np.abs(corr))[:12][::-1]
    ax.barh([cols[i] for i in order], [corr[i] for i in order])
    ax.set_xlabel("correlation with working point")
    ax.set_title("Scalar predictors of the label")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 1]
    for code, mod in enumerate(mods):
        m = (d["modulation_code"] == code) & np.isfinite(wp)
        qual = feat.get("capacity_bpcu_0db", feat.get("h_rms", feat.get("gain_db")))
        ax.scatter(qual[m], wp[m], s=8, alpha=0.4, label=mod)
    ax.set_xlabel("channel capacity at 0 dB [bit/s/Hz]")
    ax.set_ylabel(f"working point, {target} [dB]")
    ax.set_title("Label vs channel quality")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.scatter(baseline["y_test"], baseline["pred"], s=10, alpha=0.4)
    lims = [min(baseline["y_test"].min(), baseline["pred"].min()),
            max(baseline["y_test"].max(), baseline["pred"].max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("true working point [dB]")
    ax.set_ylabel("ridge prediction [dB]")
    ax.set_title(
        f"Scalar-feature baseline, held-out channels\n"
        f"RMSE {baseline['rmse']:.2f} dB, R2 {baseline['r2']:.2f}"
    )
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"{len(d['scenario_id'])} records / {len(np.unique(d['scenario_id']))} random channels",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")


def main(argv: list[str]) -> int:
    outdir = Path(argv[1] if len(argv) > 1 else "dataset/main")
    d = load_dataset(outdir)

    print("\n".join(describe(d)))
    target = pick_target(d)
    print("\nPhysical gaps")
    print("\n".join("  " + line for line in gaps(d, target)))

    print("\nRidge baseline on scalar features only (80/20 split by channel)")
    print(f"{'label':16s} {'train':>6s} {'test':>6s} {'RMSE':>7s} {'mean pred':>10s} {'R2':>6s}")
    baselines = {}
    for receiver in names(d):
        b = ridge_baseline(d, receiver)
        baselines[receiver] = b
        print(f"{receiver:16s} {b['n_train']:6d} {b['n_test']:6d} {b['rmse']:7.2f} "
              f"{b['rmse_mean_predictor']:10.2f} {b['r2']:6.2f}")
    print("  strongest weights (" + target + "): " + ", ".join(
        f"{n} {w:+.2f}" for n, w in baselines[target]["top_weights"]))

    path = outdir / "analysis.png"
    figure(d, baselines[target], path, target)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
