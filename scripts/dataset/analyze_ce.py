#!/usr/bin/env python3
"""Channel-estimation study for a fixed IRC receiver.

    .venv/bin/python scripts/dataset/analyze_ce.py dataset/irc_ce100

Labels are working points per estimator (wp_perfect, wp_ls_nn, ...). The CE loss
of an estimator is its working point minus the perfect-CSI working point on the
same frozen channel and modulation. Writes ce_analysis.txt and ce_analysis.png.
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

REFERENCE = "perfect"
SCENARIO_COLUMNS = (
    "iot_db", "num_layers_total", "num_ue", "rank", "speed_kmh",
    "delay_spread_ns", "num_interferers", "num_bits_per_symbol",
)


def estimators(d) -> list[str]:
    return [str(r) for r in d["receivers"]]


def ce_loss(d, name: str) -> np.ndarray:
    return d[f"wp_{name}"] - d[f"wp_{REFERENCE}"]


def columns(d) -> dict[str, np.ndarray]:
    out = {str(n): d["features"][:, i].astype(float) for i, n in enumerate(d["feature_names"])}
    for name in SCENARIO_COLUMNS:
        out[name] = np.asarray(d[name], dtype=float)
    return {k: v for k, v in out.items() if np.nanstd(v) > 0}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def summary_table(d, ests: list[str]) -> list[str]:
    mods = [str(m) for m in d["modulation_names"]]
    lines = [f"{'estimator':16s} {'reached':>8s} {'median':>7s} {'mean':>6s} {'p90':>6s}  "
             + "  ".join(f"{m:>6s}" for m in mods)]
    for name in ests:
        wp = d[f"wp_{name}"]
        loss = ce_loss(d, name)
        ok = np.isfinite(loss)
        per_mod = []
        for code, _ in enumerate(mods):
            m = ok & (d["modulation_code"] == code)
            per_mod.append(f"{np.median(loss[m]):+6.2f}" if m.any() else f"{'--':>6s}")
        lines.append(
            f"{name:16s} {np.isfinite(wp).mean():8.0%} {np.median(loss[ok]):+7.2f} "
            f"{loss[ok].mean():+6.2f} {np.percentile(loss[ok], 90):+6.2f}  " + "  ".join(per_mod)
        )
    return lines


def unreached_table(d, ests: list[str]) -> list[str]:
    mods = [str(m) for m in d["modulation_names"]]
    lines = [f"{'estimator':16s} " + "  ".join(f"{m:>6s}" for m in mods)]
    for name in [REFERENCE] + ests:
        wp = d[f"wp_{name}"]
        cells = []
        for code, _ in enumerate(mods):
            m = d["modulation_code"] == code
            cells.append(f"{int((~np.isfinite(wp[m])).sum()):3d}/{int(m.sum()):<3d}")
        lines.append(f"{name:16s} " + "  ".join(f"{c:>6s}" for c in cells))
    return lines


def grouped(d, ests: list[str], key: str, bins) -> list[str]:
    values = np.asarray(d[key], dtype=float)
    lines = [f"{key:18s} " + " ".join(f"{e[:10]:>10s}" for e in ests) + "      n"]
    for lo, hi, label in bins:
        m = (values >= lo) & (values < hi)
        cells = []
        for name in ests:
            loss = ce_loss(d, name)[m]
            loss = loss[np.isfinite(loss)]
            cells.append(f"{np.median(loss):+10.2f}" if loss.size else f"{'--':>10s}")
        lines.append(f"{label:18s} " + " ".join(cells) + f" {int(m.sum()):6d}")
    return lines


def by_channel(d, ests: list[str]) -> list[str]:
    lines = [f"{'channel':18s} " + " ".join(f"{e[:10]:>10s}" for e in ests) + "      n"]
    for code, chan in enumerate(str(c) for c in d["channel_names"]):
        m = d["channel_code"] == code
        cells = []
        for name in ests:
            loss = ce_loss(d, name)[m]
            loss = loss[np.isfinite(loss)]
            cells.append(f"{np.median(loss):+10.2f}" if loss.size else f"{'--':>10s}")
        lines.append(f"{chan:18s} " + " ".join(cells) + f" {int(m.sum()):6d}")
    return lines


def ranking(d, ests: list[str]) -> list[str]:
    wp = np.column_stack([d[f"wp_{e}"] for e in ests])
    wp = np.where(np.isfinite(wp), wp, np.inf)
    valid = np.isfinite(wp).any(axis=1)
    best = np.argmin(wp[valid], axis=1)
    lines = []
    for i, name in enumerate(ests):
        lines.append(f"  {name:16s} best in {np.mean(best == i):5.0%} of records")
    order_ok = np.all(np.diff(wp[valid], axis=1) <= 0.5, axis=1)
    lines.append(f"  records where WP is monotone along {' > '.join(ests)} (±0.5 dB): "
                 f"{order_ok.mean():.0%}")
    return lines


def feature_correlations(d, ests: list[str], top: int = 10) -> tuple[list[str], dict]:
    cols = columns(d)
    table = {name: {c: spearman(v, ce_loss(d, name)) for c, v in cols.items()} for name in ests}
    table["wp_perfect"] = {c: spearman(v, d[f"wp_{REFERENCE}"]) for c, v in cols.items()}
    focus = "ls_lin" if "ls_lin" in ests else ests[0]
    order = sorted(cols, key=lambda c: -abs(np.nan_to_num(table[focus][c])))
    heads = ["wp_perfect"] + ests
    lines = [f"{'feature':20s} " + " ".join(f"{h[:10]:>10s}" for h in heads)]
    for c in order[:top]:
        lines.append(f"{c:20s} " + " ".join(f"{table[h][c]:+10.2f}" for h in heads))
    return lines, table


def ridge_on_loss(d, name: str, alpha: float = 3.0, seed: int = 0) -> dict:
    """Can channel features predict the CE loss? Split by channel id."""
    cols = columns(d)
    keys = [k for k in cols if k not in ("num_bits_per_symbol",)]
    x = np.column_stack([cols[k] for k in keys] + [
        (d["modulation_code"] == c).astype(float) for c in range(len(d["modulation_names"]))
    ])
    x = np.where(np.isfinite(x), x, np.nanmedian(x, axis=0))
    y = ce_loss(d, name)
    keep = np.isfinite(y)
    x, y, sid = x[keep], y[keep], d["scenario_id"][keep]
    rng = np.random.default_rng(seed)
    uniq = np.unique(sid)
    test_ids = set(rng.choice(uniq, max(1, len(uniq) // 5), replace=False).tolist())
    test = np.array([s in test_ids for s in sid])
    mu, sd = x[~test].mean(0), x[~test].std(0)
    sd[sd == 0] = 1
    xs = np.column_stack([(x - mu) / sd, np.ones(len(x))])
    a = xs[~test].T @ xs[~test] + alpha * np.eye(xs.shape[1])
    w = np.linalg.solve(a, xs[~test].T @ y[~test])
    resid = xs[test] @ w - y[test]
    var = np.mean((y[test] - y[~test].mean()) ** 2)
    names = keys + [f"is_{m}" for m in d["modulation_names"]]
    order = np.argsort(-np.abs(w[:-1]))[:6]
    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "std": float(np.sqrt(var)),
        "r2": float(1 - np.mean(resid**2) / var) if var > 0 else 0.0,
        "top": [(names[i], float(w[i])) for i in order],
    }


def figure(d, ests: list[str], corr: dict, path: Path) -> None:
    mods = [str(m) for m in d["modulation_names"]]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    width = 0.8 / len(mods)
    x = np.arange(len(ests))
    for i, mod in enumerate(mods):
        med = []
        for name in ests:
            loss = ce_loss(d, name)[d["modulation_code"] == i]
            loss = loss[np.isfinite(loss)]
            med.append(np.median(loss) if loss.size else np.nan)
        ax.bar(x + (i - (len(mods) - 1) / 2) * width, med, width, label=mod)
    ax.set_xticks(x)
    ax.set_xticklabels(ests, rotation=15)
    ax.set_ylabel("median WP − perfect [dB]")
    ax.set_title("CE loss per estimator and modulation")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    data = [ce_loss(d, e)[np.isfinite(ce_loss(d, e))] for e in ests]
    ax.boxplot(data, tick_labels=ests, showfliers=True)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("WP − perfect [dB]")
    ax.set_title("CE loss distribution (all records)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 2]
    ref = d[f"wp_{REFERENCE}"]
    for name in ests:
        loss = ce_loss(d, name)
        m = np.isfinite(loss)
        ax.scatter(ref[m], loss[m], s=9, alpha=0.5, label=name)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("perfect-CSI working point [dB]")
    ax.set_ylabel("WP − perfect [dB]")
    ax.set_title("CE loss vs operating SNR")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    layers = d["num_layers_total"]
    for name in ests:
        loss = ce_loss(d, name)
        xs, ys = [], []
        for n in np.unique(layers):
            m = (layers == n) & np.isfinite(loss)
            if m.any():
                xs.append(n)
                ys.append(np.median(loss[m]))
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("total layers (UE × rank)")
    ax.set_ylabel("median WP − perfect [dB]")
    ax.set_title("CE loss vs number of layers")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    focus = "ls_lin" if "ls_lin" in ests else ests[0]
    feats = sorted(corr[focus], key=lambda c: abs(np.nan_to_num(corr[focus][c])))[-12:]
    heads = ["wp_perfect"] + ests
    mat = np.array([[corr[h][f] for h in heads] for f in feats])
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels(heads, rotation=30, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Spearman ρ: feature vs WP / CE loss")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    cols = columns(d)
    key = "rms_delay_samp" if "rms_delay_samp" in cols else next(iter(cols))
    for name in ests:
        loss = ce_loss(d, name)
        m = np.isfinite(loss)
        ax.scatter(cols[key][m], loss[m], s=9, alpha=0.5, label=name)
    ax.set_xlabel(key)
    ax.set_ylabel("WP − perfect [dB]")
    ax.set_title(f"CE loss vs {key}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"IRC, CE study: {len(np.unique(d['scenario_id']))} channels, "
        f"{len(d['scenario_id'])} records", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")


def main(argv: list[str]) -> int:
    outdir = Path(argv[1] if len(argv) > 1 else "dataset/irc_ce100")
    d = load_dataset(outdir)
    ests = [e for e in estimators(d) if e != REFERENCE]
    if REFERENCE not in estimators(d):
        raise SystemExit(f"{outdir} has no '{REFERENCE}' label")

    out = [
        f"IRC channel-estimation study  ({outdir})",
        f"{len(np.unique(d['scenario_id']))} channels, {len(d['scenario_id'])} records",
        "",
        "CE loss = WP(estimator) − WP(perfect CSI), dB, same frozen channel",
        *summary_table(d, ests),
        "",
        "Working point not reached (BER=1e-2 above snr_max) per modulation",
        *unreached_table(d, ests),
        "",
        "Median CE loss by IoT",
        *grouped(d, ests, "iot_db", [(0, 0.5, "no IoT"), (0.5, 8, "0-8 dB"),
                                     (8, 14, "8-14 dB"), (14, 21, "14-20 dB")]),
        "",
        "Median CE loss by total layers",
        *grouped(d, ests, "num_layers_total",
                 [(n, n + 1, f"{n} layer(s)") for n in range(1, 5)]),
        "",
        "Median CE loss by speed",
        *grouped(d, ests, "speed_kmh", [(0, 5, "3-5 km/h"), (5, 7.5, "5-7.5 km/h"),
                                        (7.5, 11, "7.5-10 km/h")]),
        "",
        "Median CE loss by channel family",
        *by_channel(d, ests),
        "",
        "Estimator ranking per record",
        *ranking(d, ests),
        "",
        "Spearman ρ of features with perfect-CSI WP and with CE loss (top by ls_lin)",
    ]
    corr_lines, corr = feature_correlations(d, ests)
    out += corr_lines
    out += ["", "Ridge on CE loss (scalar features + modulation, 80/20 split by channel)"]
    for name in ests:
        r = ridge_on_loss(d, name)
        out.append(
            f"  {name:16s} RMSE {r['rmse']:.2f} dB (std {r['std']:.2f})  R² {r['r2']:+.2f}  "
            + ", ".join(f"{n} {w:+.2f}" for n, w in r["top"][:4])
        )

    text = "\n".join(out)
    print(text)
    (outdir / "ce_analysis.txt").write_text(text + "\n")
    figure(d, ests, corr, outdir / "ce_analysis.png")
    print(f"\nWrote {outdir}/ce_analysis.txt, ce_analysis.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
