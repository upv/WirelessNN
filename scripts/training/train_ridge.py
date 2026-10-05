#!/usr/bin/env python3
"""Fit the scalar-feature ridge baseline and dump pipeline + analysis.

    .venv/bin/python scripts/training/train_ridge.py dataset/run5000 models/ridge
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from nr_ul_sim.dataset import load_dataset
from nr_ul_sim.ridge import RidgePipeline

TARGET = "lmmse"


def report(pipe: RidgePipeline, metrics: dict) -> str:
    lines = [
        "Ridge working-point model (scalar features only, no H tensor)",
        f"features: {len(pipe.feature_names)}  receivers: {', '.join(pipe.receivers)}",
        f"split: {pipe.split['n_train_channels']} train / "
        f"{pipe.split['n_test_channels']} test channels  seed={pipe.split['seed']}",
        f"alpha (inner CV): {pipe.alpha}",
        "",
        f"{'receiver':12s} {'train RMSE':>10s} {'test RMSE':>9s} {'test MAE':>8s} "
        f"{'R2':>6s} {'bias':>7s} {'p90|e|':>7s}  QPSK / 16QAM RMSE",
    ]
    for rx, rec in metrics["receivers"].items():
        q = rec.get("test_qpsk", {}).get("rmse")
        m = rec.get("test_qam16", {}).get("rmse")
        lines.append(
            f"{rx:12s} {rec['train']['rmse']:10.2f} {rec['test']['rmse']:9.2f} "
            f"{rec['test']['mae']:8.2f} {rec['test']['r2']:6.2f} "
            f"{rec['test']['bias']:7.2f} {rec['test']['p90_abs']:7.2f}  "
            f"{q:4.2f} / {m:4.2f}" if q is not None and m is not None else
            f"{rx:12s} {rec['train']['rmse']:10.2f} {rec['test']['rmse']:9.2f}"
        )
    lines.append("")
    lines.append("Largest |standardized| weights for L-MMSE (σ of feature → dB of WP):")
    rows = [r for r in pipe.coefficients() if r["receiver"] == TARGET and r["feature"] != "intercept"]
    rows.sort(key=lambda r: abs(r["weight_standardized"]), reverse=True)
    for row in rows[:10]:
        lines.append(
            f"  {row['feature']:22s}  std {row['weight_standardized']:+6.2f}   "
            f"per unit {row['weight_per_unit']:+7.3f}"
        )
    lines += [
        "",
        "Usage:",
        "  from nr_ul_sim.ridge import RidgePipeline",
        "  pipe = RidgePipeline.load('models/ridge')",
        "  wp_db = pipe.predict(data, 'lmmse')",
    ]
    return "\n".join(lines)


def figure(pipe: RidgePipeline, data: dict, out: Path) -> None:
    y = np.asarray(data[f"wp_{TARGET}"], dtype=float)
    pred = pipe.predict(data, TARGET)
    keep = np.isfinite(y)
    test_ids = set(pipe.split["test_scenario_ids"])
    test = np.array([s in test_ids for s in data["scenario_id"]]) & keep
    resid = pred[test] - y[test]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    rows = [r for r in pipe.coefficients() if r["receiver"] == TARGET and r["feature"] != "intercept"]
    rows.sort(key=lambda r: abs(r["weight_standardized"]))
    ax.barh([r["feature"] for r in rows], [r["weight_standardized"] for r in rows])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("standardized coefficient [dB / σ]")
    ax.set_title("L-MMSE coefficients")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[0, 1]
    for code, name, color in ((0, "qpsk", "C0"), (1, "qam16", "C1")):
        m = test & (data["modulation_code"] == code)
        ax.scatter(y[m], pred[m], s=8, alpha=0.35, c=color, label=name)
    lims = [min(y[test].min(), pred[test].min()), max(y[test].max(), pred[test].max())]
    ax.plot(lims, lims, "k--", lw=1)
    met = pipe.evaluate(data)["receivers"][TARGET]["test"]
    ax.set_xlabel("true working point [dB]")
    ax.set_ylabel("ridge prediction [dB]")
    ax.set_title(f"Held-out channels  RMSE {met['rmse']:.2f} dB  R² {met['r2']:.2f}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(data["iot_db"][test], resid, s=8, alpha=0.35, c=data["modulation_code"][test], cmap="coolwarm")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("IoT [dB]")
    ax.set_ylabel("prediction − true [dB]")
    ax.set_title("Residual vs IoT (test)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    groups, labels = [], []
    for ccode, chan in enumerate(pipe.channel_names):
        for mcode, mod in ((0, "qpsk"), (1, "qam16")):
            m = test & (data["channel_code"] == ccode) & (data["modulation_code"] == mcode)
            if m.any():
                groups.append(pred[m] - y[m])
                labels.append(f"{chan}\n{mod}")
    ax.boxplot(groups, tick_labels=labels)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("prediction − true [dB]")
    ax.set_title("Residual by channel family")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Ridge pipeline  ·  scalar features → working point", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")


def main(argv: list[str]) -> int:
    src = Path(argv[1] if len(argv) > 1 else "dataset/run5000")
    dst = Path(argv[2] if len(argv) > 2 else "models/ridge")
    data = load_dataset(src)
    pipe = RidgePipeline.fit(data)
    pipe.save(dst)
    metrics = pipe.evaluate(data)
    (dst / "metrics.json").write_text(json.dumps(metrics, indent=2))
    text = report(pipe, metrics)
    (dst / "analysis.txt").write_text(text + "\n")
    figure(pipe, data, dst / "analysis.png")
    print(text)
    print(f"\nWrote {dst}/pipeline.npz, coefficients.csv, metrics.json, analysis.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
