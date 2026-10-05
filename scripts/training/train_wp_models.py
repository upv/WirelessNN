#!/usr/bin/env python3
"""Benchmark regressors that predict the working point [dB] from scalar features.

    .venv/bin/python scripts/training/train_wp_models.py dataset/irc_ce10k models/wp_bench

The working point *is* an SNR (the SNR where BER crosses target_ber), so it is
the target, never an input. Scoring is 5-fold grouped by channel: the three
modulation rows of one channel realisation never straddle a fold boundary.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from nr_ul_sim.dataset import load_dataset

# Config knobs that never vary in this dataset carry no information.
CONSTANT_KEYS = ("carrier_frequency_hz", "target_ber", "num_prb", "mcs_table", "num_rx_ant")

# Scenario knobs: known before any transmission, no channel tensor needed.
SCENARIO_KEYS = (
    "iot_db",
    "speed_kmh",
    "delay_spread_ns",
    "num_ue",
    "rank",
    "num_layers_total",
    "num_ue_ant",
    "num_interferers",
    "num_bits_per_symbol",
    "target_coderate",
    "tb_size",
)

# Channel-tensor summaries (columns of data["features"]) that duplicate a
# scenario knob exactly; dropping them keeps the two feature sets disjoint.
DROP_FEATURE_COLS = ("num_rx_ant",)

ALPHA_GRID = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)
N_SPLITS = 5
SEED = 0


def build_features(data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (X, names, n_scenario_cols) with scenario columns first."""
    cols: list[np.ndarray] = []
    names: list[str] = []

    for key in SCENARIO_KEYS:
        v = np.asarray(data[key], dtype=float)
        if not np.isfinite(v).all():
            # delay_spread_ns is a CDL-only knob; UMi/UMa rows have no value.
            fill = float(np.median(v[np.isfinite(v)]))
            v = np.where(np.isfinite(v), v, fill)
        cols.append(v)
        names.append(key)

    cols.append(np.asarray(data["iot_db"], dtype=float) ** 2)
    names.append("iot_db^2")

    for code, name in enumerate(data["channel_names"]):
        cols.append((data["channel_code"] == code).astype(float))
        names.append(f"is_{name}")
    for code, name in enumerate(data["modulation_names"]):
        cols.append((data["modulation_code"] == code).astype(float))
        names.append(f"is_{name}")

    n_scenario = len(cols)

    feat_names = [str(n) for n in data["feature_names"]]
    for i, name in enumerate(feat_names):
        if name in DROP_FEATURE_COLS:
            continue
        cols.append(np.asarray(data["features"][:, i], dtype=float))
        names.append(name)

    return np.column_stack(cols), names, n_scenario


def make_models() -> dict[str, object]:
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHA_GRID)),
        "ridge_poly2": make_pipeline(
            PolynomialFeatures(2, include_bias=False),
            StandardScaler(),
            RidgeCV(alphas=ALPHA_GRID),
        ),
        "knn_25": make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=-1)
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "hist_gbdt": HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
            early_stopping=True, validation_fraction=0.15, random_state=SEED,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=1200, learning_rate=0.04, num_leaves=63, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1,
        ),
        "mlp_256_128": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-4,
                learning_rate_init=1e-3, batch_size=256, max_iter=400,
                early_stopping=True, n_iter_no_change=20, random_state=SEED,
            ),
        ),
    }


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    resid = pred - y
    var = float(np.mean((y - y.mean()) ** 2))
    return {
        "n": int(y.size),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1.0 - np.mean(resid**2) / var) if var > 0 else 0.0,
        "bias": float(resid.mean()),
        "p90_abs": float(np.percentile(np.abs(resid), 90)),
        "within_1db": float(np.mean(np.abs(resid) <= 1.0)),
        "within_2db": float(np.mean(np.abs(resid) <= 2.0)),
    }


def oof_predict(model, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, float]:
    """Out-of-fold predictions, folds grouped by channel realisation."""
    pred = np.full(y.shape, np.nan)
    cv = GroupKFold(n_splits=N_SPLITS)
    t0 = time.perf_counter()
    for train_idx, test_idx in cv.split(x, y, groups):
        est = clone(model)
        est.fit(x[train_idx], y[train_idx])
        pred[test_idx] = est.predict(x[test_idx])
    return pred, time.perf_counter() - t0


def main(argv: list[str]) -> int:
    src = Path(argv[1] if len(argv) > 1 else "dataset/irc_ce10k")
    dst = Path(argv[2] if len(argv) > 2 else "models/wp_bench")
    dst.mkdir(parents=True, exist_ok=True)

    data = load_dataset(src)
    x_all, names, n_scenario = build_features(data)
    groups = np.asarray(data["scenario_id"])
    estimators = [str(r) for r in data["estimator_names"]]
    mod_names = [str(m) for m in data["modulation_names"]]
    chan_names = [str(c) for c in data["channel_names"]]

    views = {
        "scenario": (np.arange(n_scenario), names[:n_scenario]),
        "scenario+H": (np.arange(x_all.shape[1]), names),
    }

    log: list[str] = [
        f"Working-point regression benchmark   ({src})",
        f"{len(np.unique(groups))} channel realisations, {x_all.shape[0]} records, "
        f"{len(estimators)} channel estimators",
        f"scoring: {N_SPLITS}-fold GroupKFold grouped by channel realisation "
        f"(all {len(mod_names)} modulations of a channel stay in the same fold)",
        "",
        "The working point is the SNR at BER=1e-2, so SNR is the target, not a feature.",
        f"Records where the WP was never reached below snr_max are dropped per target.",
        "",
        f"feature sets:  scenario = {n_scenario} cols (config knobs only)   "
        f"scenario+H = {x_all.shape[1]} cols (adds {x_all.shape[1] - n_scenario} channel-tensor summaries)",
        "",
    ]
    print("\n".join(log))

    results: dict[str, dict] = {}
    oof_store: dict[str, np.ndarray] = {}

    # --- full model sweep on two reference targets --------------------------- #
    sweep_targets = ["perfect", "lmmse_ce"]
    for target in sweep_targets:
        y_all = np.asarray(data[f"wp_{target}"], dtype=float)
        keep = np.isfinite(y_all)
        y = y_all[keep]
        g = groups[keep]
        header = (
            f"=== target wp_{target}   n={keep.sum()} of {keep.size} "
            f"({100 * (1 - keep.mean()):.1f}% not reached, dropped)   "
            f"std={y.std():.2f} dB ==="
        )
        print("\n" + header)
        log += ["", header,
                f"{'model':14s} {'features':12s} {'RMSE':>6s} {'MAE':>6s} {'R2':>6s} "
                f"{'p90|e|':>7s} {'<=1dB':>6s} {'<=2dB':>6s} {'fit s':>7s}"]

        base = metrics(y, np.full_like(y, y.mean()))
        line = (f"{'mean_baseline':14s} {'-':12s} {base['rmse']:6.2f} {base['mae']:6.2f} "
                f"{base['r2']:6.2f} {base['p90_abs']:7.2f} {base['within_1db']:6.2f} "
                f"{base['within_2db']:6.2f} {0.0:7.1f}")
        print(line)
        log.append(line)
        results[f"{target}|mean_baseline|-"] = base

        for view, (idx, _) in views.items():
            x = x_all[np.ix_(keep, idx)]
            for mname, model in make_models().items():
                pred, secs = oof_predict(model, x, y, g)
                m = metrics(y, pred)
                results[f"{target}|{mname}|{view}"] = m | {"fit_seconds": secs}
                oof_store[f"{target}|{mname}|{view}"] = pred
                line = (f"{mname:14s} {view:12s} {m['rmse']:6.2f} {m['mae']:6.2f} "
                        f"{m['r2']:6.2f} {m['p90_abs']:7.2f} {m['within_1db']:6.2f} "
                        f"{m['within_2db']:6.2f} {secs:7.1f}")
                print(line, flush=True)
                log.append(line)

    # --- pick the winner and run it on every estimator ----------------------- #
    best_key = min(
        (k for k in results if k.startswith("lmmse_ce|") and k.endswith("|scenario+H")),
        key=lambda k: results[k]["rmse"],
    )
    best_model_name = best_key.split("|")[1]
    print(f"\nbest model on wp_lmmse_ce (scenario+H): {best_model_name}")

    log += ["", f"=== {best_model_name} (scenario+H) on every channel estimator ===",
            f"{'estimator':18s} {'n':>6s} {'drop%':>6s} {'std':>6s} {'RMSE':>6s} {'MAE':>6s} "
            f"{'R2':>6s} {'p90|e|':>7s} {'<=1dB':>6s} {'<=2dB':>6s}"]
    per_estimator: dict[str, dict] = {}
    for est_name in estimators:
        y_all = np.asarray(data[f"wp_{est_name}"], dtype=float)
        keep = np.isfinite(y_all)
        y, g = y_all[keep], groups[keep]
        x = x_all[keep]
        pred, _ = oof_predict(make_models()[best_model_name], x, y, g)
        m = metrics(y, pred)
        per_estimator[est_name] = m
        oof_store[f"ALL|{est_name}"] = pred
        line = (f"{est_name:18s} {keep.sum():6d} {100 * (1 - keep.mean()):6.1f} {y.std():6.2f} "
                f"{m['rmse']:6.2f} {m['mae']:6.2f} {m['r2']:6.2f} {m['p90_abs']:7.2f} "
                f"{m['within_1db']:6.2f} {m['within_2db']:6.2f}")
        print(line, flush=True)
        log.append(line)

    # --- error breakdown of the winner on wp_lmmse_ce ------------------------ #
    y_all = np.asarray(data["wp_lmmse_ce"], dtype=float)
    keep = np.isfinite(y_all)
    y, pred = y_all[keep], oof_store[best_key]
    resid = pred - y

    log += ["", f"=== {best_model_name} residual breakdown on wp_lmmse_ce ==="]
    log.append(f"{'group':18s} {'n':>6s} {'RMSE':>6s} {'bias':>6s}")
    for code, name in enumerate(mod_names):
        m = data["modulation_code"][keep] == code
        log.append(f"{name:18s} {m.sum():6d} {np.sqrt(np.mean(resid[m]**2)):6.2f} {resid[m].mean():6.2f}")
    for code, name in enumerate(chan_names):
        m = data["channel_code"][keep] == code
        log.append(f"{name:18s} {m.sum():6d} {np.sqrt(np.mean(resid[m]**2)):6.2f} {resid[m].mean():6.2f}")
    iot = data["iot_db"][keep]
    for lo, hi in ((-0.1, 0.1), (0.1, 8), (8, 14), (14, 20.1)):
        m = (iot >= lo) & (iot < hi)
        if m.any():
            label = "no IoT" if hi <= 0.1 else f"IoT {lo:g}-{hi:g} dB"
            log.append(f"{label:18s} {m.sum():6d} {np.sqrt(np.mean(resid[m]**2)):6.2f} {resid[m].mean():6.2f}")

    # --- permutation importance on one held-out fold ------------------------- #
    cv = GroupKFold(n_splits=N_SPLITS)
    tr, te = next(cv.split(x_all[keep], y, groups[keep]))
    imp_model = clone(make_models()[best_model_name])
    imp_model.fit(x_all[keep][tr], y[tr])
    imp = permutation_importance(
        imp_model, x_all[keep][te], y[te], n_repeats=5, random_state=SEED,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    order = np.argsort(imp.importances_mean)[::-1]
    log += ["", f"=== permutation importance, {best_model_name} on wp_lmmse_ce "
            f"(RMSE increase in dB when a column is shuffled) ==="]
    for i in order[:20]:
        log.append(f"  {names[i]:22s} {imp.importances_mean[i]:+6.3f} +- {imp.importances_std[i]:.3f}")

    # --- figure -------------------------------------------------------------- #
    figure(dst / "wp_models.png", results, per_estimator, names, imp, order,
           y, pred, resid, data, keep, mod_names, chan_names,
           best_model_name, sweep_targets, list(make_models()))

    text = "\n".join(log)
    (dst / "report.txt").write_text(text + "\n")
    (dst / "metrics.json").write_text(json.dumps(
        {"sweep": results, "per_estimator": per_estimator,
         "best_model": best_model_name, "feature_names": names,
         "n_scenario_cols": int(n_scenario),
         "permutation_importance": {names[i]: float(imp.importances_mean[i]) for i in order}},
        indent=2))
    print(f"\nWrote {dst}/report.txt, metrics.json, wp_models.png")
    return 0


def figure(out, results, per_estimator, names, imp, order, y, pred, resid,
           data, keep, mod_names, chan_names, best, sweep_targets, model_names) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))

    ax = axes[0, 0]
    width = 0.38
    pos = np.arange(len(model_names))
    for k, (view, off) in enumerate((("scenario", -width / 2), ("scenario+H", width / 2))):
        vals = [results[f"lmmse_ce|{m}|{view}"]["rmse"] for m in model_names]
        ax.bar(pos + off, vals, width, label=view, color=f"C{k}")
        for p, v in zip(pos + off, vals):
            ax.text(p, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    base = results["lmmse_ce|mean_baseline|-"]["rmse"]
    ax.axhline(base, color="k", ls="--", lw=1, label=f"mean baseline {base:.2f}")
    ax.set_xticks(pos)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("out-of-fold RMSE [dB]")
    ax.set_title("wp_lmmse_ce: model x feature set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    for k, target in enumerate(sweep_targets):
        vals = [results[f"{target}|{m}|scenario+H"]["r2"] for m in model_names]
        ax.bar(pos + (k - 0.5) * width, vals, width, label=f"wp_{target}", color=f"C{k + 2}")
    ax.set_xticks(pos)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("out-of-fold R2")
    ax.set_title("scenario+H features, two targets")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 2]
    top = order[:15][::-1]
    ax.barh([names[i] for i in top], [imp.importances_mean[i] for i in top],
            xerr=[imp.importances_std[i] for i in top], color="C3")
    ax.set_xlabel("RMSE increase when shuffled [dB]")
    ax.set_title(f"{best}: permutation importance")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 0]
    for code, name in enumerate(mod_names):
        m = data["modulation_code"][keep] == code
        ax.scatter(y[m], pred[m], s=5, alpha=0.3, label=name, color=f"C{code}")
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    r = results_rmse(y, pred)
    ax.set_xlabel("true working point [dB]")
    ax.set_ylabel(f"{best} out-of-fold prediction [dB]")
    ax.set_title(f"wp_lmmse_ce   RMSE {r[0]:.2f} dB   R2 {r[1]:.2f}")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(data["iot_db"][keep], resid, s=5, alpha=0.25,
               c=data["modulation_code"][keep], cmap="viridis")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("IoT [dB]")
    ax.set_ylabel("prediction - true [dB]")
    ax.set_title("residual vs IoT (colour = modulation)")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    groups_box, labels = [], []
    for ccode, chan in enumerate(chan_names):
        for mcode, mod in enumerate(mod_names):
            m = (data["channel_code"][keep] == ccode) & (data["modulation_code"][keep] == mcode)
            if m.any():
                groups_box.append(resid[m])
                labels.append(f"{chan}\n{mod}")
    ax.boxplot(groups_box, tick_labels=labels, showfliers=False)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("prediction - true [dB]")
    ax.set_title("residual by channel x modulation")
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Working-point prediction benchmark  ·  best: {best}", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=125, bbox_inches="tight")


def results_rmse(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    resid = pred - y
    var = float(np.mean((y - y.mean()) ** 2))
    return float(np.sqrt(np.mean(resid**2))), float(1.0 - np.mean(resid**2) / var)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
