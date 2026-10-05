#!/usr/bin/env python3
"""Round two: stronger WP regressors plus a reachability classifier.

    .venv/bin/python scripts/training/train_wp_models2.py dataset/irc_ce10k models/wp_bench2

Round one showed a quadratic ridge edging out the boosted trees, so the models
here either widen that basis, boost its residual, or replace it with a net.
The classifier targets the records round one had to drop: those whose working
point never crosses the BER target below snr_max.
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
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from lightgbm import LGBMClassifier, LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from nr_ul_sim.dataset import load_dataset
from train_wp_models import ALPHA_GRID, N_SPLITS, SEED, build_features, metrics


class TorchMLP(BaseEstimator, RegressorMixin):
    """Plain residual-free MLP with standardised inputs and target."""

    def __init__(self, hidden=(512, 256, 128), epochs=300, lr=2e-3, batch_size=512,
                 weight_decay=1e-4, dropout=0.05, seed=SEED):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed

    def fit(self, x, y):
        torch.manual_seed(self.seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.x_mean_, self.x_std_ = x.mean(0), x.std(0)
        self.x_std_[self.x_std_ == 0] = 1.0
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std())

        xt = torch.from_numpy((x - self.x_mean_) / self.x_std_)
        yt = torch.from_numpy((y - self.y_mean_) / self.y_std_).unsqueeze(1)

        layers: list[nn.Module] = []
        prev = xt.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net_ = nn.Sequential(*layers)

        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=self.epochs * max(1, len(xt) // self.batch_size + 1))
        loss_fn = nn.HuberLoss(delta=1.0)
        gen = torch.Generator().manual_seed(self.seed)

        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(self.net_(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                sched.step()
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        xt = torch.from_numpy((x - self.x_mean_) / self.x_std_)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(xt).squeeze(1).numpy()
        return out * self.y_std_ + self.y_mean_


class BoostedRidge(BaseEstimator, RegressorMixin):
    """Quadratic ridge, then gradient-boosted trees on what it leaves behind."""

    def __init__(self, degree=2, max_iter=600, learning_rate=0.05, max_leaf_nodes=63, seed=SEED):
        self.degree = degree
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.seed = seed

    def fit(self, x, y):
        self.linear_ = make_pipeline(
            PolynomialFeatures(self.degree, include_bias=False),
            StandardScaler(),
            RidgeCV(alphas=ALPHA_GRID),
        ).fit(x, y)
        resid = y - self.linear_.predict(x)
        self.trees_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes, early_stopping=True,
            validation_fraction=0.15, random_state=self.seed,
        ).fit(x, resid)
        return self

    def predict(self, x):
        return self.linear_.predict(x) + self.trees_.predict(x)


class AverageBlend(BaseEstimator, RegressorMixin):
    """Unweighted mean of independently fitted models (no weights to overfit)."""

    def __init__(self, models=None):
        self.models = models

    def fit(self, x, y):
        self.fitted_ = [clone(m).fit(x, y) for _, m in self.models]
        return self

    def predict(self, x):
        return np.mean([m.predict(x) for m in self.fitted_], axis=0)


def round2_models() -> dict[str, object]:
    poly2 = make_pipeline(
        PolynomialFeatures(2, include_bias=False), StandardScaler(), RidgeCV(alphas=ALPHA_GRID)
    )
    lgbm_tuned = LGBMRegressor(
        n_estimators=3000, learning_rate=0.02, num_leaves=255, min_child_samples=10,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    mlp = TorchMLP()
    return {
        "ridge_poly2": poly2,
        "ridge_poly3": make_pipeline(
            PolynomialFeatures(3, include_bias=False), StandardScaler(),
            RidgeCV(alphas=ALPHA_GRID),
        ),
        "lgbm_tuned": lgbm_tuned,
        "mlp_torch": mlp,
        "boosted_ridge": BoostedRidge(),
        "blend_poly2_lgbm_mlp": AverageBlend(
            models=[("poly2", poly2), ("lgbm", lgbm_tuned), ("mlp", mlp)]
        ),
    }


def oof_predict(model, x, y, groups):
    pred = np.full(y.shape, np.nan)
    t0 = time.perf_counter()
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(x, y, groups):
        est = clone(model).fit(x[tr], y[tr])
        pred[te] = est.predict(x[te])
    return pred, time.perf_counter() - t0


def main(argv: list[str]) -> int:
    src = Path(argv[1] if len(argv) > 1 else "dataset/irc_ce10k")
    dst = Path(argv[2] if len(argv) > 2 else "models/wp_bench2")
    dst.mkdir(parents=True, exist_ok=True)

    data = load_dataset(src)
    x_all, names, _ = build_features(data)
    groups = np.asarray(data["scenario_id"])
    estimators = [str(r) for r in data["estimator_names"]]
    mod_names = [str(m) for m in data["modulation_names"]]

    log = [f"Working-point regression, round 2   ({src})",
           f"{x_all.shape[0]} records, {x_all.shape[1]} features (scenario+H), "
           f"{N_SPLITS}-fold GroupKFold by channel realisation", ""]
    print("\n".join(log))

    results: dict[str, dict] = {}
    oof: dict[str, np.ndarray] = {}

    y_all = np.asarray(data["wp_lmmse_ce"], dtype=float)
    keep = np.isfinite(y_all)
    y, g, x = y_all[keep], groups[keep], x_all[keep]

    head = (f"=== target wp_lmmse_ce   n={keep.sum()}   std={y.std():.2f} dB ===")
    print(head)
    log += [head, f"{'model':22s} {'RMSE':>6s} {'MAE':>6s} {'R2':>6s} {'p90|e|':>7s} "
            f"{'<=1dB':>6s} {'<=2dB':>6s} {'fit s':>7s}"]
    for mname, model in round2_models().items():
        pred, secs = oof_predict(model, x, y, g)
        m = metrics(y, pred)
        results[mname] = m | {"fit_seconds": secs}
        oof[mname] = pred
        line = (f"{mname:22s} {m['rmse']:6.2f} {m['mae']:6.2f} {m['r2']:6.2f} "
                f"{m['p90_abs']:7.2f} {m['within_1db']:6.2f} {m['within_2db']:6.2f} {secs:7.1f}")
        print(line, flush=True)
        log.append(line)

    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"\nbest round-2 model: {best_name}  RMSE {results[best_name]['rmse']:.2f} dB")

    # --- winner across all five estimators ----------------------------------- #
    log += ["", f"=== {best_name} on every channel estimator ===",
            f"{'estimator':18s} {'n':>6s} {'RMSE':>6s} {'MAE':>6s} {'R2':>6s} {'p90|e|':>7s} {'<=2dB':>6s}"]
    per_estimator = {}
    for est_name in estimators:
        ye = np.asarray(data[f"wp_{est_name}"], dtype=float)
        k = np.isfinite(ye)
        pred, _ = oof_predict(round2_models()[best_name], x_all[k], ye[k], groups[k])
        m = metrics(ye[k], pred)
        per_estimator[est_name] = m
        oof[f"EST|{est_name}"] = pred
        line = (f"{est_name:18s} {k.sum():6d} {m['rmse']:6.2f} {m['mae']:6.2f} {m['r2']:6.2f} "
                f"{m['p90_abs']:7.2f} {m['within_2db']:6.2f}")
        print(line, flush=True)
        log.append(line)

    # --- reachability: will the WP be reached below snr_max at all? ----------- #
    log += ["", "=== reachability classifier: P(working point never reached below snr_max) ===",
            f"{'estimator':18s} {'pos%':>6s} {'AUC':>6s} {'acc':>6s} {'brier':>6s}"]
    print("\n" + log[-2])
    reach = {}
    for est_name in estimators:
        label = (np.asarray(data[f"wp_status_{est_name}"]) == "not_reached").astype(int)
        if label.sum() < 50:
            log.append(f"{est_name:18s} {100 * label.mean():6.2f}   (too few positives)")
            continue
        prob = np.full(label.shape, np.nan)
        clf = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                             random_state=SEED, n_jobs=-1, verbose=-1)
        for tr, te in GroupKFold(n_splits=N_SPLITS).split(x_all, label, groups):
            prob[te] = clone(clf).fit(x_all[tr], label[tr]).predict_proba(x_all[te])[:, 1]
        auc = float(roc_auc_score(label, prob))
        acc = float(np.mean((prob > 0.5) == label))
        brier = float(np.mean((prob - label) ** 2))
        reach[est_name] = {"positive_rate": float(label.mean()), "auc": auc,
                           "accuracy": acc, "brier": brier}
        oof[f"REACH|{est_name}"] = prob
        line = f"{est_name:18s} {100 * label.mean():6.2f} {auc:6.3f} {acc:6.3f} {brier:6.3f}"
        print(line, flush=True)
        log.append(line)

    # --- how much of the remaining error is irreducible? --------------------- #
    pred_best = oof[best_name]
    resid = pred_best - y
    log += ["", f"=== {best_name} residual breakdown on wp_lmmse_ce ===",
            f"{'group':18s} {'n':>6s} {'RMSE':>6s} {'bias':>6s}"]
    for code, name in enumerate(mod_names):
        m = data["modulation_code"][keep] == code
        log.append(f"{name:18s} {m.sum():6d} {np.sqrt(np.mean(resid[m]**2)):6.2f} {resid[m].mean():6.2f}")
    for code, name in enumerate(data["channel_names"]):
        m = data["channel_code"][keep] == code
        log.append(f"{str(name):18s} {m.sum():6d} {np.sqrt(np.mean(resid[m]**2)):6.2f} {resid[m].mean():6.2f}")

    figure(dst / "wp_models2.png", results, per_estimator, reach, y, pred_best, resid,
           data, keep, mod_names, best_name, oof)

    text = "\n".join(log)
    (dst / "report.txt").write_text(text + "\n")
    (dst / "metrics.json").write_text(json.dumps(
        {"round2": results, "per_estimator": per_estimator, "reachability": reach,
         "best_model": best_name}, indent=2))
    np.savez_compressed(dst / "oof_predictions.npz", **oof)
    print(f"\nWrote {dst}/report.txt, metrics.json, wp_models2.png, oof_predictions.npz")
    return 0


def figure(out, results, per_estimator, reach, y, pred, resid, data, keep,
           mod_names, best, oof) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    model_names = list(results)

    ax = axes[0, 0]
    vals = [results[m]["rmse"] for m in model_names]
    bars = ax.bar(range(len(model_names)), vals, color=["C0" if m != best else "C3" for m in model_names])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("out-of-fold RMSE [dB]")
    ax.set_ylim(min(vals) - 0.15, max(vals) + 0.1)
    ax.set_title("round 2 on wp_lmmse_ce")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    est_names = list(per_estimator)
    ax.bar(range(len(est_names)), [per_estimator[e]["rmse"] for e in est_names], color="C2")
    for i, e in enumerate(est_names):
        ax.text(i, per_estimator[e]["rmse"] + 0.01, f"{per_estimator[e]['rmse']:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(est_names)))
    ax.set_xticklabels(est_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("out-of-fold RMSE [dB]")
    ax.set_title(f"{best} per channel estimator")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 2]
    if reach:
        rn = list(reach)
        ax.bar(range(len(rn)), [reach[e]["auc"] for e in rn], color="C4")
        for i, e in enumerate(rn):
            ax.text(i, reach[e]["auc"] + 0.002, f"{reach[e]['auc']:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(rn)))
        ax.set_xticklabels(rn, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.5, 1.02)
        ax.set_ylabel("ROC AUC")
        ax.set_title("P(WP not reached below snr_max)")
        ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    for code, name in enumerate(mod_names):
        m = data["modulation_code"][keep] == code
        ax.scatter(y[m], pred[m], s=5, alpha=0.3, label=name, color=f"C{code}")
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("true working point [dB]")
    ax.set_ylabel(f"{best} prediction [dB]")
    ax.set_title(f"wp_lmmse_ce   RMSE {results[best]['rmse']:.2f} dB   R2 {results[best]['r2']:.2f}")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(resid, bins=120, color="C0", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("prediction - true [dB]")
    ax.set_ylabel("records")
    ax.set_title(f"residual histogram   p90|e| {results[best]['p90_abs']:.2f} dB")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 2]
    err = np.sort(np.abs(resid))
    ax.plot(err, np.linspace(0, 1, len(err)), lw=2, label=best)
    for other in model_names:
        if other == best:
            continue
        e = np.sort(np.abs(oof[other] - y))
        ax.plot(e, np.linspace(0, 1, len(e)), lw=1, alpha=0.6, label=other)
    ax.set_xlim(0, 8)
    ax.axhline(0.9, color="k", ls=":", lw=0.8)
    ax.set_xlabel("|prediction - true| [dB]")
    ax.set_ylabel("fraction of records")
    ax.set_title("absolute-error CDF")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Working-point prediction, round 2  ·  best: {best}", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=125, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
