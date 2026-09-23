"""Ridge regression: scalar features -> working point [dB].

The model does not see the channel tensor. It is the linear baseline a network
trained on ``H.npy`` has to beat.

    from nr_ul_sim.dataset import load_dataset
    from nr_ul_sim.ridge import RidgePipeline

    data = load_dataset("dataset/run5000")
    pipe = RidgePipeline.fit(data)
    pipe.save("models/ridge")
    pred = RidgePipeline.load("models/ridge").predict(data, "lmmse")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import load_dataset

SCENARIO_FEATURES = (
    ("iot_db", "iot_db"),
    ("iot_db2", "iot_db^2"),
    ("num_layers_total", "num_layers_total"),
    ("num_ue", "num_ue"),
    ("rank", "rank"),
    ("speed_kmh", "speed_kmh"),
    ("delay_spread_ns", "delay_spread_ns"),
    ("num_interferers", "num_interferers"),
    ("bits_per_symbol", "num_bits_per_symbol"),
    ("target_coderate", "target_coderate"),
)
ALPHA_GRID = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)


def design_matrix(data: dict[str, np.ndarray], *, delay_fill: float | None = None
                  ) -> tuple[np.ndarray, list[str], float]:
    delay = np.asarray(data["delay_spread_ns"], dtype=float).copy()
    if delay_fill is None:
        finite = delay[np.isfinite(delay)]
        delay_fill = float(np.median(finite)) if finite.size else 0.0
    delay[~np.isfinite(delay)] = delay_fill

    columns = [np.asarray(data["features"][:, i], dtype=float)
               for i in range(data["features"].shape[1])]
    names = [str(n) for n in data["feature_names"]]
    lookup = {
        "iot_db": np.asarray(data["iot_db"], dtype=float),
        "iot_db^2": np.asarray(data["iot_db"], dtype=float) ** 2,
        "num_layers_total": np.asarray(data["num_layers_total"], dtype=float),
        "num_ue": np.asarray(data["num_ue"], dtype=float),
        "rank": np.asarray(data["rank"], dtype=float),
        "speed_kmh": np.asarray(data["speed_kmh"], dtype=float),
        "delay_spread_ns": delay,
        "num_interferers": np.asarray(data["num_interferers"], dtype=float),
        "num_bits_per_symbol": np.asarray(data["num_bits_per_symbol"], dtype=float),
        "target_coderate": np.asarray(data["target_coderate"], dtype=float),
    }
    for _, key in SCENARIO_FEATURES:
        columns.append(lookup[key])
        names.append(key)
    for code, name in enumerate(data["channel_names"]):
        columns.append((data["channel_code"] == code).astype(float))
        names.append(f"is_{name}")
    return np.column_stack(columns), names, delay_fill


def _channel_split(scenario: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    channels = np.unique(scenario)
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(channels) * fraction)))
    held = set(rng.choice(channels, n_test, replace=False).tolist())
    return np.array([s in held for s in scenario])


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    resid = pred - y
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    var = float(np.mean((y - y.mean()) ** 2))
    return {
        "n": int(y.size),
        "rmse": rmse,
        "mae": mae,
        "r2": float(1.0 - np.mean(resid**2) / var) if var > 0 else 0.0,
        "bias": float(resid.mean()),
        "p90_abs": float(np.percentile(np.abs(resid), 90)),
    }


def _solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    xtx.flat[:: xtx.shape[0] + 1] += alpha
    return np.linalg.solve(xtx, x.T @ y)


@dataclass
class RidgePipeline:
    feature_names: list[str]
    receivers: list[str]
    channel_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    delay_fill: float
    weights: dict[str, np.ndarray]
    alpha: dict[str, float]
    split: dict[str, Any]

    @classmethod
    def fit(
        cls,
        data: dict[str, np.ndarray],
        *,
        alpha: float | None = None,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> "RidgePipeline":
        x_raw, names, delay_fill = design_matrix(data)
        receivers = [str(r) for r in data["receivers"]]
        scenario = np.asarray(data["scenario_id"])
        test = _channel_split(scenario, test_fraction, seed)
        train = ~test

        mean = x_raw[train].mean(0)
        std = x_raw[train].std(0)
        std[std == 0] = 1.0
        x = np.column_stack([(x_raw - mean) / std, np.ones(len(x_raw))])

        weights: dict[str, np.ndarray] = {}
        alphas: dict[str, float] = {}
        for receiver in receivers:
            y = np.asarray(data[f"wp_{receiver}"], dtype=float)
            keep = np.isfinite(y)
            inner_test = _channel_split(scenario[train & keep], 0.2, seed + 1)
            x_tr, y_tr = x[train & keep], y[train & keep]
            if alpha is None:
                chosen, best = ALPHA_GRID[0], np.inf
                for cand in ALPHA_GRID:
                    w = _solve(x_tr[~inner_test], y_tr[~inner_test], cand)
                    rmse = float(np.sqrt(np.mean((x_tr[inner_test] @ w - y_tr[inner_test]) ** 2)))
                    if rmse < best:
                        chosen, best = cand, rmse
            else:
                chosen = float(alpha)
            weights[receiver] = _solve(x_tr, y_tr, chosen)
            alphas[receiver] = chosen

        return cls(
            feature_names=names,
            receivers=receivers,
            channel_names=[str(c) for c in data["channel_names"]],
            mean=mean.astype(np.float64),
            std=std.astype(np.float64),
            delay_fill=delay_fill,
            weights=weights,
            alpha=alphas,
            split={
                "seed": seed,
                "test_fraction": test_fraction,
                "n_train_channels": int(len(np.unique(scenario[train]))),
                "n_test_channels": int(len(np.unique(scenario[test]))),
                "test_scenario_ids": np.unique(scenario[test]).astype(int).tolist(),
            },
        )

    def transform(self, data: dict[str, np.ndarray]) -> np.ndarray:
        x_raw, names, _ = design_matrix(data, delay_fill=self.delay_fill)
        if names != self.feature_names:
            raise ValueError(f"feature mismatch: {names} != {self.feature_names}")
        return np.column_stack([(x_raw - self.mean) / self.std, np.ones(len(x_raw))])

    def predict(self, data: dict[str, np.ndarray], receiver: str) -> np.ndarray:
        if receiver not in self.weights:
            raise KeyError(f"no model for {receiver}; have {self.receivers}")
        return self.transform(data) @ self.weights[receiver]

    def evaluate(self, data: dict[str, np.ndarray]) -> dict[str, Any]:
        scenario = np.asarray(data["scenario_id"])
        test_ids = set(self.split["test_scenario_ids"])
        test = np.array([s in test_ids for s in scenario])
        out: dict[str, Any] = {"split": self.split, "alpha": self.alpha, "receivers": {}}
        for receiver in self.receivers:
            y = np.asarray(data[f"wp_{receiver}"], dtype=float)
            pred = self.predict(data, receiver)
            keep = np.isfinite(y)
            rec: dict[str, Any] = {
                "train": _metrics(y[keep & ~test], pred[keep & ~test]),
                "test": _metrics(y[keep & test], pred[keep & test]),
            }
            mods = [str(m) for m in data.get("modulation_names", ("qpsk", "qam16"))]
            for code, mod in enumerate(mods):
                m = keep & test & (data["modulation_code"] == code)
                if m.any():
                    rec[f"test_{mod}"] = _metrics(y[m], pred[m])
            out["receivers"][receiver] = rec
        return out

    def coefficients(self) -> list[dict[str, Any]]:
        rows = []
        for receiver, w in self.weights.items():
            intercept = float(w[-1] - np.sum(w[:-1] * self.mean / self.std))
            rows.append({
                "receiver": receiver,
                "feature": "intercept",
                "weight_standardized": float(w[-1]),
                "weight_per_unit": intercept,
            })
            for i, name in enumerate(self.feature_names):
                rows.append({
                    "receiver": receiver,
                    "feature": name,
                    "weight_standardized": float(w[i]),
                    "weight_per_unit": float(w[i] / self.std[i]),
                })
        return rows

    def save(self, outdir: str | Path) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            outdir / "pipeline.npz",
            mean=self.mean,
            std=self.std,
            delay_fill=np.array(self.delay_fill),
            feature_names=np.array(self.feature_names),
            receivers=np.array(self.receivers),
            channel_names=np.array(self.channel_names),
            **{f"w_{r}": self.weights[r] for r in self.receivers},
        )
        meta = {
            "alpha": self.alpha,
            "split": self.split,
            "feature_names": self.feature_names,
            "receivers": self.receivers,
            "channel_names": self.channel_names,
            "delay_fill": self.delay_fill,
        }
        (outdir / "pipeline.json").write_text(json.dumps(meta, indent=2))
        _write_coeff_csv(outdir / "coefficients.csv", self.coefficients())
        return outdir

    @classmethod
    def load(cls, outdir: str | Path) -> "RidgePipeline":
        outdir = Path(outdir)
        blob = np.load(outdir / "pipeline.npz", allow_pickle=False)
        meta = json.loads((outdir / "pipeline.json").read_text())
        receivers = [str(r) for r in blob["receivers"]]
        return cls(
            feature_names=[str(n) for n in blob["feature_names"]],
            receivers=receivers,
            channel_names=[str(c) for c in blob["channel_names"]],
            mean=np.asarray(blob["mean"], dtype=np.float64),
            std=np.asarray(blob["std"], dtype=np.float64),
            delay_fill=float(blob["delay_fill"]),
            weights={r: np.asarray(blob[f"w_{r}"], dtype=np.float64) for r in receivers},
            alpha={k: float(v) for k, v in meta["alpha"].items()},
            split=meta["split"],
        )


def _write_coeff_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fields = ["receiver", "feature", "weight_standardized", "weight_per_unit"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
