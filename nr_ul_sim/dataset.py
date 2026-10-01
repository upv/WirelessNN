"""Random-scenario dataset builder: channel tensor -> working point (BER = target).

Each scenario draws random link parameters (channel family, delay spread, number of
co-scheduled UEs, rank, speed, IoT level), freezes **one** channel realization and
searches the SNR where the coded BER crosses ``target_ber`` for every receiver,
channel estimator and modulation. Freezing the channel is what makes the dataset
well posed: the stored tensor, not a channel ensemble, determines the label.

The default sweep is IRC × {perfect, LS-NN, LS-linear, LS-lin+time-avg, LMMSE-CE}
× {QPSK, 16QAM, 64QAM}.

Run with ``python -m nr_ul_sim.dataset --num-samples 200 --outdir dataset/run1``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sionna.phy
import torch

from .channels import set_system_topology
from .interference import (
    random_ofdm_grid,
    serving_plus_interference,
    spatial_covariance_from_channel,
)
from .metrics import working_point_snr
from .parameters import (
    CHANNEL_ESTIMATORS,
    CHANNELS,
    MODULATIONS,
    RECEIVERS,
    SimConfig,
    canonicalize_estimator_name,
    canonicalize_receiver_name,
    detection_keys,
    snrdb_to_noise_var,
)
from .pusch import strip_guard_subcarriers
from .simulator import NRUplinkSimulator

CHANNEL_CODE = {name: i for i, name in enumerate(CHANNELS)}
MODULATION_CODE = {name: i for i, name in enumerate(MODULATIONS)}
FEATURE_NAMES = (
    "gain_db",
    "h_rms",
    "cond_db",
    "h_rank",
    "h_eff_rank",
    "capacity_bpcu_0db",
    "freq_corr",
    "time_corr",
    "rms_delay_samp",
    "h_spatial_corr",
    "stream_imbalance_db",
    "num_rx_ant",
    "num_streams",
    "ruu_rank",
    "ruu_eff_rank",
    "ruu_cond_db",
    "ruu_dom_frac",
)
RANK_REL_TOL = 1e-2


@dataclass
class DatasetConfig:
    num_samples: int = 100
    start_index: int = 0
    outdir: Path = Path("dataset")
    seed: int = 0
    resume: bool = False

    # --- scenario sampling -------------------------------------------------
    channels: tuple[str, ...] = CHANNELS
    modulations: tuple[str, ...] = MODULATIONS
    receivers: tuple[str, ...] = ("irc",)
    channel_estimators: tuple[str, ...] = CHANNEL_ESTIMATORS
    num_ue_choices: tuple[int, ...] = (1, 2, 3, 4)
    rank_choices: tuple[int, ...] = (1, 2)
    max_total_layers: int = 4
    num_rx_ant_choices: tuple[int, ...] = (4,)
    speed_kmh_range: tuple[float, float] = (3.0, 10.0)
    delay_spread_ns_range: tuple[float, float] = (30.0, 400.0)
    iot_db_range: tuple[float, float] = (0.0, 20.0)
    p_no_iot: float = 0.25
    num_interferer_choices: tuple[int, ...] = (1, 2, 3)

    # --- link configuration ------------------------------------------------
    num_prb: int = 68
    fft_size: int = 1024
    carrier_frequency: float = 3.5e9
    iot_cov: str = "perfect"
    channel_mode: str = "frozen"  # frozen | ensemble

    # --- working-point search ----------------------------------------------
    target_ber: float = 0.01
    snr_min: float = -20.0
    snr_max: float = 40.0
    coarse_step: float = 4.0
    refine_db: float = 0.5
    batch_size: int = 2
    max_mc_iter: int = 10
    num_target_bit_errors: int = 200

    # --- stored channel tensor ---------------------------------------------
    num_freq_bins: int = 64
    num_symbol_bins: int = 4
    store_interference: bool = True

    def __post_init__(self) -> None:
        self.outdir = Path(self.outdir)
        self.channels = tuple(c.lower() for c in self.channels)
        self.modulations = tuple(m.lower() for m in self.modulations)
        self.receivers = tuple(canonicalize_receiver_name(r) for r in self.receivers)
        self.channel_estimators = tuple(
            canonicalize_estimator_name(e) for e in self.channel_estimators
        ) or ("ls_lin",)
        if self.channel_mode not in ("frozen", "ensemble"):
            raise ValueError("channel_mode must be 'frozen' or 'ensemble'")
        for name, values, allowed in (
            ("channels", self.channels, CHANNELS),
            ("modulations", self.modulations, MODULATIONS),
            ("receivers", self.receivers, RECEIVERS),
            ("channel_estimators", self.channel_estimators, CHANNEL_ESTIMATORS),
        ):
            unknown = [v for v in values if v not in allowed]
            if unknown:
                raise ValueError(f"Unknown {name} {unknown}; choose from {allowed}")
        if not self.layer_combinations():
            raise ValueError("No (num_ue, rank) combination fits max_total_layers")
        if self.num_symbol_bins > 14:
            raise ValueError("num_symbol_bins cannot exceed the 14 OFDM symbols")
        if self.num_freq_bins > 12 * self.num_prb:
            raise ValueError("num_freq_bins cannot exceed the used subcarriers")

    def layer_combinations(self) -> list[tuple[int, int]]:
        limit = min(self.max_total_layers, 8)
        return [
            (int(ue), int(rank))
            for ue in self.num_ue_choices
            for rank in self.rank_choices
            if 1 <= ue <= 4 and 1 <= rank <= 2 and ue * rank <= limit
        ]

    def summary(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items()}
        out["outdir"] = str(self.outdir)
        return out


class FrozenChannelSimulator(NRUplinkSimulator):
    """Simulator that reuses a single channel realization in every slot."""

    def __init__(self, cfg: SimConfig):
        super().__init__(cfg)
        self.h_frozen: torch.Tensor | None = None
        self.h_int_frozen: torch.Tensor | None = None

    def freeze(self, h=None, h_int=None):
        cfg = self.cfg
        system_level = cfg.channel in ("umi", "uma")
        if h is None:
            if system_level:
                set_system_topology(self.serving_model, cfg, 1, cfg.num_ue)
            h = self.gen_serving(1)
        if h_int is None and self.gen_int is not None:
            if system_level:
                set_system_topology(self.int_model, cfg, 1, cfg.num_interferers)
            h_int = self.gen_int(1)
        self.h_frozen, self.h_int_frozen = h, h_int
        return h, h_int

    def generate_slot(self, batch_size: int, snr_db: float, iot_db: float):
        if self.h_frozen is None:
            self.freeze()
        cfg = self.cfg
        x, b = self.transmitter(batch_size)
        no = snrdb_to_noise_var(snr_db)

        h_s = self.h_frozen.expand(batch_size, *self.h_frozen.shape[1:])
        y_s = self.apply_channel(x, h_s)

        h_i = y_i = None
        if iot_db > 0.0 and self.h_int_frozen is not None:
            h_i = self.h_int_frozen.expand(batch_size, *self.h_int_frozen.shape[1:])
            x_i = random_ofdm_grid(
                batch_size, cfg.num_interferers, cfg.resolved_ue_ant,
                cfg.num_ofdm_symbols, cfg.fft_size,
                dtype=x.real.dtype, device=x.device,
                guard_carriers=cfg.guard_carriers,
            )
            y_i = self.apply_channel(x_i, h_i)

        return {
            "b": b,
            "y": serving_plus_interference(y_s, y_i, no, iot_db),
            "h": h_s,
            "h_int": h_i,
            "no": no,
            "iot_db": iot_db,
        }


class BerProbe:
    """Cached Monte-Carlo BER of all receivers at a given SNR, for one IoT level."""

    def __init__(self, sim: NRUplinkSimulator, iot_db: float):
        self.sim = sim
        self.iot_db = float(iot_db)
        self.ber: dict[float, dict[str, float]] = {}
        self.stats: dict[float, dict[str, dict]] = {}

    def _seed_point(self, snr_db: float) -> None:
        """Own random stream per SNR, so the search path cannot influence the BER."""
        seed = (self.sim.cfg.seed * 1_000_003 + int(round(snr_db * 1000))) % (2**31)
        sionna.phy.config.seed = seed

    def __call__(self, snr_db: float) -> dict[str, float]:
        key = round(float(snr_db), 3)
        if key not in self.ber:
            self._seed_point(key)
            stats = self.sim.measure_point(key, self.iot_db)
            self.ber[key] = {n: s.ber for n, s in stats.items()}
            self.stats[key] = {n: s.as_dict() for n, s in stats.items()}
        return self.ber[key]

    def curve(self) -> dict[str, Any]:
        snr = sorted(self.ber)
        names = [key for key, _, _ in detection_keys(
            self.sim.cfg.receivers, self.sim.cfg.channel_estimators
        )]
        return {
            "snr_db": snr,
            "ber": {n: [self.ber[s][n] for s in snr] for n in names},
            "bler": {n: [self.stats[s][n]["bler"] for s in snr] for n in names},
        }


def search_working_point(
    probe: BerProbe,
    receiver: str,
    *,
    target: float,
    snr_min: float,
    snr_max: float,
    coarse_step: float,
    refine_db: float,
    start_db: float | None = None,
) -> tuple[float | None, str]:
    """Bracket the BER = target crossing on a coarse grid, then bisect it."""
    eps = 1e-9

    def ber(snr: float) -> float:
        return probe(min(max(snr, snr_min), snr_max))[receiver]

    def snap(snr: float) -> float:
        steps = round((snr - snr_min) / coarse_step)
        return min(max(snr_min + steps * coarse_step, snr_min), snr_max)

    start = snap(snr_min if start_db is None else start_db)
    if ber(start) > target:
        lo = snr = start
        while snr < snr_max - eps:
            snr = min(snr + coarse_step, snr_max)
            if ber(snr) <= target:
                hi = snr
                break
            lo = snr
        else:
            return None, "not_reached"
    else:
        hi = snr = start
        while snr > snr_min + eps:
            snr = max(snr - coarse_step, snr_min)
            if ber(snr) > target:
                lo = snr
                break
            hi = snr
        else:
            return float(snr_min), "below_range"

    while hi - lo > refine_db + eps:
        mid = 0.5 * (lo + hi)
        if ber(mid) <= target:
            hi = mid
        else:
            lo = mid

    if ber(hi) <= 0.0:
        # Error-free upper point: log interpolation would collapse onto ``lo``.
        return 0.5 * (lo + hi), "ok"
    wp = working_point_snr([lo, hi], [ber(lo), ber(hi)], target)
    return (None, "not_reached") if wp is None else (float(wp), "ok")


def _bin_index(length: int, num_bins: int) -> np.ndarray:
    if num_bins >= length:
        return np.arange(length)
    return np.linspace(0, length - 1, num_bins).round().astype(int)


def compact_channel(h: torch.Tensor, cfg: SimConfig, dcfg: DatasetConfig) -> np.ndarray:
    """[1, 1, Rx, Tx, TxAnt, Sym, Sc] -> [Rx, streams, num_symbol_bins, num_freq_bins]."""
    h = strip_guard_subcarriers(h, cfg.guard_carriers)[0, 0]
    rx_ant, num_tx, tx_ant, num_sym, num_sc = h.shape
    idx_sym = torch.as_tensor(_bin_index(num_sym, dcfg.num_symbol_bins), device=h.device)
    idx_sc = torch.as_tensor(_bin_index(num_sc, dcfg.num_freq_bins), device=h.device)
    h = h.index_select(-2, idx_sym).index_select(-1, idx_sc)
    h = h.reshape(rx_ant, num_tx * tx_ant, idx_sym.numel(), idx_sc.numel())
    return h.detach().cpu().numpy().astype(np.complex64)


def channel_arrays(
    h: torch.Tensor, h_int: torch.Tensor | None, cfg: SimConfig, dcfg: DatasetConfig
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    arrays = {"h": compact_channel(h, cfg, dcfg)}
    r_iot = interference_covariance(h_int)
    if r_iot is not None:
        arrays["r_iot"] = r_iot
    if dcfg.store_interference and h_int is not None:
        arrays["h_int"] = compact_channel(h_int, cfg, dcfg)
    return arrays, channel_features(arrays["h"], arrays.get("r_iot"))


def interference_covariance(h_int: torch.Tensor | None) -> np.ndarray | None:
    """Spatial covariance of the interferers, normalized to trace = num_rx_ant."""
    if h_int is None:
        return None
    r = spatial_covariance_from_channel(h_int)[0]
    power = torch.diagonal(r).real.mean()
    if power > 0:
        r = r / power.to(r.dtype)
    return r.detach().cpu().numpy().astype(np.complex64)


def _numerical_rank(values: np.ndarray, rel: float = RANK_REL_TOL) -> float:
    """Count components above ``rel`` times the largest (descending 1-D array)."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = np.maximum(values, 0.0)
    if values.size == 0 or values[0] <= 0:
        return 0.0
    return float(np.sum(values >= rel * values[0]))


def _effective_rank(values: np.ndarray) -> float:
    """Participation ratio (sum v)^2 / sum v^2 of a non-negative 1-D spectrum."""
    values = np.maximum(np.asarray(values, dtype=float).reshape(-1), 0.0)
    energy = float(values.sum())
    if energy <= 0:
        return 0.0
    return float(energy**2 / np.square(values).sum())


def _rms_delay_samples(h: np.ndarray) -> float:
    """RMS delay (samples) from the IFFT of the stored frequency response."""
    tap = np.fft.ifft(h, axis=-1)
    pdp = np.mean(np.abs(tap) ** 2, axis=(0, 1, 2))
    total = float(pdp.sum())
    if total <= 0:
        return 0.0
    pdp = pdp / total
    tau = np.arange(pdp.size, dtype=float)
    mean_tau = float(pdp @ tau)
    return float(np.sqrt(max(pdp @ (tau - mean_tau) ** 2, 0.0)))


def _ruu_features(r_iot: np.ndarray | None) -> dict[str, float]:
    empty = {
        "ruu_rank": 0.0,
        "ruu_eff_rank": 0.0,
        "ruu_cond_db": 0.0,
        "ruu_dom_frac": 0.0,
    }
    if r_iot is None:
        return empty
    eig = np.maximum(np.linalg.eigvalsh(r_iot).real, 0.0)[::-1]
    energy = float(eig.sum())
    if energy <= 0:
        return empty
    return {
        "ruu_rank": _numerical_rank(eig),
        "ruu_eff_rank": _effective_rank(eig),
        "ruu_cond_db": float(10.0 * np.log10(max(eig[0] / max(eig[-1], 1e-12), 1.0))),
        "ruu_dom_frac": float(eig[0] / energy),
    }


def channel_features(h: np.ndarray, r_iot: np.ndarray | None = None) -> dict[str, float]:
    rx_ant, streams, _, num_sc = h.shape
    power = np.abs(h) ** 2
    gain = float(np.mean(power))
    mats = np.transpose(h, (2, 3, 0, 1))
    sv = np.maximum(np.linalg.svd(mats, compute_uv=False), 1e-12)
    ranks = np.sum(sv >= RANK_REL_TOL * sv[..., :1], axis=-1)

    def lag_corr(axis: int) -> float:
        if h.shape[axis] < 2 or gain <= 0:
            return 1.0
        a = np.take(h, np.arange(h.shape[axis] - 1), axis=axis)
        b = np.take(h, np.arange(1, h.shape[axis]), axis=axis)
        return float(np.clip(np.abs(np.mean(a * np.conj(b))) / gain, 0.0, 1.0))

    stream_pow = np.mean(power, axis=(0, 2, 3))
    rx_gram = np.tensordot(h, np.conj(h), axes=([1, 2, 3], [1, 2, 3])) / max(
        streams * h.shape[2] * num_sc, 1
    )
    off = rx_gram - np.diag(np.diag(rx_gram))
    denom = float(np.mean(np.abs(np.diag(rx_gram)))) if rx_ant else 1.0

    features = {
        "gain_db": float(10.0 * np.log10(max(gain, 1e-12))),
        "h_rms": float(np.sqrt(max(gain, 0.0))),
        "cond_db": float(np.median(20.0 * np.log10(sv[..., 0] / sv[..., -1]))),
        "h_rank": float(np.median(ranks)),
        "h_eff_rank": float(np.mean([_effective_rank(row) for row in sv.reshape(-1, sv.shape[-1])])),
        "capacity_bpcu_0db": float(np.mean(np.sum(np.log2(1.0 + sv**2), axis=-1))),
        "freq_corr": lag_corr(3),
        "time_corr": lag_corr(2),
        "rms_delay_samp": _rms_delay_samples(h),
        "h_spatial_corr": float(np.mean(np.abs(off)) / max(denom, 1e-12)) if rx_ant > 1 else 0.0,
        "stream_imbalance_db": (
            0.0 if streams < 2 or stream_pow.min() <= 0
            else float(10.0 * np.log10(stream_pow.max() / stream_pow.min()))
        ),
        "num_rx_ant": float(rx_ant),
        "num_streams": float(streams),
    }
    features.update(_ruu_features(r_iot))
    return features


def _scalar(value: Any) -> float:
    """Sionna exposes TB parameters as scalars, arrays or tensors."""
    if hasattr(value, "reshape"):
        return float(np.asarray(value).reshape(-1)[0])
    return float(value)


def sample_scenario(rng: np.random.Generator, dcfg: DatasetConfig) -> dict[str, Any]:
    channel = str(rng.choice(dcfg.channels))
    combos = dcfg.layer_combinations()
    num_ue, rank = combos[int(rng.integers(len(combos)))]
    lo_ds, hi_ds = dcfg.delay_spread_ns_range
    delay_spread_ns = (
        None
        if channel in ("umi", "uma")
        else float(np.exp(rng.uniform(np.log(lo_ds), np.log(hi_ds))))
    )
    if rng.random() < dcfg.p_no_iot:
        iot_db, num_interferers = 0.0, 0
    else:
        iot_db = float(rng.uniform(*dcfg.iot_db_range))
        num_interferers = int(rng.choice(dcfg.num_interferer_choices))
    return {
        "channel": channel,
        "delay_spread_ns": delay_spread_ns,
        "num_ue": num_ue,
        "rank": rank,
        "num_rx_ant": int(rng.choice(dcfg.num_rx_ant_choices)),
        "speed_kmh": float(rng.uniform(*dcfg.speed_kmh_range)),
        "iot_db": iot_db,
        "num_interferers": num_interferers,
    }


def scenario_config(scenario: dict[str, Any], modulation: str, seed: int,
                    dcfg: DatasetConfig) -> SimConfig:
    return SimConfig(
        channel=scenario["channel"],
        delay_spread_ns=scenario["delay_spread_ns"],
        num_ue=scenario["num_ue"],
        num_layers=scenario["rank"],
        num_rx_ant=scenario["num_rx_ant"],
        speed_kmh=scenario["speed_kmh"],
        modulation=modulation,
        snr_db=[0.0],
        iot_db=[scenario["iot_db"]],
        num_interferers=scenario["num_interferers"],
        receivers=dcfg.receivers,
        channel_estimators=dcfg.channel_estimators,
        iot_cov=dcfg.iot_cov,
        carrier_frequency=dcfg.carrier_frequency,
        num_prb=dcfg.num_prb,
        fft_size=dcfg.fft_size,
        batch_size=dcfg.batch_size,
        max_mc_iter=dcfg.max_mc_iter,
        num_target_bit_errors=dcfg.num_target_bit_errors,
        target_ber=dcfg.target_ber,
        seed=seed,
    )


def run_scenario(
    dcfg: DatasetConfig, index: int, hints: dict[str, float] | None = None
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """One random scenario: freeze a channel, label it for every modulation."""
    rng = np.random.default_rng([dcfg.seed, index])
    scenario = sample_scenario(rng, dcfg)
    seed = int(rng.integers(1, 2**31 - 1))
    hints = {} if hints is None else hints

    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    features: dict[str, float] = {}
    h_frozen = h_int_frozen = None

    for modulation in dcfg.modulations:
        t0 = time.time()
        cfg = scenario_config(scenario, modulation, seed, dcfg)
        if dcfg.channel_mode == "frozen":
            sim = FrozenChannelSimulator(cfg)
            h_frozen, h_int_frozen = sim.freeze(h_frozen, h_int_frozen)
        else:
            sim = NRUplinkSimulator(cfg)
            if h_frozen is None:
                slot = sim.generate_slot(1, 0.0, scenario["iot_db"])
                h_frozen, h_int_frozen = slot["h"][:1], (
                    None if slot["h_int"] is None else slot["h_int"][:1]
                )

        if not arrays:
            arrays, features = channel_arrays(h_frozen, h_int_frozen, cfg, dcfg)

        probe = BerProbe(sim, scenario["iot_db"])
        working_point: dict[str, float | None] = {}
        status: dict[str, str] = {}
        hint = hints.get(modulation)
        for key, _receiver, _estimator in detection_keys(dcfg.receivers, dcfg.channel_estimators):
            wp, state = search_working_point(
                probe, key,
                target=dcfg.target_ber,
                snr_min=dcfg.snr_min,
                snr_max=dcfg.snr_max,
                coarse_step=dcfg.coarse_step,
                refine_db=dcfg.refine_db,
                start_db=hint,
            )
            working_point[key] = wp
            status[key] = state
            if wp is not None:
                hint = wp
        if hint is not None:
            hints[modulation] = hint

        table, mcs_index = cfg.resolved_mcs
        records.append({
            "scenario_id": index,
            "channel": cfg.channel,
            "channel_code": CHANNEL_CODE[cfg.channel],
            "modulation": modulation,
            "modulation_code": MODULATION_CODE[modulation],
            "iot_db": scenario["iot_db"],
            "num_interferers": scenario["num_interferers"],
            "num_ue": cfg.num_ue,
            "rank": cfg.num_layers,
            "num_layers_total": cfg.num_ue * cfg.num_layers,
            "num_rx_ant": cfg.num_rx_ant,
            "num_ue_ant": cfg.resolved_ue_ant,
            "speed_kmh": cfg.speed_kmh,
            "delay_spread_ns": scenario["delay_spread_ns"],
            "carrier_frequency_hz": cfg.carrier_frequency,
            "num_prb": cfg.num_prb,
            "fft_size": cfg.fft_size,
            "mcs_table": table,
            "mcs_index": mcs_index,
            "num_bits_per_symbol": int(_scalar(sim.transmitter._num_bits_per_symbol)),
            "target_coderate": float(_scalar(sim.transmitter._target_coderate)),
            "tb_size": int(_scalar(sim.transmitter._tb_size)),
            "target_ber": dcfg.target_ber,
            "channel_mode": dcfg.channel_mode,
            "channel_estimators": list(dcfg.channel_estimators),
            "working_point_db": working_point,
            "working_point_status": status,
            "features": features,
            "channel_shape": list(arrays["h"].shape),
            "ber_curve": probe.curve(),
            "num_ber_points": len(probe.ber),
            "seed": seed,
            "duration_s": time.time() - t0,
        })
    return records, arrays


# --------------------------------------------------------------------------- #
# dataset files
# --------------------------------------------------------------------------- #

ARRAY_FILES = {"h": "H.npy", "h_int": "H_int.npy", "r_iot": "R_iot.npy"}
PATH_FIELDS = {"h": "channel_file", "h_int": "h_int_file", "r_iot": "r_iot_file"}
SCENARIO_KEYS = (
    "scenario_id", "channel", "channel_code", "iot_db", "num_interferers",
    "num_ue", "rank", "num_layers_total", "num_rx_ant", "num_ue_ant",
    "speed_kmh", "delay_spread_ns", "carrier_frequency_hz", "num_prb",
    "fft_size", "target_ber", "channel_mode", "channel_estimators",
    "seed", "features", "channel_shape",
)


def save_channel_arrays(
    outdir: Path, index: int, arrays: dict[str, np.ndarray], records: list[dict[str, Any]]
) -> dict[str, str]:
    """Write one folder: H.npy / H_int.npy / R_iot.npy + meta.json."""
    folder = Path("channels") / f"{index:06d}"
    (outdir / folder).mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    files: dict[str, str] = {}
    for key, name in ARRAY_FILES.items():
        if key not in arrays:
            continue
        rel = folder / name
        np.save(outdir / rel, arrays[key])
        paths[PATH_FIELDS[key]] = str(rel)
        files[name] = name

    head = records[0]
    scenario_meta = {k: head[k] for k in SCENARIO_KEYS if k in head}
    scenario_meta["files"] = files
    scenario_meta["modulations"] = {
        rec["modulation"]: {
            "modulation_code": rec["modulation_code"],
            "mcs_table": rec["mcs_table"],
            "mcs_index": rec["mcs_index"],
            "num_bits_per_symbol": rec["num_bits_per_symbol"],
            "target_coderate": rec["target_coderate"],
            "tb_size": rec["tb_size"],
            "working_point_db": rec["working_point_db"],
            "working_point_status": rec["working_point_status"],
        }
        for rec in records
    }
    (outdir / folder / "meta.json").write_text(json.dumps(scenario_meta, indent=2))
    return paths


def load_channel_arrays(outdir: Path, record: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, field in PATH_FIELDS.items():
        rel = record.get(field)
        if rel:
            arrays[key] = np.load(outdir / rel)
    return arrays


def _existing_scenarios(meta_path: Path) -> set[int]:
    if not meta_path.exists():
        return set()
    done = set()
    for line in meta_path.read_text().splitlines():
        if line.strip():
            done.add(int(json.loads(line)["scenario_id"]))
    return done


def generate_dataset(dcfg: DatasetConfig, verbose: bool = True) -> Path:
    outdir = dcfg.outdir
    (outdir / "channels").mkdir(parents=True, exist_ok=True)
    meta_path = outdir / "meta.jsonl"
    if meta_path.exists() and not dcfg.resume:
        raise FileExistsError(f"{meta_path} exists; pass resume=True to continue it")
    (outdir / "dataset_config.json").write_text(json.dumps(dcfg.summary(), indent=2))

    done = _existing_scenarios(meta_path) if dcfg.resume else set()
    indices = [
        i for i in range(dcfg.start_index, dcfg.start_index + dcfg.num_samples)
        if i not in done
    ]
    hints: dict[str, float] = {}
    t_start = time.time()

    for count, index in enumerate(indices, start=1):
        records, arrays = run_scenario(dcfg, index, hints)
        paths = save_channel_arrays(outdir, index, arrays, records)
        with meta_path.open("a") as fh:
            for record in records:
                record.update(paths)
                fh.write(json.dumps(record) + "\n")
        if verbose:
            elapsed = time.time() - t_start
            eta = elapsed / count * (len(indices) - count)
            head = records[0]
            wp_keys = list(records[0]["working_point_db"])
            wps = " ".join(
                f"{r['modulation']}:" + ",".join(
                    "--" if r["working_point_db"][n] is None
                    else f"{r['working_point_db'][n]:.1f}" for n in wp_keys
                )
                for r in records
            )
            print(
                f"[{count}/{len(indices)}] #{index} {head['channel']:5s} "
                f"{head['num_ue']}ue x r{head['rank']} "
                f"IoT={head['iot_db']:4.1f} dB v={head['speed_kmh']:4.1f} km/h | "
                f"{wps} | {sum(r['duration_s'] for r in records):5.1f} s "
                f"| ETA {eta / 60:.1f} min",
                flush=True,
            )

    return pack_dataset(outdir, verbose=verbose)


def pack_dataset(outdir: Path, verbose: bool = True) -> Path:
    """Stack the per-scenario tensors into one padded npz plus a flat CSV."""
    outdir = Path(outdir)
    meta_path = outdir / "meta.jsonl"
    records = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records in {meta_path}")
    receivers = list(records[0]["working_point_db"])

    tensors = {}
    for record in records:
        path = record["channel_file"]
        if path not in tensors:
            tensors[path] = load_channel_arrays(outdir, record)

    def stacked(key: str) -> np.ndarray:
        shapes = [t[key].shape for t in tensors.values() if key in t]
        if not shapes:
            return np.zeros((len(records), 0), dtype=np.complex64)
        dims = np.max(np.array(shapes), axis=0)
        out = np.zeros((len(records), *dims), dtype=np.complex64)
        for i, record in enumerate(records):
            arr = tensors[record["channel_file"]].get(key)
            if arr is not None:
                out[(i, *(slice(0, n) for n in arr.shape))] = arr
        return out

    columns: dict[str, np.ndarray] = {
        "H": stacked("h"),
        "H_int": stacked("h_int"),
        "R_iot": stacked("r_iot"),
        "features": np.array(
            [[record["features"][name] for name in FEATURE_NAMES] for record in records],
            dtype=np.float32,
        ),
        "feature_names": np.array(FEATURE_NAMES),
        "receivers": np.array(receivers),
        "estimator_names": np.array(records[0].get("channel_estimators", [])),
        "channel_names": np.array(CHANNELS),
        "modulation_names": np.array(MODULATIONS),
        "scenario_id": np.array([r["scenario_id"] for r in records], dtype=np.int64),
    }
    for name in (
        "iot_db", "speed_kmh", "target_coderate", "carrier_frequency_hz", "target_ber",
    ):
        columns[name] = np.array([r[name] for r in records], dtype=np.float32)
    columns["delay_spread_ns"] = np.array(
        [np.nan if r["delay_spread_ns"] is None else r["delay_spread_ns"] for r in records],
        dtype=np.float32,
    )
    for name in (
        "channel_code", "modulation_code", "num_ue", "rank", "num_layers_total",
        "num_rx_ant", "num_ue_ant", "num_interferers", "num_prb", "mcs_table",
        "mcs_index", "num_bits_per_symbol", "tb_size",
    ):
        columns[name] = np.array([r[name] for r in records], dtype=np.int32)
    for receiver in receivers:
        columns[f"wp_{receiver}"] = np.array(
            [
                np.nan if r["working_point_db"][receiver] is None
                else r["working_point_db"][receiver]
                for r in records
            ],
            dtype=np.float32,
        )
        columns[f"wp_status_{receiver}"] = np.array(
            [r["working_point_status"][receiver] for r in records]
        )

    npz_path = outdir / "dataset.npz"
    np.savez_compressed(npz_path, **columns)
    _write_csv(outdir / "dataset.csv", records, receivers)
    if verbose:
        print(
            f"\nPacked {len(records)} records "
            f"({len(tensors)} channels, H {columns['H'].shape}) -> {npz_path}"
        )
    return npz_path


def _write_csv(path: Path, records: list[dict[str, Any]], receivers: Sequence[str]) -> None:
    scalar_keys = []
    seen = set()
    for record in records:
        for k, v in record.items():
            if k not in seen and not isinstance(v, (dict, list)):
                seen.add(k)
                scalar_keys.append(k)
    fields = (
        scalar_keys
        + [f"wp_{r}" for r in receivers]
        + [f"wp_status_{r}" for r in receivers]
        + list(FEATURE_NAMES)
    )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {k: record.get(k) for k in scalar_keys}
            row.update({f"wp_{r}": record["working_point_db"][r] for r in receivers})
            row.update({f"wp_status_{r}": record["working_point_status"][r] for r in receivers})
            row.update({name: record["features"][name] for name in FEATURE_NAMES})
            writer.writerow(row)


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load ``dataset.npz`` (or a run directory) into a dict of arrays."""
    path = Path(path)
    if path.is_dir():
        path = path / "dataset.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _int_list(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in text.split(",") if v.strip())


def _str_list(text: str) -> tuple[str, ...]:
    return tuple(v.strip().lower() for v in text.split(",") if v.strip())


def _pair(text: str) -> tuple[float, float]:
    lo, hi = (float(v) for v in text.split(","))
    return lo, hi


def build_parser() -> argparse.ArgumentParser:
    d = DatasetConfig()
    p = argparse.ArgumentParser(description="Channel -> working-point dataset (Sionna NR PUSCH)")
    p.add_argument("--num-samples", type=int, default=d.num_samples)
    p.add_argument("--start-index", type=int, default=d.start_index, help="Shard offset")
    p.add_argument("--outdir", type=Path, default=d.outdir)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--resume", action="store_true", help="Append to an existing run")
    p.add_argument("--pack-only", action="store_true", help="Only rebuild dataset.npz/csv")

    p.add_argument("--channels", type=_str_list, default=d.channels)
    p.add_argument("--modulations", type=_str_list, default=d.modulations)
    p.add_argument("--receivers", type=_str_list, default=d.receivers)
    p.add_argument("--estimators", type=_str_list, default=d.channel_estimators,
                   dest="channel_estimators")
    p.add_argument("--num-ue", type=_int_list, default=d.num_ue_choices, dest="num_ue_choices")
    p.add_argument("--ranks", type=_int_list, default=d.rank_choices, dest="rank_choices")
    p.add_argument("--max-total-layers", type=int, default=d.max_total_layers)
    p.add_argument("--num-rx-ant", type=_int_list, default=d.num_rx_ant_choices,
                   dest="num_rx_ant_choices")
    p.add_argument("--speed-kmh", type=_pair, default=d.speed_kmh_range,
                   dest="speed_kmh_range", help="min,max")
    p.add_argument("--delay-spread-ns", type=_pair, default=d.delay_spread_ns_range,
                   dest="delay_spread_ns_range", help="min,max (CDL only)")
    p.add_argument("--iot-db", type=_pair, default=d.iot_db_range, dest="iot_db_range",
                   help="min,max INR")
    p.add_argument("--p-no-iot", type=float, default=d.p_no_iot)
    p.add_argument("--num-interferers", type=_int_list, default=d.num_interferer_choices,
                   dest="num_interferer_choices")

    p.add_argument("--num-prb", type=int, default=d.num_prb)
    p.add_argument("--fft-size", type=int, default=d.fft_size)
    p.add_argument("--carrier-ghz", type=float, default=d.carrier_frequency / 1e9)
    p.add_argument("--iot-cov", default=d.iot_cov, choices=["perfect", "estimated", "residual"])
    p.add_argument("--channel-mode", default=d.channel_mode, choices=["frozen", "ensemble"])

    p.add_argument("--target-ber", type=float, default=d.target_ber)
    p.add_argument("--snr-min", type=float, default=d.snr_min)
    p.add_argument("--snr-max", type=float, default=d.snr_max)
    p.add_argument("--coarse-step", type=float, default=d.coarse_step)
    p.add_argument("--refine-db", type=float, default=d.refine_db)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--max-mc-iter", type=int, default=d.max_mc_iter)
    p.add_argument("--num-target-bit-errors", type=int, default=d.num_target_bit_errors)

    p.add_argument("--num-freq-bins", type=int, default=d.num_freq_bins)
    p.add_argument("--num-symbol-bins", type=int, default=d.num_symbol_bins)
    p.add_argument("--no-interference-tensor", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> DatasetConfig:
    fields = {
        f: getattr(args, f) for f in DatasetConfig.__dataclass_fields__
        if hasattr(args, f)
    }
    fields["carrier_frequency"] = args.carrier_ghz * 1e9
    fields["store_interference"] = not args.no_interference_tensor
    return DatasetConfig(**fields)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pack_only:
        pack_dataset(Path(args.outdir))
        return 0
    dcfg = config_from_args(args)
    print("Working-point dataset")
    for key, value in dcfg.summary().items():
        print(f"  {key}: {value}")
    print()
    generate_dataset(dcfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
