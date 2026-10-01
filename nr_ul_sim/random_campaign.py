"""One configuration of the random-channel campaign: draw, simulate, record.

A configuration is (channel model, #UE, rank, modulation) from the plan plus
everything drawn from its seed: IoT and the number of interferers, and the
large-scale channel of every UE and interferer (:mod:`nr_ul_sim.random_channels`).
The IRC receiver is run with every channel estimator on shared slots and the
BER = 1e-2 working point of each estimator is searched adaptively with
censoring (:mod:`nr_ul_sim.wp_search`).

Channel-estimator priors: ``lmmse_ce`` gets a TDL prior with the delay spread
of UE 0 of this configuration (the drawn CDL delay spread, or the LSP delay
spread of the UMi/UMa drop) and the highest UE speed, i.e. a matched prior.
``lmmse_exp`` keeps its robust definition: exponential PDP with that delay
spread on CDL and 1000 ns on UMi/UMa. The windowed estimators use their
default windows.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from .interference import irc_interference_covariance, serving_plus_interference
from .parameters import CHANNEL_ESTIMATORS, MODULATIONS, SimConfig, snrdb_to_noise_var
from .random_channels import (
    CDL_LETTER,
    build_frozen_system_level,
    build_random_cdl,
    sample_cdl_link,
)
from .simulator import NRUplinkSimulator
from .wp_search import search_working_point

CHANNELS = ("cdl-b", "cdl-c", "cdl-d", "umi", "uma")
CELLS = [(ue, r, mod) for ue in (1, 2) for r in (1, 2) for mod in MODULATIONS]

# search range and starting point, IRC with perfect CSI, 1 UE, rank 1, IoT 0
SNR_RANGE_DB = {"qpsk": (-16.0, 36.0), "qam16": (-10.0, 42.0), "qam64": (-4.0, 48.0)}
BASE_WP_DB = {"qpsk": -5.0, "qam16": 2.0, "qam64": 7.0}
IOT_ZERO_PROBABILITY = 0.2
IOT_DB_RANGE = (0.5, 20.0)


def plan_configs(per_model: int, channels=CHANNELS, seed: int = 2026) -> list[dict]:
    """Balanced plan: every (#UE, rank, modulation) cell equally often per model.

    Rows are shuffled across models so that a partial run covers every model.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for ch in channels:
        cells = [CELLS[i % len(CELLS)] for i in range(per_model)]
        rng.shuffle(cells)
        rows += [{"channel": ch, "num_ue": ue, "rank": r, "modulation": mod}
                 for ue, r, mod in cells]
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = i
        row["seed"] = int((seed * 1_000_003 + i) % (2**31 - 1))
    return rows


def start_snr(mod: str, num_ue: int, rank: int, iot_db: float) -> float:
    lo, hi = SNR_RANGE_DB[mod]
    s = BASE_WP_DB[mod] + 3.0 * (num_ue * rank - 1) + 0.8 * iot_db
    return float(min(max(s, lo), hi))


class RandomChannelSimulator(NRUplinkSimulator):
    """NRUplinkSimulator with prebuilt, per-configuration fixed channel models."""

    def __init__(self, cfg: SimConfig, serving_model, interferer_model):
        self._prebuilt = [serving_model, interferer_model]
        super().__init__(cfg)

    def _make_model(self, num_tx: int):
        return self._prebuilt.pop(0)

    def generate_slot(self, batch_size: int, snr_db: float, iot_db: float):
        x, b = self.transmit(batch_size)
        no = snrdb_to_noise_var(snr_db)
        h_s = self.gen_serving(batch_size)
        y_s = self.apply_channel(x, h_s)
        h_i = y_i = None
        if iot_db > 0.0 and self.gen_int is not None:
            h_i = self.gen_int(batch_size)
            y_i = self.apply_channel(self.interferer_grid(batch_size, x), h_i)
        return {
            "b": b,
            "y": serving_plus_interference(y_s, y_i, no, iot_db, self.cfg.num_interferer_streams),
            "h": h_s,
            "h_int": h_i,
            "no": no,
            "iot_db": iot_db,
        }


def _f(x) -> float:
    return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)


@torch.no_grad()
def channel_features(sim: RandomChannelSimulator, cfg: SimConfig, iot_db: float,
                     batch_size: int = 16, snr_ref_db: float = 10.0) -> dict[str, Any]:
    """Measured features of the frozen channel (averaged over ``batch_size`` slots)."""
    rg = sim.transmitter.resource_grid
    out: dict[str, Any] = {}

    # --- impulse response: per-UE power delay profile
    a, tau = sim.serving_model(batch_size, rg.num_ofdm_symbols, 1.0 / rg.ofdm_symbol_duration)
    p = (a.abs() ** 2).double().mean(dim=(0, 1, 2, 4, 6))       # [num_tx, paths]
    t = tau.double().mean(dim=(0, 1))                             # [num_tx, paths]
    for u in range(cfg.num_ue):
        w = p[u] / p[u].sum()
        mean = (w * t[u]).sum()
        out[f"ue{u}_rms_ds_ns"] = _f(torch.sqrt(torch.clamp((w * t[u] ** 2).sum() - mean**2, min=0))) * 1e9
        out[f"ue{u}_mean_delay_ns"] = _f(mean) * 1e9
        out[f"ue{u}_max_path_frac"] = _f(w.max())
        out[f"ue{u}_num_paths_10db"] = int((w > w.max() * 0.1).sum())

    # --- frequency-domain effective channel on the used subcarriers
    he = sim.true_channel(sim.gen_serving(batch_size))  # [B, 1, rx, ue, layer, sym, sc]
    e = (he.abs() ** 2).mean()

    def fcorr(d: int) -> float:
        return _f((he[..., :-d] * he[..., d:].conj()).mean().abs() / e)

    out["freq_corr_1prb"] = fcorr(12)
    out["freq_corr_4prb"] = fcorr(48)
    last = he.shape[-2] - 1
    out["time_corr_slot"] = _f(
        (he[..., 0, :] * he[..., last, :].conj()).mean().abs()
        / torch.sqrt((he[..., 0, :].abs() ** 2).mean() * (he[..., last, :].abs() ** 2).mean())
    )
    b, _, rx, nue, nl, nsym, nsc = he.shape
    h = he[:, 0].permute(0, 4, 5, 1, 2, 3).reshape(b, nsym, nsc, rx, nue * nl)  # [B,sym,sc,rx,L]
    num_layers_total = nue * nl
    gram = h.conj().transpose(-1, -2) @ h
    eig = torch.linalg.eigvalsh(gram).clamp(min=1e-12)
    if num_layers_total > 1:
        out["cond_number_db_mean"] = _f((10 * torch.log10(eig[..., -1] / eig[..., 0])).mean())
        out["min_eig_norm_mean"] = _f((eig[..., 0] / eig.mean(dim=-1)).mean())
    else:
        out["cond_number_db_mean"] = 0.0
        out["min_eig_norm_mean"] = 1.0
    if nue == 2:
        h1, h2 = h[..., 0], h[..., nl]
        c = (h1.conj() * h2).sum(-1).abs() ** 2 / ((h1.abs() ** 2).sum(-1) * (h2.abs() ** 2).sum(-1))
        out["inter_ue_corr"] = _f(c.mean())
    else:
        out["inter_ue_corr"] = 0.0

    # --- reference post-IRC SINR with perfect CSI and covariance
    no = snrdb_to_noise_var(snr_ref_db)
    eye_rx = torch.eye(rx, dtype=h.dtype, device=h.device)
    r = no * eye_rx.expand(b, rx, rx)
    if iot_db > 0.0 and sim.gen_int is not None:
        dummy = torch.zeros(b, 1, rx, 1, 1, dtype=h.dtype, device=h.device)
        r_int = irc_interference_covariance(sim.gen_int(batch_size), dummy, no, iot_db, "perfect")
        ev = torch.linalg.eigvalsh(r_int).clamp(min=0)
        out["int_cov_top_eig_frac"] = _f((ev[..., -1] / ev.sum(-1).clamp(min=1e-30)).mean())
        hs = h.reshape(b, -1, rx, num_layers_total)
        proj = torch.einsum("bnrl,brk,bnkl->bnl", hs.conj(), r_int, hs).real
        norm = (hs.abs() ** 2).sum(-2) * (torch.diagonal(r_int, dim1=-2, dim2=-1).real.sum(-1) / rx)[:, None, None]
        out["int_serving_overlap"] = _f((proj / norm.clamp(min=1e-30)).mean())
        r = r + r_int
    else:
        out["int_cov_top_eig_frac"] = None
        out["int_serving_overlap"] = None
    r_inv = torch.linalg.inv(r)[:, None, None]                      # [B,1,1,rx,rx]
    a_mat = torch.eye(num_layers_total, dtype=h.dtype, device=h.device) + h.conj().transpose(-1, -2) @ r_inv @ h
    mse = torch.diagonal(torch.linalg.inv(a_mat), dim1=-2, dim2=-1).real.clamp(min=1e-12)
    sinr = (1.0 / mse - 1.0).clamp(min=1e-6).double()
    sinr_db = 10 * torch.log10(sinr)
    out["sinr_ref_snr_db"] = snr_ref_db
    out["sinr_ref_mean_db"] = _f(sinr_db.mean())
    out["sinr_ref_p10_db"] = _f(torch.quantile(sinr_db.flatten()[:1_000_000].float(), 0.1))
    out["sinr_ref_eff_db"] = 10 * math.log10(max(2.0 ** _f(torch.log2(1 + sinr).mean()) - 1.0, 1e-12))
    return out


def run_config(row: dict, *, num_prb: int = 16, fft_size: int = 1024, batch_size: int = 8,
               max_mc_iter: int = 8, target_bit_errors: int = 300, target_block_errors: int = 20,
               target_ber: float = 1e-2, max_points: int = 10,
               estimators=CHANNEL_ESTIMATORS) -> tuple[dict, dict]:
    """Simulate one planned configuration; returns (json record, arrays to save)."""
    import sionna.phy

    t0 = time.time()
    seed = int(row["seed"])
    rng = np.random.default_rng(seed)
    sionna.phy.config.seed = seed
    torch.manual_seed(seed)

    channel, mod = row["channel"], row["modulation"]
    num_ue, rank = int(row["num_ue"]), int(row["rank"])
    iot_db = 0.0 if rng.random() < IOT_ZERO_PROBABILITY else float(rng.uniform(*IOT_DB_RANGE))
    num_int = 0 if iot_db == 0.0 else int(rng.integers(1, 3))

    cfg = SimConfig(
        channel=channel, num_ue=num_ue, num_layers=rank, modulation=mod,
        iot_db=[iot_db], num_interferers=num_int, receivers=("irc",),
        channel_estimators=tuple(estimators), num_prb=num_prb, fft_size=fft_size,
        batch_size=batch_size, max_mc_iter=max_mc_iter,
        num_target_bit_errors=target_bit_errors, num_target_block_errors=target_block_errors,
        target_ber=target_ber, seed=seed,
    )

    arrays: dict[str, Any] = {}
    if channel in CDL_LETTER:
        ue_links = [sample_cdl_link(rng, channel) for _ in range(num_ue)]
        int_links = [sample_cdl_link(rng, channel) for _ in range(num_int)]
        serving, ue_info, ue_arr = build_random_cdl(cfg, ue_links)
        interf, int_info, int_arr = (build_random_cdl(cfg, int_links) if num_int else (None, [], []))
        for i, a in enumerate(ue_arr):
            arrays.update({f"ue{i}_{k}": np.asarray(v) for k, v in a.items()})
        for i, a in enumerate(int_arr):
            arrays.update({f"int{i}_{k}": np.asarray(v) for k, v in a.items()})
        ds_ue0_ns = ue_links[0]["ds_ns"]
    else:
        serving, ue_info, ue_arr = build_frozen_system_level(cfg, num_ue, rng)
        interf, int_info, int_arr = (
            build_frozen_system_level(cfg, num_int, rng) if num_int else (None, [], {}))
        arrays.update({f"ue_{k}": v for k, v in ue_arr.items()})
        arrays.update({f"int_{k}": v for k, v in (int_arr or {}).items()})
        ds_ue0_ns = ue_info[0]["lsp_ds_ns"]
    cfg.delay_spread_ns = float(ds_ue0_ns)              # prior of lmmse_ce / lmmse_exp (CDL)
    cfg.speed_kmh = float(max(u["speed_kmh"] for u in ue_info))

    sim = RandomChannelSimulator(cfg, serving, interf)
    t_build = time.time() - t0
    features = channel_features(sim, cfg, iot_db)
    t_feat = time.time() - t0 - t_build

    names = [key for key, _, _ in sim._keys()]

    def measure(snr: float) -> dict[str, dict]:
        stats = sim.measure_point(snr, iot_db)
        return {n: s.as_dict() for n, s in stats.items()}

    lo, hi = SNR_RANGE_DB[mod]
    s0 = start_snr(mod, num_ue, rank, iot_db)
    search = search_working_point(measure, names, lo=lo, hi=hi, start=s0,
                                  target=target_ber, max_points=max_points)
    t_total = time.time() - t0

    record = {
        **{k: row[k] for k in ("id", "seed", "channel", "num_ue", "rank", "modulation")},
        "num_layers_total": num_ue * rank,
        "iot_db": iot_db,
        "num_interferers": num_int,
        "mcs": list(cfg.resolved_mcs),
        "num_prb": num_prb,
        "num_rx_ant": cfg.num_rx_ant,
        "num_ue_ant": cfg.resolved_ue_ant,
        "prior_ds_ns": cfg.delay_spread_ns,
        "prior_speed_kmh": cfg.speed_kmh,
        "ue": ue_info,
        "interferers": int_info,
        "features": features,
        "search": {"lo_db": lo, "hi_db": hi, "start_db": s0, "target_ber": target_ber,
                   "num_points": search["num_points"], "snr_db": search["snr_db"]},
        "estimators": search["estimators"],
        "timing_s": {"build": t_build, "features": t_feat, "total": t_total},
        "sim": {"batch_size": batch_size, "max_mc_iter": max_mc_iter,
                "target_bit_errors": target_bit_errors, "target_block_errors": target_block_errors,
                "fft_size": fft_size, "carrier_ghz": cfg.carrier_frequency / 1e9,
                "scs_khz": cfg.subcarrier_spacing_khz},
    }
    return record, arrays
