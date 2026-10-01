"""Random channel configurations that stay fixed for one working-point search.

Every configuration of the random-channel campaign draws its own large-scale
channel once and keeps it for all Monte-Carlo slots of its SNR sweep; only
the small-scale fading (random ray phases, and for CDL the random ray
coupling) changes from slot to slot. The working point of a configuration is
therefore a function of the drawn parameters, which is what the regression
model is fitted to.

CDL-B/C/D (TR 38.901 7.7.1), one independent link per UE and per interferer:

* RMS delay spread, log-uniform in ``DS_NS_RANGE``;
* UE speed and direction of travel (horizontal plane);
* angular scaling and shift of the cluster angles, TR 38.901 7.7.5.1
  (eq. 7.7-5): phi_scaled = AS_desired / AS_model * (phi - mu_model) + mu_desired,
  on both link ends. The BS-side mean azimuth / zenith encode the UE position in
  the sector, the UE-side mean azimuth the UE orientation;
* a small random per-cluster jitter of angles and delays, so that the cluster
  positions differ between configurations beyond a rigid rotation;
* CDL-D only: the Ricean K-factor (TR 38.901 7.7.6).

The CDL tables are in downlink convention: AoD/ZoD are BS-side angles, AoA/ZoA
UE-side angles. Sionna swaps them for the uplink.

UMi/UMa (TR 38.901 7.5): one single-sector drop per configuration (UE
positions, indoor state, LoS state, UE velocities), the large-scale
parameters of that drop and its clusters/rays are generated once and tiled
over the batch by :class:`FrozenSystemLevelChannel`.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch

CDL_LETTER = {"cdl-b": "B", "cdl-c": "C", "cdl-d": "D"}
SYSTEM_LEVEL = ("umi", "uma")

DS_NS_RANGE = (20.0, 1000.0)          # log-uniform
SPEED_KMH_RANGE = (0.5, 120.0)        # outdoor / CDL, uniform
INDOOR_SPEED_KMH_RANGE = (0.5, 5.0)   # indoor UMi/UMa UEs, uniform
AS_RATIO_RANGE = (0.5, 2.0)           # AS_desired / AS_model, log-uniform
BS_AZ_MEAN_RANGE_DEG = (-60.0, 60.0)  # UE azimuth inside a 120 deg sector
BS_ZEN_MEAN_RANGE_DEG = (92.0, 110.0)  # UE below the BS antenna
UE_AZ_MEAN_RANGE_DEG = (-180.0, 180.0)  # UE orientation
UE_ZEN_SHIFT_RANGE_DEG = (-10.0, 10.0)
CLUSTER_AZ_JITTER_DEG = 5.0
CLUSTER_ZEN_JITTER_DEG = 2.0
CLUSTER_DELAY_JITTER = 0.1            # sigma of the log-normal delay factor
K_FACTOR_DB_RANGE = (0.0, 15.0)       # CDL-D
INDOOR_PROBABILITY = 0.3              # UMi/UMa drops


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------

def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def wrap180(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def speed_to_velocity(speed_kmh: float, azimuth_deg: float) -> list[float]:
    v = speed_kmh / 3.6
    a = math.radians(azimuth_deg)
    return [v * math.cos(a), v * math.sin(a), 0.0]


def sample_cdl_link(rng: np.random.Generator, channel: str) -> dict[str, Any]:
    """Large-scale parameters of one CDL link (one UE or one interferer)."""
    p = {
        "ds_ns": log_uniform(rng, *DS_NS_RANGE),
        "speed_kmh": float(rng.uniform(*SPEED_KMH_RANGE)),
        "velocity_azimuth_deg": float(rng.uniform(-180.0, 180.0)),
        "bs_as_ratio": log_uniform(rng, *AS_RATIO_RANGE),
        "bs_zs_ratio": log_uniform(rng, *AS_RATIO_RANGE),
        "ue_as_ratio": log_uniform(rng, *AS_RATIO_RANGE),
        "ue_zs_ratio": log_uniform(rng, *AS_RATIO_RANGE),
        "bs_az_mean_deg": float(rng.uniform(*BS_AZ_MEAN_RANGE_DEG)),
        "bs_zen_mean_deg": float(rng.uniform(*BS_ZEN_MEAN_RANGE_DEG)),
        "ue_az_mean_deg": float(rng.uniform(*UE_AZ_MEAN_RANGE_DEG)),
        "ue_zen_shift_deg": float(rng.uniform(*UE_ZEN_SHIFT_RANGE_DEG)),
        "jitter_seed": int(rng.integers(0, 2**31 - 1)),
    }
    if channel == "cdl-d":
        p["k_factor_db"] = float(rng.uniform(*K_FACTOR_DB_RANGE))
    return p


# ---------------------------------------------------------------------------
# angular scaling, TR 38.901 7.7.5.1
# ---------------------------------------------------------------------------

def circular_mean_spread(angles_deg, powers) -> tuple[float, float]:
    """Power-weighted circular mean and angular spread (TR 38.901 A.1) [deg]."""
    w = np.asarray(powers, dtype=float)
    z = np.sum(w * np.exp(1j * np.deg2rad(angles_deg))) / np.sum(w)
    spread = math.sqrt(max(-2.0 * math.log(max(abs(z), 1e-12)), 0.0))
    return float(np.rad2deg(np.angle(z))), float(np.rad2deg(spread))


def linear_mean_spread(angles_deg, powers) -> tuple[float, float]:
    w = np.asarray(powers, dtype=float)
    a = np.asarray(angles_deg, dtype=float)
    mu = float(np.sum(w * a) / np.sum(w))
    return mu, float(math.sqrt(np.sum(w * (a - mu) ** 2) / np.sum(w)))


def scale_azimuth(angles_deg, powers, ratio: float, new_mean_deg: float):
    mu, _ = circular_mean_spread(angles_deg, powers)
    return wrap180(ratio * wrap180(np.asarray(angles_deg) - mu) + new_mean_deg)


def scale_zenith(angles_deg, powers, ratio: float, new_mean_deg: float):
    mu, _ = linear_mean_spread(angles_deg, powers)
    return np.clip(ratio * (np.asarray(angles_deg, dtype=float) - mu) + new_mean_deg, 0.0, 180.0)


def load_cdl_table(channel: str, spec_version: str = "19.2") -> dict:
    from sionna.phy.channel.tr38901 import models

    source = models.parameter_file(
        f"CDL-{CDL_LETTER[channel]}.json", models._validate_spec_version(spec_version)
    )
    return dict(models.load_json(source))


def randomize_cdl_table(table: dict, link: dict) -> dict:
    """Scaled / shifted / jittered cluster table of one link.

    Returns the JSON overrides for Sionna's CDL loader (normalised delays and
    the four cluster angles in downlink convention) and the resulting
    statistics, before and after the change.
    """
    powers = np.power(10.0, np.asarray(table["powers"], dtype=float) / 10.0)
    powers = powers / powers.sum()
    delays = np.asarray(table["delays"], dtype=float)
    bs_az = np.asarray(table["aod"], dtype=float)
    bs_zen = np.asarray(table["zod"], dtype=float)
    ue_az = np.asarray(table["aoa"], dtype=float)
    ue_zen = np.asarray(table["zoa"], dtype=float)

    stats: dict[str, float] = {}
    for tag, ang, circ in (("bs_as", bs_az, True), ("bs_zs", bs_zen, False),
                           ("ue_as", ue_az, True), ("ue_zs", ue_zen, False)):
        mu, sp = (circular_mean_spread if circ else linear_mean_spread)(ang, powers)
        stats[f"model_{tag}_mean_deg"] = mu
        stats[f"model_{tag}_deg"] = sp

    ue_zen_mean = stats["model_ue_zs_mean_deg"] + link["ue_zen_shift_deg"]
    bs_az = scale_azimuth(bs_az, powers, link["bs_as_ratio"], link["bs_az_mean_deg"])
    bs_zen = scale_zenith(bs_zen, powers, link["bs_zs_ratio"], link["bs_zen_mean_deg"])
    ue_az = scale_azimuth(ue_az, powers, link["ue_as_ratio"], link["ue_az_mean_deg"])
    ue_zen = scale_zenith(ue_zen, powers, link["ue_zs_ratio"], ue_zen_mean)

    jrng = np.random.default_rng(link["jitter_seed"])
    n = len(delays)
    bs_az = wrap180(bs_az + jrng.normal(0.0, CLUSTER_AZ_JITTER_DEG, n))
    ue_az = wrap180(ue_az + jrng.normal(0.0, CLUSTER_AZ_JITTER_DEG, n))
    bs_zen = np.clip(bs_zen + jrng.normal(0.0, CLUSTER_ZEN_JITTER_DEG, n), 0.0, 180.0)
    ue_zen = np.clip(ue_zen + jrng.normal(0.0, CLUSTER_ZEN_JITTER_DEG, n), 0.0, 180.0)
    factors = np.exp(jrng.normal(0.0, CLUSTER_DELAY_JITTER, n))
    delays = np.where(delays > 0.0, delays * factors, delays)

    for tag, ang, circ in (("bs_as", bs_az, True), ("bs_zs", bs_zen, False),
                           ("ue_as", ue_az, True), ("ue_zs", ue_zen, False)):
        mu, sp = (circular_mean_spread if circ else linear_mean_spread)(ang, powers)
        stats[f"{tag}_mean_deg"] = mu
        stats[f"{tag}_deg"] = sp
    w = powers
    mean_d = float(np.sum(w * delays))
    stats["norm_rms_ds"] = float(math.sqrt(max(np.sum(w * delays**2) - mean_d**2, 0.0)))

    return {
        "override": {
            "delays": delays.tolist(),
            "aod": bs_az.tolist(),
            "zod": bs_zen.tolist(),
            "aoa": ue_az.tolist(),
            "zoa": ue_zen.tolist(),
        },
        "powers_lin": powers.tolist(),
        "stats": stats,
    }


def _random_cdl_class():
    from sionna.phy.channel.tr38901 import CDL
    import sionna.phy.channel.tr38901.cdl as cdl_module

    class RandomCDL(CDL):
        """Sionna CDL whose cluster table is replaced by a randomised one."""

        def __init__(self, *, overrides: dict, k_factor_db: float | None = None, **kwargs):
            object.__setattr__(self, "_nrul_overrides", overrides)
            object.__setattr__(self, "_nrul_k_factor_db", k_factor_db)
            super().__init__(**kwargs)

        def _load_parameters(self) -> None:
            original = cdl_module.models.load_json
            overrides = self._nrul_overrides

            def patched(source):
                params = dict(original(source))
                params.update(overrides)
                return params

            cdl_module.models.load_json = patched
            try:
                super()._load_parameters()
            finally:
                cdl_module.models.load_json = original
            if self._nrul_k_factor_db is not None and self._has_los:
                self._k_factor = torch.tensor(
                    10.0 ** (self._nrul_k_factor_db / 10.0), dtype=self.dtype, device=self.device
                )

    return RandomCDL


def build_random_cdl(cfg, links: list[dict]):
    """IndependentLinkChannel of randomised CDL links; returns (model, per-link info)."""
    from .channels import IndependentLinkChannel, make_array

    RandomCDL = _random_cdl_class()
    table = load_cdl_table(cfg.channel)
    ut_array = make_array(cfg.resolved_ue_ant, cfg.carrier_frequency, bs=False)
    bs_array = make_array(cfg.num_rx_ant, cfg.carrier_frequency, bs=True)
    models, info, arrays = [], [], []
    for link in links:
        rnd = randomize_cdl_table(table, link)
        o = rnd["override"]
        arrays.append({
            "delays_s": (np.asarray(o["delays"]) * link["ds_ns"] * 1e-9).tolist(),
            "powers_lin": rnd["powers_lin"],
            "bs_az_deg": o["aod"], "bs_zen_deg": o["zod"],
            "ue_az_deg": o["aoa"], "ue_zen_deg": o["zoa"],
        })
        models.append(
            RandomCDL(
                overrides=rnd["override"],
                k_factor_db=link.get("k_factor_db"),
                model=CDL_LETTER[cfg.channel],
                delay_spread=link["ds_ns"] * 1e-9,
                carrier_frequency=cfg.carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction="uplink",
                ut_velocity=torch.tensor(
                    speed_to_velocity(link["speed_kmh"], link["velocity_azimuth_deg"])
                ),
            )
        )
        info.append({**link, **rnd["stats"]})
    return IndependentLinkChannel(models), info, arrays


# ---------------------------------------------------------------------------
# UMi / UMa: one frozen drop
# ---------------------------------------------------------------------------

def _tile(t: torch.Tensor | None, batch_size: int):
    if t is None or not torch.is_tensor(t):
        return t
    return t.repeat(batch_size, *([1] * (t.dim() - 1)))


class FrozenSystemLevelChannel:
    """UMi/UMa channel whose topology, LSPs and clusters/rays are fixed.

    ``set_topology`` must have been called on ``model`` with batch size 1. Each
    call tiles the frozen quantities over the batch; Sionna draws fresh random
    ray phases per batch example, which is the small-scale fading.
    """

    def __init__(self, model):
        self.model = model
        self.lsp = model._lsp
        self.rays = model._ray_sampler(self.lsp)

    def __call__(self, batch_size: int, num_time_steps: int, sampling_frequency: float):
        from sionna.phy.channel.utils import deg_2_rad
        from sionna.phy.channel.tr38901.channel_coefficients import Topology

        m, sc, lsp = self.model, self.model._scenario, self.lsp
        b = int(batch_size)
        rays = copy.copy(self.rays)
        for name, value in vars(self.rays).items():
            if torch.is_tensor(value):
                setattr(rays, name, _tile(value, b))
        rays.phases = None
        uplink = sc.direction == "uplink"
        topology = Topology(
            velocities=_tile(sc.ut_velocities, b),
            moving_end="tx" if uplink else "rx",
            los_aoa=deg_2_rad(_tile(sc.los_aoa, b)),
            los_aod=deg_2_rad(_tile(sc.los_aod, b)),
            los_zoa=deg_2_rad(_tile(sc.los_zoa, b)),
            los_zod=deg_2_rad(_tile(sc.los_zod, b)),
            los=_tile(sc.los, b),
            distance_3d=_tile(sc.distance_3d, b),
            tx_orientations=_tile(sc.ut_orientations if uplink else sc.bs_orientations, b),
            rx_orientations=_tile(sc.bs_orientations if uplink else sc.ut_orientations, b),
        )
        c_ds = _tile(sc.get_param("cDS"), b) * 1e-9
        k_factor = _tile(lsp.k_factor, b)
        sf = _tile(lsp.sf, b)
        pathloss = _tile(getattr(lsp, "pathloss", None), b)
        if uplink:
            p5 = (0, 2, 1, 3, 4)
            rays.aod, rays.aoa = rays.aoa.permute(*p5), rays.aod.permute(*p5)
            rays.zod, rays.zoa = rays.zoa.permute(*p5), rays.zod.permute(*p5)
            rays.powers = rays.powers.permute(0, 2, 1, 3)
            rays.delays = rays.delays.permute(0, 2, 1, 3)
            for name in ("cluster_sort_indices", "strongest_cluster_indices"):
                if getattr(rays, name, None) is not None:
                    setattr(rays, name, getattr(rays, name).permute(0, 2, 1, 3))
            rays.xpr = rays.xpr.permute(*p5)
            if getattr(rays, "blockage_loss_db", None) is not None:
                rays.blockage_loss_db = rays.blockage_loss_db.permute(*p5)
            if getattr(rays, "los_blockage_loss_db", None) is not None:
                rays.los_blockage_loss_db = rays.los_blockage_loss_db.permute(0, 2, 1)
            topology.los_aoa, topology.los_aod = (
                topology.los_aod.permute(0, 2, 1), topology.los_aoa.permute(0, 2, 1))
            topology.los_zoa, topology.los_zod = (
                topology.los_zod.permute(0, 2, 1), topology.los_zoa.permute(0, 2, 1))
            topology.los = topology.los.permute(0, 2, 1)
            topology.distance_3d = topology.distance_3d.permute(0, 2, 1)
            c_ds = c_ds.permute(0, 2, 1)
            k_factor = k_factor.permute(0, 2, 1)
            sf = sf.permute(0, 2, 1)
        h, delays = m._cir_sampler(num_time_steps, sampling_frequency, k_factor, rays, topology, c_ds)
        h = m._step_12(h, sf, pathloss)
        h = h.permute(0, 2, 4, 1, 5, 3, 6).detach()
        delays = delays.permute(0, 2, 1, 3).detach()
        return h, delays


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy()


def build_frozen_system_level(cfg, num_ut: int, rng: np.random.Generator):
    """Draw one drop with ``num_ut`` UEs; returns (channel, per-UE info, arrays)."""
    from sionna.phy.channel import gen_single_sector_topology

    from .channels import build_system_level_channel

    model = build_system_level_channel(cfg)
    ut_loc, bs_loc, ut_orient, bs_orient, _vel, in_state = gen_single_sector_topology(
        1, num_ut, cfg.channel, indoor_probability=INDOOR_PROBABILITY,
        min_ut_velocity=0.0, max_ut_velocity=0.0,
    )
    indoor = _np(in_state)[0].astype(bool)
    speeds, headings, vel = [], [], []
    for u in range(num_ut):
        rng_range = INDOOR_SPEED_KMH_RANGE if indoor[u] else SPEED_KMH_RANGE
        speeds.append(float(rng.uniform(*rng_range)))
        headings.append(float(rng.uniform(-180.0, 180.0)))
        vel.append(speed_to_velocity(speeds[-1], headings[-1]))
    velocities = torch.tensor([vel], dtype=ut_loc.dtype, device=ut_loc.device)
    model.set_topology(ut_loc, bs_loc, ut_orient, bs_orient, velocities, in_state)
    frozen = FrozenSystemLevelChannel(model)

    sc, lsp, rays = model._scenario, frozen.lsp, frozen.rays
    info = []
    for u in range(num_ut):
        k = float(_np(lsp.k_factor)[0, 0, u])
        info.append({
            "speed_kmh": speeds[u],
            "velocity_azimuth_deg": headings[u],
            "indoor": bool(indoor[u]),
            "los": bool(_np(sc.los)[0, 0, u]),
            "distance_2d_m": float(_np(sc.distance_2d)[0, 0, u]),
            "distance_3d_m": float(_np(sc.distance_3d)[0, 0, u]),
            "distance_2d_in_m": float(_np(sc.distance_2d_in)[0, 0, u]),
            "ut_height_m": float(_np(sc.h_ut)[0, u]),
            "ut_x_m": float(_np(sc.ut_loc)[0, u, 0]),
            "ut_y_m": float(_np(sc.ut_loc)[0, u, 1]),
            "bs_az_deg": float(_np(sc.los_aod)[0, 0, u]),   # LoS direction seen from the BS
            "bs_zen_deg": float(_np(sc.los_zod)[0, 0, u]),
            "ue_az_deg": float(_np(sc.los_aoa)[0, 0, u]),
            "ue_zen_deg": float(_np(sc.los_zoa)[0, 0, u]),
            "lsp_ds_ns": float(_np(lsp.ds)[0, 0, u]) * 1e9,
            "lsp_asd_deg": float(_np(lsp.asd)[0, 0, u]),
            "lsp_asa_deg": float(_np(lsp.asa)[0, 0, u]),
            "lsp_zsd_deg": float(_np(lsp.zsd)[0, 0, u]),
            "lsp_zsa_deg": float(_np(lsp.zsa)[0, 0, u]),
            "lsp_k_factor_db": 10.0 * math.log10(k) if k > 0 else None,
            "lsp_sf_db": 10.0 * math.log10(max(float(_np(lsp.sf)[0, 0, u]), 1e-30)),
            "num_clusters": int(_np(rays.delays).shape[-1]),
        })
    arrays = {
        name: _np(value)[0]
        for name, value in vars(rays).items()
        if torch.is_tensor(value)
    }
    arrays.update({"ut_loc": _np(sc.ut_loc)[0], "bs_loc": _np(sc.bs_loc)[0],
                   "ut_orientations": _np(sc.ut_orientations)[0],
                   "ut_velocities": _np(sc.ut_velocities)[0]})
    return frozen, info, arrays
