"""DMRS channel estimators used by the PUSCH receiver.

``perfect`` is the true frequency-domain channel (genie). The LS variants
reuse Sionna's PUSCH LS estimator with different interpolators. ``lmmse_ce``
applies LMMSE interpolation with a TDL covariance prior built from the
scenario delay spread and speed: the TDL-B/C tap delays are the CDL-B/C
cluster delays, so on CDL this prior *is* the channel's PDP (a genie prior,
which collapses as soon as the delay spread is off by 20 %). ``lmmse_exp``
uses a smooth exponential PDP with the same RMS delay spread, the robust
prior a real receiver would use. ``ls_hard_window`` / ``ls_soft_window``
denoise the LS estimate in the delay (tap) domain, see
:mod:`nr_ul_sim.windowed_ce`.
"""

from __future__ import annotations

import numpy as np
import torch

from .parameters import SimConfig

LS_INTERPOLATION = {
    "ls_nn": "nn",
    "ls_lin": "lin",
    "ls_lin_time_avg": "lin_time_avg",
}
WINDOW_MODE = {
    "ls_hard_window": "hard",
    "ls_soft_window": "soft",
}

ROBUST_PRIOR_DS_SYSTEM_LEVEL = 1000e-9  # exponential prior for UMi/UMa drops [s]

# CDL letter ≈ TDL letter; UMi/UMa use a typical NLoS urban prior.
TDL_PRIOR = {
    "cdl-b": "B",
    "cdl-c": "C",
    "cdl-d": "D",
    "umi": "C",
    "uma": "C",
}


def lmmse_covariances(cfg: SimConfig, resource_grid):
    """Time / frequency covariance of a TDL prior matched to the scenario."""
    from sionna.phy.ofdm import tdl_freq_cov_mat, tdl_time_cov_mat

    model = TDL_PRIOR.get(cfg.channel, "C")
    cov_freq = tdl_freq_cov_mat(
        model,
        float(resource_grid.subcarrier_spacing),
        int(resource_grid.num_effective_subcarriers),
        cfg.delay_spread,
    )
    cov_time = tdl_time_cov_mat(
        model,
        cfg.speed_mps,
        cfg.carrier_frequency,
        float(resource_grid.ofdm_symbol_duration),
        int(resource_grid.num_ofdm_symbols),
    )
    return cov_time, cov_freq


def exp_pdp_freq_cov_mat(num_subcarriers: int, subcarrier_spacing: float, rms_delay_spread: float):
    """Frequency covariance of an exponential PDP: R(df) = 1 / (1 + j 2 pi df tau_rms)."""
    k = np.arange(num_subcarriers)
    df = (k[:, None] - k[None, :]) * float(subcarrier_spacing)
    r = 1.0 / (1.0 + 2j * np.pi * df * float(rms_delay_spread))
    return torch.as_tensor(r.astype(np.complex64))


def robust_lmmse_covariances(cfg: SimConfig, resource_grid):
    """Jakes time covariance (as lmmse_ce) and an exponential-PDP frequency prior."""
    from sionna.phy.ofdm import tdl_time_cov_mat

    if cfg.ce_lmmse_prior_ds_ns is not None:
        ds = float(cfg.ce_lmmse_prior_ds_ns) * 1e-9
    elif cfg.channel in ("umi", "uma"):
        # the drop draws its own delay spread; a long prior only costs a little
        # noise, a short one over-smooths the channel (UMi 129 ns: -12 dB floor)
        ds = ROBUST_PRIOR_DS_SYSTEM_LEVEL
    else:
        ds = cfg.delay_spread
    cov_freq = exp_pdp_freq_cov_mat(
        int(resource_grid.num_effective_subcarriers), float(resource_grid.subcarrier_spacing), ds
    )
    cov_time = tdl_time_cov_mat(
        TDL_PRIOR.get(cfg.channel, "C"),
        cfg.speed_mps,
        cfg.carrier_frequency,
        float(resource_grid.ofdm_symbol_duration),
        int(resource_grid.num_ofdm_symbols),
    )
    return cov_time, cov_freq


def build_estimator(name: str, transmitter, cfg: SimConfig):
    """Return a callable ``(y, no) -> (h_hat, err_var)``, or ``None`` for perfect CSI."""
    if name == "perfect":
        return None

    from sionna.phy.nr import PUSCHLMMSEChannelEstimator, PUSCHLSChannelEstimator

    from .windowed_ce import WindowedLSChannelEstimator

    args = (
        transmitter.resource_grid,
        transmitter._dmrs_length,
        transmitter._dmrs_additional_position,
        transmitter._num_cdm_groups_without_data,
    )
    if name in LS_INTERPOLATION:
        return PUSCHLSChannelEstimator(*args, interpolation_type=LS_INTERPOLATION[name])
    if name in WINDOW_MODE:
        return WindowedLSChannelEstimator(
            *args,
            mode=WINDOW_MODE[name],
            window_pos_us=cfg.ce_window_pos_us,
            window_neg_us=cfg.ce_window_neg_us,
            soft_threshold=cfg.ce_soft_threshold,
            soft_within_window=cfg.ce_soft_within_window,
            time_interp=cfg.ce_time_interp,
        )
    if name in ("lmmse_ce", "lmmse_exp"):
        build = lmmse_covariances if name == "lmmse_ce" else robust_lmmse_covariances
        cov_time, cov_freq = build(cfg, transmitter.resource_grid)
        return PUSCHLMMSEChannelEstimator(
            *args, cov_mat_time=cov_time, cov_mat_freq=cov_freq
        )
    raise ValueError(f"Unknown channel estimator {name!r}")
