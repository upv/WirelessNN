"""DMRS channel estimators used by the PUSCH receiver.

``perfect`` is the true frequency-domain channel (genie). The LS variants
reuse Sionna's PUSCH LS estimator with different interpolators. ``lmmse``
applies LMMSE interpolation with a TDL covariance prior built from the
scenario delay spread and speed.
"""

from __future__ import annotations

from .parameters import SimConfig

LS_INTERPOLATION = {
    "ls_nn": "nn",
    "ls_lin": "lin",
    "ls_lin_time_avg": "lin_time_avg",
}

# CDL letter ≈ TDL letter; UMi/UMa use a typical NLoS urban prior.
TDL_PRIOR = {
    "cdl-b": "B",
    "cdl-c": "C",
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


def build_estimator(name: str, transmitter, cfg: SimConfig):
    """Return a callable ``(y, no) -> (h_hat, err_var)``, or ``None`` for perfect CSI."""
    if name == "perfect":
        return None

    from sionna.phy.nr import PUSCHLMMSEChannelEstimator, PUSCHLSChannelEstimator

    args = (
        transmitter.resource_grid,
        transmitter._dmrs_length,
        transmitter._dmrs_additional_position,
        transmitter._num_cdm_groups_without_data,
    )
    if name in LS_INTERPOLATION:
        return PUSCHLSChannelEstimator(*args, interpolation_type=LS_INTERPOLATION[name])
    if name == "lmmse_ce":
        cov_time, cov_freq = lmmse_covariances(cfg, transmitter.resource_grid)
        return PUSCHLMMSEChannelEstimator(
            *args, cov_mat_time=cov_time, cov_mat_freq=cov_freq
        )
    raise ValueError(f"Unknown channel estimator {name!r}")
