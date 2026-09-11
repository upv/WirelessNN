"""NR PUSCH configuration: DMRS Type-1, 68 PRB / 816 subcarriers, FFT 1024."""

from __future__ import annotations

from typing import Any

import numpy as np

from .parameters import SimConfig


def _as_int(value: Any) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel"):
        return int(value.reshape(-1)[0].item())
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def build_pusch_configs(cfg: SimConfig):
    """Create one PUSCHConfig per UE with orthogonal DMRS Type-1 ports."""
    from sionna.phy.nr import PUSCHConfig

    mcs_table, mcs_index = cfg.resolved_mcs
    dmrs_length = cfg.resolved_dmrs_length
    num_ue_ant = cfg.resolved_ue_ant
    if num_ue_ant not in (1, 2, 4):
        raise ValueError("NR PUSCH antenna ports must be 1, 2, or 4")
    if cfg.num_layers > num_ue_ant:
        raise ValueError("num_layers cannot exceed the number of UE antenna ports")

    base = PUSCHConfig()
    base.carrier.subcarrier_spacing = cfg.subcarrier_spacing_khz
    base.carrier.n_size_grid = cfg.num_prb
    base.n_size_bwp = cfg.num_prb
    base.mapping_type = cfg.mapping_type
    base.symbol_allocation = [0, cfg.num_ofdm_symbols]
    base.num_antenna_ports = num_ue_ant
    base.num_layers = cfg.num_layers
    if num_ue_ant > cfg.num_layers:
        base.precoding = "codebook"
        base.tpmi = 0
    else:
        base.precoding = "non-codebook"

    base.dmrs.config_type = 1
    base.dmrs.length = dmrs_length
    base.dmrs.additional_position = cfg.dmrs_additional_position
    base.dmrs.num_cdm_groups_without_data = cfg.num_cdm_groups_without_data
    base.tb.mcs_table = mcs_table
    base.tb.mcs_index = mcs_index

    configs = []
    for ue in range(cfg.num_ue):
        pc = base.clone()
        start = ue * cfg.num_layers
        pc.dmrs.dmrs_port_set = list(range(start, start + cfg.num_layers))
        pc.n_rnti = ue + 1
        configs.append(pc)
    return configs


def pad_transmitter_fft(transmitter, cfg: SimConfig):
    """Rebuild the OFDM grid with FFT size 1024 and guard subcarriers.

    Sionna's PUSCHTransmitter uses ``fft_size = 12 * num_prb`` (816 for 68 PRBs).
    NR 25 MHz / 30 kHz numerology uses a 1024-point FFT, so unused bins are
    filled with left/right guards: 104 + 816 + 104.
    """
    from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper

    old = transmitter.resource_grid
    used = old.fft_size
    if used != cfg.num_used_subcarriers:
        raise RuntimeError(
            f"PUSCH grid has {used} subcarriers, expected {cfg.num_used_subcarriers}"
        )
    if cfg.fft_size == used:
        return old

    left, right = cfg.guard_carriers
    cp_samples = int(round(old.cyclic_prefix_length * cfg.fft_size / used))
    rg = ResourceGrid(
        num_ofdm_symbols=old.num_ofdm_symbols,
        fft_size=cfg.fft_size,
        subcarrier_spacing=old.subcarrier_spacing,
        num_tx=old.num_tx,
        num_streams_per_tx=old.num_streams_per_tx,
        cyclic_prefix_length=cp_samples,
        num_guard_carriers=(left, right),
        dc_null=False,
        pilot_pattern=transmitter.pilot_pattern,
        precision=transmitter.precision,
        device=transmitter.device,
    )
    transmitter._resource_grid = rg
    transmitter._resource_grid_mapper = ResourceGridMapper(
        rg, precision=transmitter.precision, device=transmitter.device
    )
    return rg


def build_transmitter(cfg: SimConfig):
    from sionna.phy.nr import PUSCHTransmitter

    configs = build_pusch_configs(cfg)
    transmitter = PUSCHTransmitter(configs, output_domain=cfg.domain)
    pad_transmitter_fft(transmitter, cfg)
    return transmitter, configs


def build_stream_management(cfg: SimConfig):
    from sionna.phy.mimo import StreamManagement

    rx_tx_association = np.ones([1, cfg.num_ue], dtype=bool)
    return StreamManagement(rx_tx_association, cfg.num_layers)
