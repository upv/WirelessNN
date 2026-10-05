"""PUSCH transmitter: Sionna ``PUSCHTransmitter`` on a guard-padded FFT grid.

One ``PUSCHConfig`` per UE (DMRS Type-1, consecutive ports per UE, MCS from the
config). Sionna maps only the allocated PRBs, so :func:`pad_transmitter_fft`
rebuilds the resource grid with the configured FFT size and guard carriers;
:func:`strip_guard_subcarriers` goes back to the used subcarriers.
"""

from __future__ import annotations

from .parameters import SimConfig


def build_pusch_configs(cfg: SimConfig):
    """One Sionna ``PUSCHConfig`` per UE; UE ``u`` gets DMRS ports ``u*rank .. u*rank+rank-1``."""
    from sionna.phy.nr import PUSCHConfig

    mcs_table, mcs_index = cfg.resolved_mcs
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
    base.precoding = "codebook" if num_ue_ant > cfg.num_layers else "non-codebook"
    if num_ue_ant > cfg.num_layers:
        base.tpmi = 0
    base.dmrs.config_type = 1
    base.dmrs.length = cfg.resolved_dmrs_length
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
    """Sionna maps used PRBs only; rebuild the grid with FFT guards."""
    from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper

    old = transmitter.resource_grid
    used = old.fft_size
    if used != cfg.num_used_subcarriers:
        raise RuntimeError(
            f"PUSCH grid has {used} subcarriers, expected {cfg.num_used_subcarriers}"
        )
    if cfg.fft_size == used:
        return

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


def strip_guard_subcarriers(h, guard_carriers: tuple[int, int]):
    """Keep the used subcarriers of a tensor whose last axis is the FFT bin."""
    left, right = int(guard_carriers[0]), int(guard_carriers[1])
    if left <= 0 and right <= 0:
        return h
    return h[..., left : h.shape[-1] - right] if right else h[..., left:]


def build_transmitter(cfg: SimConfig):
    """Frequency-domain ``PUSCHTransmitter`` for all UEs on the padded FFT grid."""
    from sionna.phy.nr import PUSCHTransmitter

    configs = build_pusch_configs(cfg)
    try:
        transmitter = PUSCHTransmitter(configs, output_domain="freq")
    except ValueError as exc:
        if "BG1" in str(exc) or "BG2" in str(exc):
            table, index = cfg.resolved_mcs
            raise ValueError(
                f"MCS table {table} index {index} gives target coderate "
                f"{configs[0].tb.target_coderate:.3f} for a {configs[0].tb_size}-bit TB, "
                "which needs LDPC repetition (BG1 below rate 1/3 or BG2 below 1/5); "
                "Sionna does not support it. Pick a higher --mcs-index or fewer PRBs."
            ) from exc
        raise
    pad_transmitter_fft(transmitter, cfg)
    return transmitter
