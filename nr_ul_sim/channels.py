from __future__ import annotations

import torch
from sionna.phy.channel import ChannelModel

from .parameters import SimConfig

_ARRAY_LAYOUT = {
    1: (1, 1, "single", "V"),
    2: (1, 1, "dual", "cross"),
    4: (1, 2, "dual", "cross"),
    8: (2, 2, "dual", "cross"),
}


def make_array(num_ant: int, carrier_frequency: float, *, bs: bool):
    from sionna.phy.channel.tr38901 import AntennaArray

    if num_ant not in _ARRAY_LAYOUT:
        raise ValueError(f"Unsupported antenna count {num_ant}; use 1, 2, 4 or 8")
    rows, cols, pol, ptype = _ARRAY_LAYOUT[num_ant]
    return AntennaArray(
        num_rows=rows,
        num_cols=cols,
        polarization=pol,
        polarization_type=ptype,
        antenna_pattern="38.901" if bs else "omni",
        carrier_frequency=carrier_frequency,
    )


class IndependentLinkChannel(ChannelModel):
    """Independent CDL realizations, stacked on the TX axis."""

    def __init__(self, links):
        super().__init__()
        if not links:
            raise ValueError("At least one link model is required")
        self.links = links

    def __call__(self, batch_size, num_time_steps, sampling_frequency):
        a_list, tau_list = [], []
        for link in self.links:
            a, tau = link(batch_size, num_time_steps, sampling_frequency)
            a_list.append(a)
            tau_list.append(tau)
        return torch.cat(a_list, dim=3), torch.cat(tau_list, dim=2)


def build_cdl_links(cfg: SimConfig, num_tx: int):
    from sionna.phy.channel.tr38901 import CDL

    letter = {"cdl-b": "B", "cdl-c": "C", "cdl-d": "D"}[cfg.channel]
    ut_array = make_array(cfg.resolved_ue_ant, cfg.carrier_frequency, bs=False)
    bs_array = make_array(cfg.num_rx_ant, cfg.carrier_frequency, bs=True)
    speed = cfg.speed_mps
    links = [
        CDL(
            model=letter,
            delay_spread=cfg.delay_spread,
            carrier_frequency=cfg.carrier_frequency,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            min_speed=speed,
            max_speed=speed,
        )
        for _ in range(num_tx)
    ]
    return IndependentLinkChannel(links)


def build_system_level_channel(cfg: SimConfig):
    from sionna.phy.channel.tr38901 import UMa, UMi

    kwargs = dict(
        carrier_frequency=cfg.carrier_frequency,
        o2i_model=cfg.o2i_model,
        ut_array=make_array(cfg.resolved_ue_ant, cfg.carrier_frequency, bs=False),
        bs_array=make_array(cfg.num_rx_ant, cfg.carrier_frequency, bs=True),
        direction="uplink",
        enable_pathloss=cfg.enable_pathloss,
        enable_shadow_fading=cfg.enable_shadow_fading,
    )
    if cfg.channel == "umi":
        return UMi(**kwargs)
    if cfg.channel == "uma":
        return UMa(**kwargs)
    raise ValueError(f"Not a system-level model: {cfg.channel}")


def set_system_topology(channel_model, cfg: SimConfig, batch_size: int, num_ut: int) -> None:
    from sionna.phy.channel import gen_single_sector_topology

    key = (batch_size, num_ut)
    if getattr(channel_model, "_nrul_alloc", None) != key:
        if hasattr(channel_model, "allocate_topology_tensors"):
            channel_model.allocate_topology_tensors(
                batch_size=batch_size, num_bs=1, num_ut=num_ut
            )
        channel_model._nrul_alloc = key
    channel_model.set_topology(
        *gen_single_sector_topology(
            batch_size,
            num_ut,
            cfg.channel,
            min_ut_velocity=cfg.speed_mps,
            max_ut_velocity=cfg.speed_mps,
        )
    )
