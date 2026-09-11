"""Antenna arrays and 3GPP channel models (CDL-B/C, UMi, UMa)."""

from __future__ import annotations

from typing import Any

import torch
from sionna.phy.channel import ChannelModel

from .parameters import SimConfig


def make_array(num_ant: int, carrier_frequency: float, *, bs: bool):
    """Build a 3GPP 38.901 antenna array with the requested number of elements."""
    from sionna.phy.channel.tr38901 import AntennaArray

    pattern = "38.901" if bs else "omni"
    if num_ant == 1:
        return AntennaArray(
            num_rows=1,
            num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern=pattern,
            carrier_frequency=carrier_frequency,
        )
    if num_ant == 2:
        return AntennaArray(
            num_rows=1,
            num_cols=1,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern=pattern,
            carrier_frequency=carrier_frequency,
        )
    if num_ant == 4:
        return AntennaArray(
            num_rows=1,
            num_cols=2,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern=pattern,
            carrier_frequency=carrier_frequency,
        )
    if num_ant == 8:
        return AntennaArray(
            num_rows=2,
            num_cols=2,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern=pattern,
            carrier_frequency=carrier_frequency,
        )
    raise ValueError(f"Unsupported antenna count {num_ant}; use 1, 2, 4 or 8")


class IndependentLinkChannel(ChannelModel):
    """Stack independent single-link models (CDL) along the transmitter axis."""

    def __init__(self, links: list[Any]):
        super().__init__()
        if not links:
            raise ValueError("At least one link model is required")
        self.links = links

    def __call__(self, batch_size, num_time_steps, sampling_frequency):
        a_list = []
        tau_list = []
        for link in self.links:
            a, tau = link(batch_size, num_time_steps, sampling_frequency)
            a_list.append(a)
            tau_list.append(tau)
        a = torch.cat(a_list, dim=3)
        tau = torch.cat(tau_list, dim=2)
        return a, tau


def _cdl_model_letter(channel: str) -> str:
    mapping = {"cdl-b": "B", "cdl-c": "C"}
    if channel not in mapping:
        raise ValueError(f"{channel} is not a CDL model")
    return mapping[channel]


def build_cdl_links(cfg: SimConfig, num_tx: int):
    from sionna.phy.channel.tr38901 import CDL

    ut_array = make_array(cfg.resolved_ue_ant, cfg.carrier_frequency, bs=False)
    bs_array = make_array(cfg.num_rx_ant, cfg.carrier_frequency, bs=True)
    letter = _cdl_model_letter(cfg.channel)
    speed = cfg.speed_mps
    links = []
    for _ in range(num_tx):
        links.append(
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
        )
    return IndependentLinkChannel(links)


def build_system_level_channel(cfg: SimConfig):
    from sionna.phy.channel.tr38901 import UMa, UMi

    ut_array = make_array(cfg.resolved_ue_ant, cfg.carrier_frequency, bs=False)
    bs_array = make_array(cfg.num_rx_ant, cfg.carrier_frequency, bs=True)
    common = dict(
        carrier_frequency=cfg.carrier_frequency,
        o2i_model=cfg.o2i_model,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="uplink",
        enable_pathloss=cfg.enable_pathloss,
        enable_shadow_fading=cfg.enable_shadow_fading,
    )
    if cfg.channel == "umi":
        return UMi(**common)
    if cfg.channel == "uma":
        return UMa(**common)
    raise ValueError(f"Not a system-level model: {cfg.channel}")


def is_system_level(channel: str) -> bool:
    return channel in ("umi", "uma")


def set_system_topology(channel_model, cfg: SimConfig, batch_size: int, num_ut: int) -> None:
    from sionna.phy.channel import gen_single_sector_topology

    alloc_key = (batch_size, num_ut)
    if getattr(channel_model, "_nrul_alloc", None) != alloc_key:
        if hasattr(channel_model, "allocate_topology_tensors"):
            channel_model.allocate_topology_tensors(
                batch_size=batch_size, num_bs=1, num_ut=num_ut
            )
        channel_model._nrul_alloc = alloc_key
    topology = gen_single_sector_topology(
        batch_size,
        num_ut,
        cfg.channel,
        min_ut_velocity=cfg.speed_mps,
        max_ut_velocity=cfg.speed_mps,
    )
    channel_model.set_topology(*topology)
