"""Linear MIMO receivers: MR, L-MMSE, ZF and IRC."""

from __future__ import annotations

from typing import Callable

import torch

from sionna.phy.mimo import lmmse_equalizer
from sionna.phy.nr import PUSCHReceiver
from sionna.phy.ofdm import LinearDetector

from .parameters import SimConfig
from .pusch import _as_int, build_stream_management


class ExtraCovarianceLMMSE:
    """LMMSE equalizer that adds a spatial interference covariance to ``S``.

    Sionna's OFDM equalizer already builds ``S ≈ N0 I`` (plus intra-cell terms
    from unassociated streams). IRC uses ``S + R_iot`` so that the combiner
    rejects spatially coloured other-cell interference.
    """

    def __init__(self):
        self.r_int = 0.0

    def set_covariance(self, r_int: torch.Tensor | float) -> None:
        self.r_int = r_int

    def __call__(self, y, h, s, precision=None, **kwargs):
        r = self.r_int
        if not torch.is_tensor(r):
            s_tot = s
        else:
            r = r.to(device=s.device, dtype=s.dtype)
            # s is [..., M, M]. Broadcast a [M, M] or per-batch [B, M, M] covariance.
            if r.ndim >= 2:
                r = r.reshape(-1, r.shape[-2], r.shape[-1])
            if r.shape[0] == 1:
                r = r[0]
            elif r.ndim == 3 and r.shape[0] == s.shape[0]:
                r = r.reshape(s.shape[0], *([1] * (s.ndim - 3)), r.shape[-2], r.shape[-1])
            else:
                r = r.mean(dim=0)
            s_tot = s + r
        return lmmse_equalizer(
            y, h, s_tot, whiten_interference=True, precision=precision
        )


def _bits_per_symbol(transmitter) -> int:
    return _as_int(transmitter._num_bits_per_symbol)


def make_linear_detector(
    name: str,
    transmitter,
    cfg: SimConfig,
    irc_equalizer: ExtraCovarianceLMMSE | None = None,
):
    sm = build_stream_management(cfg)
    bits = _bits_per_symbol(transmitter)
    if name == "irc":
        if irc_equalizer is None:
            raise ValueError("IRC detector requires ExtraCovarianceLMMSE")
        equalizer: str | Callable = irc_equalizer
    else:
        equalizer = {"mr": "mf", "lmmse": "lmmse", "zf": "zf"}[name]
    return LinearDetector(
        equalizer=equalizer,
        output="bit",
        demapping_method="maxlog",
        resource_grid=transmitter.resource_grid,
        stream_management=sm,
        constellation_type="qam",
        num_bits_per_symbol=bits,
    )


def make_pusch_receiver(transmitter, detector, cfg: SimConfig) -> PUSCHReceiver:
    kwargs = dict(
        pusch_transmitter=transmitter,
        mimo_detector=detector,
        input_domain=cfg.domain,
    )
    if cfg.perfect_csi:
        kwargs["channel_estimator"] = "perfect"
    return PUSCHReceiver(**kwargs)


def build_receivers(transmitter, cfg: SimConfig):
    """Instantiate every requested receiver sharing the same PUSCH chain."""
    irc_eq = ExtraCovarianceLMMSE()
    receivers = {}
    for name in cfg.receivers:
        det = make_linear_detector(
            name, transmitter, cfg, irc_equalizer=irc_eq if name == "irc" else None
        )
        receivers[name] = make_pusch_receiver(transmitter, det, cfg)
    return receivers, irc_eq
