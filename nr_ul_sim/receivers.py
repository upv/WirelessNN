from __future__ import annotations

import numpy as np
import torch
from sionna.phy.mimo import StreamManagement, lmmse_equalizer
from sionna.phy.nr import LayerDemapper, TBDecoder
from sionna.phy.ofdm import LinearDetector

from .channel_estimation import build_estimator
from .parameters import SimConfig
from .pusch import strip_guard_subcarriers

EQUALIZER_BY_NAME = {
    "mr": "mf",
    "lmmse": "lmmse",
    "ideal_mmse": "lmmse",
    "zf": "zf",
}


class ExtraCovarianceLMMSE:
    """LMMSE with S = N0 I + R_iot. R_iot is [batch, rx, rx]."""

    def __init__(self):
        self.r_int = 0.0

    def set_covariance(self, r_int: torch.Tensor | float) -> None:
        self.r_int = r_int

    def __call__(self, y, h, s, precision=None, **kwargs):
        r = self.r_int
        if torch.is_tensor(r):
            r = r.to(device=s.device, dtype=s.dtype)
            r = r.reshape(r.shape[0], *([1] * (s.ndim - r.ndim)), r.shape[-2], r.shape[-1])
            s = s + r
        return lmmse_equalizer(y, h, s, whiten_interference=True, precision=precision)


def _precoding_w(transmitter):
    if transmitter._precoding != "codebook":
        return None
    w = getattr(transmitter._precoder, "_w", None)
    if w is None:
        return None
    from sionna.phy.utils import insert_dims

    return insert_dims(w, 2, 1)


def effective_channel(h, guard_carriers, w=None):
    """True H on used subcarriers, times codebook W if present."""
    h = strip_guard_subcarriers(h, guard_carriers)
    if w is None:
        return h
    h = h.permute(0, 1, 3, 5, 6, 2, 4)
    h = torch.matmul(h, w)
    return h.permute(0, 1, 5, 2, 6, 3, 4)


class PuschRx:
    """One or more DMRS channel estimators, then linear MIMO + TB decode."""

    def __init__(self, transmitter, cfg: SimConfig):
        sm = StreamManagement(np.ones([1, cfg.num_ue], dtype=bool), cfg.num_layers)
        bits = transmitter._num_bits_per_symbol
        bits = int(bits.reshape(-1)[0] if hasattr(bits, "reshape") else bits)
        self.irc_eq = ExtraCovarianceLMMSE()
        self.w = _precoding_w(transmitter)
        self.estimators = {
            name: build_estimator(name, transmitter, cfg)
            for name in cfg.channel_estimators
        }
        self.detectors = {}
        for name in cfg.receivers:
            self.detectors[name] = LinearDetector(
                equalizer=self.irc_eq if name == "irc" else EQUALIZER_BY_NAME[name],
                output="bit",
                demapping_method="maxlog",
                resource_grid=transmitter.resource_grid,
                stream_management=sm,
                constellation_type="qam",
                num_bits_per_symbol=bits,
            )
        self.layer_demapper = LayerDemapper(transmitter._layer_mapper, num_bits_per_symbol=bits)
        self.tb_decoder = TBDecoder(transmitter._tb_encoder)

    def estimate(self, y, no, name: str | None = None):
        if name is None:
            name = next((n for n in self.estimators if n != "perfect"), None)
            if name is None:
                raise ValueError("no DMRS estimator configured")
        estimator = self.estimators[name]
        if estimator is None:
            raise ValueError("perfect CSI is the true channel, not a DMRS estimator")
        if not torch.is_tensor(no):
            no = torch.tensor(no, device=y.device, dtype=y.real.dtype)
        return estimator(y, no)

    def decode(self, y, h_hat, err_var, no, name: str):
        llr = self.detectors[name](y, h_hat, err_var, no)
        llr = self.layer_demapper(llr)
        b_hat, _crc = self.tb_decoder(llr)
        return b_hat


def build_receivers(transmitter, cfg: SimConfig) -> PuschRx:
    return PuschRx(transmitter, cfg)
