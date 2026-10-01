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


def effective_channel(h, guard_carriers, w=None, scale: float = 1.0):
    """True H on used subcarriers, times codebook W if present and the layer power scale."""
    h = strip_guard_subcarriers(h, guard_carriers)
    if w is not None:
        h = h.permute(0, 1, 3, 5, 6, 2, 4)
        h = torch.matmul(h, w)
        h = h.permute(0, 1, 5, 2, 6, 3, 4)
    if scale != 1.0:
        h = h * scale
    return h


def pilot_grid(pilot_pattern) -> torch.Tensor:
    """DMRS symbols on the effective grid: [num_tx, num_streams, num_sym, num_sc]."""
    mask = pilot_pattern.mask.cpu().numpy()
    pilots = pilot_pattern.pilots.cpu().numpy()
    grid = np.zeros(mask.shape, dtype=pilots.dtype)
    for t in range(mask.shape[0]):
        for s in range(mask.shape[1]):
            grid[t, s][np.where(mask[t, s])] = pilots[t, s]
    return torch.as_tensor(grid)


class PuschRx:
    """One or more DMRS channel estimators, then linear MIMO + TB decode."""

    def __init__(self, transmitter, cfg: SimConfig):
        sm = StreamManagement(np.ones([1, cfg.num_ue], dtype=bool), cfg.num_layers)
        bits = transmitter._num_bits_per_symbol
        bits = int(bits.reshape(-1)[0] if hasattr(bits, "reshape") else bits)
        self.irc_eq = ExtraCovarianceLMMSE()
        self.w = _precoding_w(transmitter)
        self.pilot_grid = pilot_grid(transmitter.pilot_pattern)
        self.dmrs_syms = [
            int(s) for s in np.where(self.pilot_grid[0, 0].abs().numpy().any(axis=1))[0]
        ]
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
        h_hat, err_var = estimator(y, no)
        if name == "ls_lin_time_avg" and len(self.dmrs_syms) > 1:
            # Sionna averages the estimates of all DMRS symbols but keeps the
            # single-symbol error variance; the average of n independent
            # estimates has 1/n of it.
            err_var = err_var / len(self.dmrs_syms)
        return h_hat, err_var

    def decode(self, y, h_hat, err_var, no, name: str):
        llr = self.detectors[name](y, h_hat, err_var, no)
        llr = self.layer_demapper(llr)
        b_hat, _crc = self.tb_decoder(llr)
        return b_hat


def build_receivers(transmitter, cfg: SimConfig) -> PuschRx:
    return PuschRx(transmitter, cfg)
