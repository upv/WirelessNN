"""Delay-domain (tap) windowed DMRS channel estimation.

Both estimators start from Sionna's CDM-aware LS estimate at the DMRS
positions (OCC despreading over the two pilots of a 4-subcarrier block).
Per DMRS OFDM symbol the ``M`` block estimates are taken to the delay domain
with an ``M``-point IDFT, denoised, and transformed back onto every used
subcarrier with an ``N``-point DFT (``N = num_effective_subcarriers``), which
is the exact interpolation model for a channel whose taps lie inside the
kept window. Estimates are then interpolated linearly across OFDM symbols.

Tap spacing is ``1 / (N * subcarrier_spacing)``; the unambiguous delay range
of the block grid is ``M`` taps. Positive delays occupy taps ``0 .. M/2-1``,
the wrapped end of the vector ``M/2 .. M-1`` holds "negative" delays that the
band-limited (Dirichlet) leakage of the first path and any residual timing
offset produce. Leakage is the price of a rectangular band: a path at a
fractional delay spreads as ``1/(pi k)`` over the taps, so the energy outside
a window of ``L`` taps on either side is about ``1/(pi^2 L)``. With the
default 1 us (24 taps) negative window the hard estimator floors near
-24 dB NMSE on CDL-C; below ~15 dB SNR it beats linear interpolation.

hard window
    Taps inside ``[-neg, +pos]`` are kept, all other taps are set to zero.
    The estimation-error variance shrinks by ``kept / M``.

soft window
    Per-tap Wiener gain ``g_k = max(P_k - a*s2, 0) / P_k`` where ``P_k`` is the
    tap power averaged over receive antennas and DMRS symbols and ``s2`` the
    noise power per tap (``err_var / M``). ``a`` is a threshold factor that
    protects against noise taps whose sample power exceeds the mean. Optionally
    the gain is applied inside the hard window only.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from sionna.phy.nr import PUSCHLSChannelEstimator


class WindowedLSChannelEstimator(PUSCHLSChannelEstimator):
    """Sionna PUSCH LS + delay-domain window + DFT interpolation."""

    def __init__(
        self,
        resource_grid,
        dmrs_length: int,
        dmrs_additional_position: int,
        num_cdm_groups_without_data: int,
        *,
        mode: str = "hard",
        window_pos_us: float = 3.0,
        window_neg_us: float = 1.0,
        soft_threshold: float = 1.5,
        soft_within_window: bool = False,
        time_interp: str = "linear",
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            resource_grid,
            dmrs_length,
            dmrs_additional_position,
            num_cdm_groups_without_data,
            interpolation_type="nn",
            precision=precision,
            device=device,
            **kwargs,
        )
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'")
        if time_interp not in ("linear", "avg"):
            raise ValueError("time_interp must be 'linear' or 'avg'")
        self.mode = mode
        self.soft_threshold = float(soft_threshold)
        self.soft_within_window = bool(soft_within_window)
        self.time_interp = time_interp

        rg = resource_grid
        self.num_sc = int(rg.num_effective_subcarriers)
        self.num_sym = int(rg.num_ofdm_symbols)
        self.tap_spacing_s = 1.0 / (self.num_sc * float(rg.subcarrier_spacing))

        mask = self._pilot_pattern.mask.cpu().numpy()  # [Tx, S, Nsym, Nsc]
        pilots = self._pilot_pattern.pilots.cpu().numpy()  # [Tx, S, Npil]
        num_tx, num_s = mask.shape[:2]
        dmrs_syms = np.where(mask[0, 0].any(axis=1))[0]
        self.dmrs_syms = [int(s) for s in dmrs_syms]
        n_dmrs = len(self.dmrs_syms)
        pil_per_sym = int(mask[0, 0].sum() // n_dmrs)
        if pil_per_sym != self.num_sc:
            raise ValueError("windowed CE expects DMRS on every subcarrier of a DMRS symbol")

        spacing = 2 * num_cdm_groups_without_data  # subcarriers per despread block
        self.block_spacing = spacing
        self.num_blocks = self.num_sc // spacing
        m = self.num_blocks
        if m * spacing != self.num_sc:
            raise ValueError("number of subcarriers must be a multiple of the DMRS block")

        # Per stream: pilot index (within the flat pilot vector) of one
        # representative per block, and the block-centre offset ``c + 1``.
        gather = np.zeros((num_tx, num_s, n_dmrs, m), dtype=np.int64)
        centre = np.zeros((num_tx, num_s), dtype=np.int64)
        for t in range(num_tx):
            for s in range(num_s):
                first = pilots[t, s, :pil_per_sym]
                nz = np.where(np.abs(first) > 0)[0]
                if nz.size != 2 * m:
                    raise ValueError("unexpected DMRS comb layout")
                rep = nz[0::2]
                c = int(nz[0])
                if not np.all(rep == np.arange(m) * spacing + c):
                    raise ValueError("unexpected DMRS comb layout")
                centre[t, s] = c + 1
                for i in range(n_dmrs):
                    gather[t, s, i] = i * pil_per_sym + rep
        self.register_buffer("_gather_idx", torch.as_tensor(gather, device=self.device))
        self._centre = centre

        # Delay-domain window
        pos_taps = int(math.floor(float(window_pos_us) * 1e-6 / self.tap_spacing_s))
        neg_taps = int(math.floor(float(window_neg_us) * 1e-6 / self.tap_spacing_s))
        half = (m + 1) // 2
        pos_taps = max(0, min(pos_taps, half - 1))
        neg_taps = max(0, min(neg_taps, m - half))
        self.window_pos_taps = pos_taps
        self.window_neg_taps = neg_taps
        win = np.zeros(m, dtype=np.float32)
        win[: pos_taps + 1] = 1.0
        if neg_taps:
            win[m - neg_taps :] = 1.0
        self.register_buffer("_window", torch.as_tensor(win, device=self.device))
        self.num_kept_taps = int(win.sum())

        # Position of tap k in the length-N zero-padded vector.
        k = np.arange(m)
        pad_idx = np.where(k < half, k, self.num_sc - (m - k))
        self.register_buffer("_pad_idx", torch.as_tensor(pad_idx, device=self.device))

        # Time interpolation weights [Nsym, Ndmrs]
        w = np.zeros((self.num_sym, n_dmrs), dtype=np.float32)
        t_d = np.asarray(self.dmrs_syms, dtype=float)
        for l in range(self.num_sym):
            if time_interp == "avg" or n_dmrs == 1:
                w[l] = 1.0 / n_dmrs
                continue
            if l <= t_d[0]:
                w[l, 0] = 1.0
            elif l >= t_d[-1]:
                w[l, -1] = 1.0
            else:
                j = int(np.searchsorted(t_d, l, side="right")) - 1
                frac = (l - t_d[j]) / (t_d[j + 1] - t_d[j])
                w[l, j] = 1.0 - frac
                w[l, j + 1] = frac
        self.register_buffer("_time_w", torch.as_tensor(w, device=self.device))

    # ------------------------------------------------------------------ #
    def _block_estimates(self, h_p: torch.Tensor) -> torch.Tensor:
        """[B, Rx, RxAnt, Tx, S, Npil] -> [B, Rx, RxAnt, Tx, S, Ndmrs, M]."""
        b, rx, rxa, tx, s, _ = h_p.shape
        idx = self._gather_idx.to(h_p.device)
        idx = idx.reshape(1, 1, 1, tx, s, -1).expand(b, rx, rxa, -1, -1, -1)
        out = torch.gather(h_p, -1, idx)
        return out.reshape(b, rx, rxa, tx, s, idx.shape[-1] // self.num_blocks, self.num_blocks)

    def _tap_statistics(self, taps: torch.Tensor, err_p: torch.Tensor):
        """Tap power and per-tap noise power, both [B, 1, 1, Tx, S, 1, M]."""
        m = taps.shape[-1]
        power = taps.abs().square().mean(dim=(1, 2, 5), keepdim=True)
        noise = (err_p.mean(dim=(1, 2, 5), keepdim=True) / m).expand_as(power)
        return power, noise

    def _tap_gains(self, taps: torch.Tensor, err_p: torch.Tensor) -> torch.Tensor:
        """Per-tap gain [B, 1, 1, Tx, S, 1, M]."""
        win = self._window.to(taps.device).reshape(1, 1, 1, 1, 1, 1, -1)
        power, noise = self._tap_statistics(taps, err_p)
        if self.mode == "hard":
            return win.expand_as(power)
        gain = torch.clamp(
            (power - self.soft_threshold * noise) / torch.clamp(power, min=1e-30), min=0.0, max=1.0
        )
        if self.soft_within_window:
            gain = gain * win
        return gain

    def _error_variance(self, taps, err_p, gain) -> torch.Tensor:
        """Per-subcarrier error variance of the windowed estimate, [B, 1, 1, Tx, S, 1, 1].

        Tap k contributes ``(1-g_k)^2 P_h,k`` (channel energy removed by the
        window, with ``P_h,k = max(P_k - s2, 0)`` its estimate) plus
        ``g_k^2 s2`` (noise kept). The DFT back to the subcarriers sums these.
        """
        power, noise = self._tap_statistics(taps, err_p)
        removed = (1.0 - gain).square() * torch.clamp(power - noise, min=0.0)
        kept = gain.square() * noise
        return (removed + kept).sum(dim=-1, keepdim=True)

    def _to_full_grid(self, taps: torch.Tensor, centre: np.ndarray) -> torch.Tensor:
        """Taps [..., Tx, S, Nd, M] -> channel on all subcarriers [..., Tx, S, Nd, N]."""
        n = self.num_sc
        padded = torch.zeros(taps.shape[:-1] + (n,), dtype=taps.dtype, device=taps.device)
        idx = self._pad_idx.to(taps.device)
        padded.index_copy_(-1, idx, taps)
        g = torch.fft.fft(padded, dim=-1)
        out = torch.empty_like(g)
        for t in range(g.shape[-4]):
            for s in range(g.shape[-3]):
                out[..., t, s, :, :] = torch.roll(g[..., t, s, :, :], shifts=int(centre[t, s]), dims=-1)
        return out

    def call(self, y: torch.Tensor, no: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_pilots = self._extract_pilots(y)
        h_p, err_p = self.estimate_at_pilot_locations(y_pilots, no)
        err_p = torch.broadcast_to(err_p, h_p.shape)
        h_blk = self._block_estimates(h_p)  # [B,Rx,RxAnt,Tx,S,Nd,M]
        err_blk = self._block_estimates(err_p.to(h_p.dtype)).real
        taps = torch.fft.ifft(h_blk, dim=-1)
        gain = self._tap_gains(taps, err_blk)
        err_d = self._error_variance(taps, err_blk, gain)  # [B,1,1,Tx,S,1,1]
        taps = taps * gain.to(taps.dtype)
        h_d = self._to_full_grid(taps, self._centre)  # [B,Rx,RxAnt,Tx,S,Nd,N]
        err_d = torch.broadcast_to(err_d, h_d.shape)

        # Linear combination across DMRS symbols: variance follows the squared weights.
        w = self._time_w.to(h_d.device)
        h_hat = torch.einsum("ld,...dn->...ln", w.to(h_d.dtype), h_d)
        err_var = torch.einsum("ld,...dn->...ln", w * w, err_d)
        return h_hat.contiguous(), err_var.contiguous()
