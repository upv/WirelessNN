"""Channel estimators from two papers, on the PUSCH DMRS of this simulator.

EqDeepRx (Honkala et al., arXiv 2602.11834), Sec. II-B / III-A:

``ls_fir``         LS at the DMRS, a static frequency-domain smoothing filter
                   over the pilots, linear interpolation (the paper's baseline).
``denoise_nn``     LS, the learned DenoiseNN instead of the static filter
                   (per RX-antenna / layer pilot grid, 1D residual convolutions
                   along frequency with down/up-sampling, a 1x1 mixer across
                   the DMRS symbols), linear interpolation.

A-MMSE (Ha et al., arXiv 2506.00452):

``lmmse_data``     2D LMMSE with the time-frequency covariance estimated from
                   training channels (the paper's practical "LMMSE mismatch").
``lmmse_data_1d``  1D frequency-domain LMMSE (paper eq. 9) with the same
                   covariance, linear interpolation over time.
``a_mmse``         fixed linear filter learned with the two-stage (frequency,
                   then temporal) Attention encoder and a residual FC decoder.
``ra_a_mmse``      rank-reduced A-MMSE filter W U V^T.

Adaptation to the 5G NR uplink of this simulator:

* Every estimator works on the CDM-despread LS estimates of one RX antenna and
  one layer: per DMRS symbol one value per 4-subcarrier DMRS block, at the
  block centre ``4k + c + 1`` (``c`` = CDM-group comb offset). 68 PRB give
  2 x 204 inputs per layer and antenna.
* A-MMSE learns its filter per chunk of 4 PRB (48 subcarriers x 14 symbols
  from 2 x 12 pilot values) and applies the same filter to every chunk; the
  paper's transformer is sized for 6 RB and does not scale to 816 x 14
  outputs. One filter is trained per SNR and comb offset; the receiver picks
  the one closest to its noise level.
* The learned estimators (DenoiseNN, A-MMSE) report an error variance from a
  table measured on validation data; the LMMSE ones report it analytically.

The trained weights and statistics live in ``models/paper_ce`` and are made
by ``scripts/training/train_paper_ce.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .windowed_ce import WindowedLSChannelEstimator

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "paper_ce"
CHUNK_PRB = 4
BLOCK_SC = 4


# --------------------------------------------------------------------------- #
# shared front end: despread LS block estimates, interpolation
# --------------------------------------------------------------------------- #

def linear_interp_matrix(num_sc: int, positions: np.ndarray) -> np.ndarray:
    """[num_sc, len(positions)] linear interpolation, constant beyond the ends."""
    m = len(positions)
    out = np.zeros((num_sc, m), dtype=np.float32)
    for n in range(num_sc):
        if n <= positions[0]:
            out[n, 0] = 1.0
        elif n >= positions[-1]:
            out[n, -1] = 1.0
        else:
            j = int(np.searchsorted(positions, n, side="right")) - 1
            t = (n - positions[j]) / (positions[j + 1] - positions[j])
            out[n, j] = 1.0 - t
            out[n, j + 1] = t
    return out


class BlockEstimator(WindowedLSChannelEstimator):
    """Despread LS block estimates plus a per-estimator ``process`` step."""

    def __init__(self, resource_grid, dmrs_length, dmrs_additional_position,
                 num_cdm_groups_without_data, **kwargs):
        super().__init__(resource_grid, dmrs_length, dmrs_additional_position,
                         num_cdm_groups_without_data, mode="hard", **kwargs)
        self.rg = resource_grid
        # comb offset c = centre - 1 of every (tx, stream)
        self.comb = (np.asarray(self._centre) - 1).astype(int)
        self._interp = {}
        for c in range(self.block_spacing // 2):  # comb offsets of all CDM groups
            pos = np.arange(self.num_blocks) * self.block_spacing + c + 1
            self._interp[int(c)] = torch.as_tensor(linear_interp_matrix(self.num_sc, pos))

    # -- helpers ----------------------------------------------------------
    def blocks(self, y, no):
        """[B,1,Rx,Tx,S,Nd,M] block estimates and the LS error variance per block [B]."""
        y_p = self._extract_pilots(y)
        h_p, err_p = self.estimate_at_pilot_locations(y_p, no)
        err_p = torch.broadcast_to(err_p, h_p.shape)
        h_blk = self._block_estimates(h_p)
        err_blk = self._block_estimates(err_p.to(h_p.dtype)).real
        var = err_blk.reshape(err_blk.shape[0], -1).mean(dim=1)
        return h_blk, var

    def per_stream(self, x: torch.Tensor, fn):
        """Apply ``fn(x_ts, comb)`` to every (tx, stream) slice of dim 3/4."""
        outs = []
        for t in range(x.shape[3]):
            row = [fn(x[:, :, :, t, s], int(self.comb[t, s])) for s in range(x.shape[4])]
            outs.append(torch.stack(row, dim=3))
        return torch.stack(outs, dim=3)

    def freq_interp(self, blk: torch.Tensor, comb: int) -> torch.Tensor:
        f = self._interp[comb].to(blk.device, blk.real.dtype)
        return torch.einsum("nm,...m->...n", f.to(blk.dtype), blk)

    def interp_time(self, h_d: torch.Tensor) -> torch.Tensor:
        w = self._time_w.to(h_d.device, h_d.real.dtype).to(h_d.dtype)
        return torch.einsum("ld,...dn->...ln", w, h_d)

    def interp_grid(self, blk: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of block values [B,1,Rx,Tx,S,Nd,M] to [..,14,N]."""
        h_d = self.per_stream(blk, self.freq_interp)
        return self.interp_time(h_d)

    @staticmethod
    def snr_db_from_block_var(var: torch.Tensor) -> torch.Tensor:
        """Per-antenna SNR of the slot from the despread LS variance N0 / 4."""
        return -10.0 * torch.log10(4.0 * var.clamp(min=1e-30))

    def call(self, y, no):
        h_blk, var = self.blocks(y, no)
        return self.process(h_blk, var)

    def process(self, h_blk, var):  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# EqDeepRx baseline: static frequency-domain smoothing filter
# --------------------------------------------------------------------------- #

def smoothing_taps(num_taps: int, block_spacing_hz: float, tau_lo: float, tau_hi: float):
    """Hann-windowed complex FIR passing delays in [tau_lo, tau_hi] (seconds).

    With h[k] = sum_p a_p exp(-j 2 pi k df tau_p), the output sum_m g[m] h[k-m]
    scales path p by G(tau_p) = sum_m g[m] exp(+j 2 pi m df tau_p).
    Returns (complex taps g, real low-pass prototype used for edge normalisation).
    """
    k = np.arange(num_taps) - num_taps // 2
    width = tau_hi - tau_lo
    centre = 0.5 * (tau_lo + tau_hi)
    lp = block_spacing_hz * width * np.sinc(k * block_spacing_hz * width)
    lp = lp * np.hanning(num_taps + 2)[1:-1]
    lp = lp / lp.sum()
    g = lp * np.exp(-2j * np.pi * k * block_spacing_hz * centre)
    return g.astype(np.complex64), lp.astype(np.float32)


class FirSmoothingEstimator(BlockEstimator):
    """EqDeepRx baseline: LS, static frequency smoothing, linear interpolation."""

    def __init__(self, *args, num_taps: int = 17, tau_lo_us: float = -1.0,
                 tau_hi_us: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        df = self.block_spacing * float(self.rg.subcarrier_spacing)
        g, lp = smoothing_taps(num_taps, df, tau_lo_us * 1e-6, tau_hi_us * 1e-6)
        self.taps = torch.as_tensor(g)
        m = self.num_blocks
        half = num_taps // 2
        # dense [M, M] smoothing matrix with renormalisation at the band edges
        mat = np.zeros((m, m), dtype=np.complex64)
        for i in range(m):
            lo, hi = max(0, i - half), min(m, i + half + 1)
            idx = np.arange(lo, hi)
            taps = g[i - idx + half]
            mat[i, lo:hi] = taps / lp[i - idx + half].sum()
        self.smooth = torch.as_tensor(mat)
        self.noise_gain = float((np.abs(g) ** 2).sum())

    def process(self, h_blk, var):
        s = self.smooth.to(h_blk.device)
        sm = torch.einsum("km,...m->...k", s, h_blk)
        h_hat = self.interp_grid(sm)
        err = (var * self.noise_gain).reshape(-1, 1, 1, 1, 1, 1, 1)
        return h_hat, torch.broadcast_to(err, h_hat.shape).contiguous()


# --------------------------------------------------------------------------- #
# EqDeepRx DenoiseNN
# --------------------------------------------------------------------------- #

class SubsampledResBlock(nn.Module):
    """Conv (stride 2 along frequency) -> GELU -> conv -> upsample, residual."""

    def __init__(self, ch: int, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.down = nn.Conv2d(ch, ch, (1, kernel), stride=(1, 2), padding=(0, pad))
        self.conv = nn.Conv2d(ch, ch, (1, kernel), padding=(0, pad))
        self.act = nn.GELU()

    def forward(self, x):
        z = self.conv(self.act(self.down(x)))
        z = nn.functional.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + z


class SymbolMixer(nn.Module):
    """1x1 mixing of a channel subset across the DMRS symbols, per subcarrier."""

    def __init__(self, mix_ch: int, num_sym: int):
        super().__init__()
        self.mix_ch, self.num_sym = mix_ch, num_sym
        self.conv = nn.Conv2d(mix_ch * num_sym, mix_ch * num_sym, 1)

    def forward(self, x):
        b, c, s, f = x.shape
        sub = x[:, : self.mix_ch].reshape(b, self.mix_ch * s, 1, f)
        sub = self.conv(sub).reshape(b, self.mix_ch, s, f)
        return torch.cat([x[:, : self.mix_ch] + sub, x[:, self.mix_ch:]], dim=1)


class DenoiseNN(nn.Module):
    """Pilot-domain denoiser: [B, 2 (re/im), Nd, M] -> same shape."""

    def __init__(self, num_sym: int = 2, channels: int = 32, depth: int = 4,
                 mix_ch: int = 8, kernel: int = 5):
        super().__init__()
        self.config = dict(num_sym=num_sym, channels=channels, depth=depth, mix_ch=mix_ch, kernel=kernel)
        self.stem = nn.Conv2d(2, channels, (1, kernel), padding=(0, kernel // 2))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(SubsampledResBlock(channels, kernel))
            self.blocks.append(SymbolMixer(mix_ch, num_sym))
        self.head = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        z = self.stem(x)
        for blk in self.blocks:
            z = blk(z)
        return x + self.head(z)


def complex_to_ch(x: torch.Tensor) -> torch.Tensor:
    """[..., Nd, M] complex -> [N, 2, Nd, M] real."""
    return torch.stack([x.real, x.imag], dim=-3).reshape(-1, 2, *x.shape[-2:])


def ch_to_complex(x: torch.Tensor, shape) -> torch.Tensor:
    return torch.complex(x[:, 0], x[:, 1]).reshape(shape)


class ErrVarTable:
    """Error variance vs SNR, measured on validation data (interpolated in dB)."""

    def __init__(self, snr_db, mse):
        self.snr = np.asarray(snr_db, dtype=float)
        self.log_mse = np.log10(np.maximum(np.asarray(mse, dtype=float), 1e-12))

    def __call__(self, snr_db: torch.Tensor) -> torch.Tensor:
        s = snr_db.detach().cpu().double().numpy()
        v = 10.0 ** np.interp(s, self.snr, self.log_mse)
        return torch.as_tensor(v, dtype=torch.float32, device=snr_db.device)


class DenoiseNNEstimator(BlockEstimator):
    def __init__(self, *args, model_dir=DEFAULT_MODEL_DIR, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt = torch.load(Path(model_dir) / "denoise_nn.pt", map_location="cpu", weights_only=False)
        self.net = DenoiseNN(**ckpt["config"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.err_table = ErrVarTable(ckpt["errvar_snr_db"], ckpt["errvar_mse"])

    @torch.no_grad()
    def process(self, h_blk, var):
        net = self.net.to(h_blk.device)
        den = ch_to_complex(net(complex_to_ch(h_blk)), h_blk.shape)
        h_hat = self.interp_grid(den)
        err = self.err_table(self.snr_db_from_block_var(var)).reshape(-1, 1, 1, 1, 1, 1, 1)
        return h_hat, torch.broadcast_to(err, h_hat.shape).contiguous()


# --------------------------------------------------------------------------- #
# A-MMSE paper: data-driven LMMSE (2D and 1D)
# --------------------------------------------------------------------------- #

def freq_cov_from_samples(h: torch.Tensor) -> torch.Tensor:
    """Sum over samples of the time-frequency autocorrelation (wide-sense stationary in f).

    h: [S, T, N] channel grids. Returns (sum R[t1, t2, d], counts[d]) with
    d = Δn + N - 1 and R[t1, t2, Δn] = sum_n h[t1, n + Δn] h*[t2, n].
    """
    s, t, n = h.shape
    nfft = 1 << int(math.ceil(math.log2(2 * n - 1)))
    f = torch.fft.fft(h, n=nfft, dim=-1)                      # [S, T, nfft]
    cross = torch.einsum("sak,sbk->abk", f, f.conj())         # [T, T, nfft]
    r = torch.fft.ifft(cross, dim=-1)                         # lag axis, circular
    lags = torch.arange(-(n - 1), n, device=h.device) % nfft
    return r[..., lags]


class DataCovariance:
    """R[t1, t2, Δn] = E[H(t1, n + Δn) H*(t2, n)] estimated from training grids."""

    def __init__(self, r: torch.Tensor, num_sc: int):
        self.r = r            # [T, T, 2N-1]
        self.num_sc = num_sc

    def at(self, t1, t2, dn):
        """Broadcast lookup; lags outside the band are zero."""
        n = self.num_sc
        idx = dn + n - 1
        valid = (idx >= 0) & (idx < 2 * n - 1)
        out = self.r[t1, t2, idx.clamp(0, 2 * n - 2)]
        return torch.where(valid, out, torch.zeros_like(out))


def lmmse_matrices(cov: DataCovariance, dmrs_syms, num_blocks: int, spacing: int, comb: int,
                   num_sym: int, num_sc: int, device):
    """R_xx [L, L] and R_hx [T*N, L] for x = despread pilot blocks (average of
    the two pilots at 4k + c and 4k + c + 2 on each DMRS symbol)."""
    d = torch.as_tensor(dmrs_syms, device=device)
    k = torch.arange(num_blocks, device=device)
    pilot_sc = [spacing * k + comb, spacing * k + comb + 2]   # two pilots per block
    nd = len(dmrs_syms)
    # x index = (d, k)
    t_x = d.repeat_interleave(num_blocks)                      # [L]
    rxx = 0
    for a in pilot_sc:
        for b in pilot_sc:
            pa = a.repeat(nd)
            pb = b.repeat(nd)
            rxx = rxx + cov.at(t_x[:, None], t_x[None, :], pa[:, None] - pb[None, :])
    rxx = rxx / 4.0
    t_h = torch.arange(num_sym, device=device).repeat_interleave(num_sc)   # [T*N]
    n_h = torch.arange(num_sc, device=device).repeat(num_sym)
    rhx = 0
    for b in pilot_sc:
        pb = b.repeat(nd)
        rhx = rhx + cov.at(t_h[:, None], t_x[None, :], n_h[:, None] - pb[None, :])
    rhx = rhx / 2.0
    return rxx, rhx


def psd_clip(r: torch.Tensor) -> torch.Tensor:
    """Hermitian part with negative eigenvalues set to zero."""
    r = 0.5 * (r + r.conj().transpose(-1, -2))
    ev, vec = torch.linalg.eigh(r)
    return (vec * ev.clamp(min=0.0).to(vec.dtype)[..., None, :]) @ vec.conj().transpose(-1, -2)


class DataLMMSEEstimator(BlockEstimator):
    """LMMSE with the covariance estimated from training channels.

    ``mode="2d"``: joint time-frequency filter from all DMRS blocks to the
    whole slot. ``mode="1d"``: per DMRS symbol frequency-only filter (A-MMSE
    paper eq. 9) followed by linear interpolation over time.
    """

    def __init__(self, *args, model_dir=DEFAULT_MODEL_DIR, mode: str = "2d", **kwargs):
        super().__init__(*args, **kwargs)
        stats = torch.load(Path(model_dir) / "data_cov.pt", map_location="cpu", weights_only=False)
        r = stats["r"]
        n_train = r.shape[-1] // 2 + 1
        if self.num_sc > n_train:
            raise ValueError(f"covariance trained for {n_train} subcarriers, need {self.num_sc}")
        mid = n_train - 1
        r = r[..., mid - (self.num_sc - 1): mid + self.num_sc]
        if mode == "1d":
            # frequency correlation only: average the per-symbol autocorrelation
            rf = torch.stack([r[i, i] for i in range(r.shape[0])]).mean(dim=0)
            r = rf.reshape(1, 1, -1).expand(self.num_sym, self.num_sym, -1).clone()
        self.mode = mode
        self.cov_cpu = r
        self._cache: dict = {}

    def _filters(self, comb: int, var: float, device):
        key = (comb, round(10 * math.log10(max(var, 1e-30)), 1), str(device))
        if key not in self._cache:
            cov = DataCovariance(self.cov_cpu.to(device), self.num_sc)
            if self.mode == "2d":
                rxx, rhx = lmmse_matrices(cov, self.dmrs_syms, self.num_blocks, self.block_spacing,
                                          comb, self.num_sym, self.num_sc, device)
                rxx = psd_clip(rxx)
                eye = torch.eye(rxx.shape[0], device=device, dtype=rxx.dtype)
                w = torch.linalg.solve((rxx + var * eye).T, rhx.T).T    # rhx (rxx + var I)^-1, [T*N, L]
                err = (cov.r[torch.arange(self.num_sym), torch.arange(self.num_sym), self.num_sc - 1].real.mean()
                       - (w * rhx.conj()).sum(-1).real.mean())
            else:
                rxx, rhx = lmmse_matrices(cov, [0], self.num_blocks, self.block_spacing,
                                          comb, 1, self.num_sc, device)
                rxx = psd_clip(rxx)
                eye = torch.eye(rxx.shape[0], device=device, dtype=rxx.dtype)
                w = torch.linalg.solve((rxx + var * eye).T, rhx.T).T    # [N, M]
                err = cov.r[0, 0, self.num_sc - 1].real - (w * rhx.conj()).sum(-1).real.mean()
            self._cache[key] = (w, float(err.clamp(min=1e-9)))
        return self._cache[key]

    def process(self, h_blk, var):
        b = h_blk.shape[0]
        outs, errs = [], []
        for i in range(b):  # noise level is per slot; W is cached per level
            v = float(var[i])

            def apply(x, comb):
                w, _ = self._filters(comb, v, x.device)
                if self.mode == "2d":
                    flat = x.reshape(*x.shape[:-2], -1)                       # [.., Nd*M]
                    return (flat @ w.T).reshape(*x.shape[:-2], self.num_sym, self.num_sc)
                return torch.einsum("nm,...dm->...dn", w, x)                  # [.., Nd, N]

            h = self.per_stream(h_blk[i: i + 1], apply)
            if self.mode == "1d":
                h = self.interp_time(h)
            outs.append(h)
            errs.append(self._filters(int(self.comb[0, 0]), v, h_blk.device)[1])
        h_hat = torch.cat(outs, dim=0)
        err = torch.as_tensor(errs, device=h_hat.device, dtype=torch.float32).reshape(-1, 1, 1, 1, 1, 1, 1)
        return h_hat, torch.broadcast_to(err, h_hat.shape).contiguous()


# --------------------------------------------------------------------------- #
# A-MMSE: two-stage Attention encoder that learns the linear filter
# --------------------------------------------------------------------------- #

class EncoderBlock(nn.Module):
    """MHA -> Add & LayerNorm -> FFN -> Add & LayerNorm."""

    def __init__(self, dim: int, heads: int, ffn_mult: int = 2):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_mult * dim), nn.GELU(), nn.Linear(ffn_mult * dim, dim))
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.n1(x + self.mha(x, x, x, need_weights=False)[0])
        return self.n2(x + self.ffn(x))


class AMMSENet(nn.Module):
    """x_in (2L real) -> W (N*M x L complex), A-MMSE paper Sec. IV-B.

    Frequency encoder: scalar embedding to d_e = N with a learned position
    embedding, MHA with L/2 heads. Temporal encoder: projection to N*M, MHA
    with M heads. Residual FC decoder, rows split into Re / Im of W^T.
    """

    def __init__(self, num_pilots: int, num_sc: int, num_sym: int, dec_layers: int = 2):
        super().__init__()
        self.config = dict(num_pilots=num_pilots, num_sc=num_sc, num_sym=num_sym, dec_layers=dec_layers)
        two_l = 2 * num_pilots
        self.l = num_pilots
        self.emb = nn.Linear(1, num_sc)
        self.pos = nn.Parameter(torch.zeros(two_l, num_sc))
        nn.init.normal_(self.pos, std=0.02)
        self.freq = EncoderBlock(num_sc, max(1, num_pilots // 2))
        self.proj = nn.Linear(num_sc, num_sc * num_sym)
        self.temp = EncoderBlock(num_sc * num_sym, num_sym, ffn_mult=1)
        self.dec = nn.ModuleList([nn.Linear(num_sc * num_sym, num_sc * num_sym) for _ in range(dec_layers)])
        self.act = nn.GELU()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """x_in [B, 2L] -> W [B, N*M, L] complex."""
        z = self.emb(x_in.unsqueeze(-1)) + self.pos               # [B, 2L, N]
        z = self.freq(z)
        z = self.temp(self.proj(z))                               # [B, 2L, N*M]
        for lin in self.dec:
            z = z + lin(self.act(z))
        w_out = torch.complex(z[:, : self.l], z[:, self.l:])      # [B, L, N*M]
        return w_out.transpose(1, 2)


class AMMSEEstimator(BlockEstimator):
    """Fixed learned filter per 4-PRB chunk, chosen by the SNR of the slot."""

    def __init__(self, *args, model_dir=DEFAULT_MODEL_DIR, rank_adaptive: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt = torch.load(Path(model_dir) / "a_mmse.pt", map_location="cpu", weights_only=False)
        key = "w_ra" if rank_adaptive else "w"
        self.snr_bank = np.asarray(ckpt["snr_db"], dtype=float)
        self.w = {int(c): ckpt[key][c] for c in ckpt[key]}            # [n_snr, N*M, L]
        self.chunk_blocks = int(ckpt["chunk_blocks"])
        self.chunk_sc = int(ckpt["chunk_sc"])
        if self.num_blocks % self.chunk_blocks:
            raise ValueError("number of DMRS blocks must be a multiple of the A-MMSE chunk")
        tab = ckpt["errvar_ra" if rank_adaptive else "errvar"]
        self.err_table = ErrVarTable(tab["snr_db"], tab["mse"])

    def process(self, h_blk, var):
        snr = self.snr_db_from_block_var(var)
        sel = [int(np.argmin(np.abs(self.snr_bank - float(s)))) for s in snr]
        nchunk = self.num_blocks // self.chunk_blocks
        outs = []
        for i, j in enumerate(sel):
            def apply(x, comb, j=j):
                w = self.w[comb][j].to(x.device)                         # [chunk_sc*T, L]
                xc = x.reshape(*x.shape[:-2], x.shape[-2], nchunk, self.chunk_blocks)
                xc = xc.movedim(-2, -3).reshape(*x.shape[:-2], nchunk, -1)   # [.., C, Nd*Kb]
                hc = (xc @ w.T).reshape(*x.shape[:-2], nchunk, self.num_sym, self.chunk_sc)
                return hc.movedim(-3, -2).reshape(*x.shape[:-2], self.num_sym, self.num_sc)
            outs.append(self.per_stream(h_blk[i: i + 1], apply))
        h_hat = torch.cat(outs, dim=0)
        err = self.err_table(snr).reshape(-1, 1, 1, 1, 1, 1, 1)
        return h_hat, torch.broadcast_to(err, h_hat.shape).contiguous()


def chunk_inputs(blk: torch.Tensor, chunk_blocks: int) -> torch.Tensor:
    """[S, Nd, M] -> [S * C, Nd * Kb] (Nd-major inside a chunk)."""
    s, nd, m = blk.shape
    c = m // chunk_blocks
    return blk.reshape(s, nd, c, chunk_blocks).permute(0, 2, 1, 3).reshape(s * c, nd * chunk_blocks)


def chunk_targets(grid: torch.Tensor, chunk_sc: int) -> torch.Tensor:
    """[S, T, N] -> [S * C, T * chunk_sc] (T-major inside a chunk)."""
    s, t, n = grid.shape
    c = n // chunk_sc
    return grid.reshape(s, t, c, chunk_sc).permute(0, 2, 1, 3).reshape(s * c, t * chunk_sc)


def oas_shrinkage(s: torch.Tensor, n: int) -> torch.Tensor:
    """Shrinkage weight rho for (1-rho) S + rho tr(S)/p I, complex Gaussian data.

    Oracle weight minimising E||rho F + (1-rho) S - Sigma||_F^2 with the random
    target F = tr(S)/p I, using the complex-Wishart moments
    E tr S^2 = a + b/n, E tr^2 S = b + a/n (a = tr Sigma^2, b = tr^2 Sigma):

        rho* = (b/n - a/(n p)) / (a + b/n - b/p - a/(n p)),

    with a, b replaced by their unbiased estimates from S (oracle-approximating
    shrinkage in the spirit of OAS, cf. the complex-valued OAS cited by EqDeepRx).
    """
    p = s.shape[-1]
    tr = torch.diagonal(s, dim1=-2, dim2=-1).real.sum(-1)
    tr_s2 = (s * s.conj()).real.sum((-2, -1))                     # tr(S S^H) = sum |s_ij|^2 = tr(S^2)
    tr2_s = tr ** 2
    a = (n * n * tr_s2 - n * tr2_s) / (n * n - 1)                  # tr Sigma^2
    b = (n * n * tr2_s - n * tr_s2) / (n * n - 1)                  # tr^2 Sigma
    num = b / n - a / (n * p)
    den = a + b / n - b / p - a / (n * p)
    rho = torch.where(den > 0, num / den.clamp(min=1e-30), torch.ones_like(tr))
    return rho.clamp(0.0, 1.0)
