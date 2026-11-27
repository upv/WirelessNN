"""Channel estimators from the 2026-11 literature survey, adapted to the nr_ul_sim
4x2 PUSCH setting (one sample = one (rx antenna, layer) slice of a slot).

Common interface
  input   x  [B, 2, Nd, M]   OCC-despread DMRS LS, one value per 4-subcarrier block
                             (re/im, Nd DMRS symbols, M blocks = 3 per PRB), RMS-normalised
  output  y  [B, 2, 14, 4M]  channel on the full slot (re/im, symbol, subcarrier)
Every model works for any PRB count (M multiple of 3).

Models (deviations from the papers are listed in each docstring)
  chea      CHEA d=16        Abdollahpour et al., arXiv 2607.16462
  chea_xl   CHEA-XL d=64     same paper
  helena    HELENA           Camelo Botero et al., arXiv 2506.13408
  swift     SwiftChannel     Lyu et al., arXiv 2605.01931 (student network, no KD/QAT)
  denoiserx EqDeepRx DenoiseNN + fixed linear interpolation, Honkala et al., arXiv 2602.11834

Classical baselines in the same interface (no weights)
  hard_window, soft_window   delay-domain windowed LS as nr_ul_sim ls_hard_window / ls_soft_window
  lmmse_exp                  frequency LMMSE with an exponential delay prior (fixed tau)
"""

from __future__ import annotations

import bisect
import math

import torch
import torch.nn.functional as F
from torch import nn

BLOCK_SC = 4          # subcarriers per despread DMRS block
BLOCKS_PER_PRB = 3
NUM_SYM = 14
DEFAULT_DMRS = (2, 11)


# ----------------------------------------------------------------------------- helpers
def to_input(blk: torch.Tensor):
    """Complex LS blocks [N, Nd, M] -> real normalised [N, 2, Nd, M] and RMS scale [N]."""
    scale = blk.reshape(len(blk), -1).abs().pow(2).mean(1).sqrt().clamp(min=1e-12)
    x = blk / scale[:, None, None]
    return torch.stack([x.real, x.imag], 1).float(), scale


def time_interp_matrix(dmrs, n_sym=NUM_SYM) -> torch.Tensor:
    """[n_sym, Nd] piecewise-linear interpolation/extrapolation through the DMRS symbols."""
    d = list(dmrs)
    w = torch.zeros(n_sym, len(d))
    if len(d) == 1:
        w[:, 0] = 1.0
        return w
    for t in range(n_sym):
        j = max(0, min(len(d) - 2, bisect.bisect_right(d, t) - 1))
        a = (t - d[j]) / (d[j + 1] - d[j])
        w[t, j], w[t, j + 1] = 1 - a, a
    return w


class LSGrid(nn.Module):
    """Fixed interpolation of block LS to the slot: linear in frequency (block centres),
    linear in time through the DMRS symbols.  [B, 2, Nd, M] -> [B, 2, 14, 4M]."""

    def __init__(self, dmrs=DEFAULT_DMRS, n_sym=NUM_SYM):
        super().__init__()
        self.register_buffer("w", time_interp_matrix(dmrs, n_sym), persistent=False)

    def forward(self, x):
        b, c, nd, m = x.shape
        xf = F.interpolate(x.reshape(b, c * nd, m), scale_factor=BLOCK_SC, mode="linear",
                           align_corners=False).reshape(b, c, nd, m * BLOCK_SC)
        return torch.einsum("tn,bcnf->bctf", self.w, xf)


def zero_filled_grid(x, dmrs, n_sym=NUM_SYM):
    """Block LS placed on the DMRS symbols (each block value on its 4 subcarriers), zeros elsewhere."""
    b, c, nd, m = x.shape
    g = x.new_zeros(b, c, n_sym, m * BLOCK_SC)
    g[:, :, list(dmrs), :] = x.repeat_interleave(BLOCK_SC, dim=-1)
    return g


class TBlock(nn.Module):
    """Pre-norm transformer block, optional cross-attention."""

    def __init__(self, d, heads, dff, cross=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = cross
        if cross:
            self.lnq, self.lnkv = nn.LayerNorm(d), nn.LayerNorm(d)
            self.catt = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))

    def forward(self, x, ctx=None):
        h = self.ln1(x)
        x = x + self.att(h, h, h, need_weights=False)[0]
        if self.cross:
            q, kv = self.lnq(x), self.lnkv(ctx)
            x = x + self.catt(q, kv, kv, need_weights=False)[0]
        return x + self.ffn(self.ln2(x))


# ----------------------------------------------------------------------------- CHEA
class CHEA(nn.Module):
    """CHEA: multi-resolution windowed attention (arXiv 2607.16462).

    Per stage: high-resolution encoder block inside windows of `hi_win` tokens,
    low-resolution encoder block on tokens average-pooled by `pool` inside windows of
    `hi_win` pooled tokens (= pool x wider), decoder block (self-attention in the
    high-resolution window, cross-attention to the low-resolution window that contains
    it, FFN). A per-PRB linear upsampler maps the PRB's tokens to its 14 x 12 slot,
    added to the fixed LS interpolation.

    Deviations: token = one 4-subcarrier DMRS block with both DMRS symbols (paper: 3
    subcarriers), so 6 PRB = 18 tokens (paper 24); heads d/8; odd stages use windows
    shifted by half a window (paper: not stated); zero padding to whole low-res windows
    plus a validity feature; residual over LS interpolation.
    """

    def __init__(self, d=16, stages=2, heads=None, dff=None, hi_win=18, pool=4,
                 nd=2, dmrs=DEFAULT_DMRS, shift=True, n_sym=NUM_SYM):
        super().__init__()
        heads = heads or max(1, d // 8)
        dff = dff or 2 * d
        self.config = dict(d=d, stages=stages, heads=heads, dff=dff, hi_win=hi_win, pool=pool,
                           nd=nd, dmrs=tuple(dmrs), shift=shift, n_sym=n_sym)
        self.n_sym = n_sym
        self.hi_win, self.pool, self.shift = hi_win, pool, shift
        self.lo_span = hi_win * pool                                  # blocks per low-res window
        self.embed = nn.Linear(2 * nd + 1, d)
        self.pos_hi = nn.Parameter(torch.zeros(hi_win, d))
        self.pos_lo = nn.Parameter(torch.zeros(hi_win, d))
        nn.init.normal_(self.pos_hi, std=0.02)
        nn.init.normal_(self.pos_lo, std=0.02)
        self.hi = nn.ModuleList(TBlock(d, heads, dff) for _ in range(stages))
        self.lo = nn.ModuleList(TBlock(d, heads, dff) for _ in range(stages))
        self.dec = nn.ModuleList(TBlock(d, heads, dff, cross=True) for _ in range(stages))
        self.ln_out = nn.LayerNorm(d)
        self.up = nn.Linear(BLOCKS_PER_PRB * d, 2 * n_sym * BLOCKS_PER_PRB * BLOCK_SC)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.ls = LSGrid(dmrs, n_sym)

    def forward(self, x):
        b, _, nd, m = x.shape
        base = self.ls(x)
        mp = math.ceil(m / self.lo_span) * self.lo_span
        tok = torch.cat([x.reshape(b, 2 * nd, m), x.new_ones(b, 1, m)], 1)
        tok = F.pad(tok, (0, mp - m)).transpose(1, 2)                 # [B, Mp, 2Nd+1]
        h = self.embed(tok)
        nw, nl = mp // self.hi_win, mp // self.lo_span
        for s, (hi, lo, dec) in enumerate(zip(self.hi, self.lo, self.dec)):
            sh = (self.hi_win // 2) if (self.shift and s % 2 == 1) else 0
            if sh:
                h = torch.roll(h, -sh, dims=1)
            hw = h.reshape(b * nw, self.hi_win, -1) + self.pos_hi
            hw = hi(hw)
            l = F.avg_pool1d(hw.reshape(b, mp, -1).transpose(1, 2), self.pool).transpose(1, 2)
            lw = lo(l.reshape(b * nl, self.hi_win, -1) + self.pos_lo)        # [B*nl, hi_win, d]
            ctx = lw.reshape(b, nl, self.hi_win, -1).repeat_interleave(self.pool, dim=1)
            hw = dec(hw, ctx.reshape(b * nw, self.hi_win, -1))
            h = hw.reshape(b, mp, -1)
            if sh:
                h = torch.roll(h, sh, dims=1)
        h = self.ln_out(h[:, :m])                                      # [B, M, d]
        p = m // BLOCKS_PER_PRB
        u = self.up(h.reshape(b, p, -1))                               # [B, P, 2*14*12]
        u = u.reshape(b, p, 2, self.n_sym, BLOCKS_PER_PRB * BLOCK_SC).permute(0, 2, 3, 1, 4)
        return base + u.reshape(b, 2, self.n_sym, m * BLOCK_SC)


def chea_xl(**kw):
    return CHEA(d=64, **kw)


# ----------------------------------------------------------------------------- HELENA
class SE(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, d // r), nn.ReLU(), nn.Linear(d // r, d), nn.Sigmoid())

    def forward(self, x):                                              # [B, T, d]
        return x * self.fc(x.mean(1, keepdim=True))


class HELENA(nn.Module):
    """HELENA (arXiv 2506.13408): two Conv2D+ReLU on the zero-filled LS grid (kernels
    12x2 and 6x7 in subcarrier x symbol), dropout 0.1, per-PRB patches (12 subcarriers x
    14 symbols) embedded to d=64, 4-head self-attention over PRB tokens + LayerNorm,
    squeeze-and-excitation (r=4), linear head per PRB, global residual.

    Deviations: conv widths 16 and 8 (paper: not stated; gives ~0.13M params vs 0.116M);
    the global residual is the fixed LS interpolation instead of the zero-filled input.
    """

    def __init__(self, d=64, heads=4, c1=16, c2=8, dropout=0.1, nd=2, dmrs=DEFAULT_DMRS, n_sym=NUM_SYM):
        super().__init__()
        self.config = dict(d=d, heads=heads, c1=c1, c2=c2, dropout=dropout, nd=nd, dmrs=tuple(dmrs), n_sym=n_sym)
        self.n_sym = n_sym
        self.dmrs = tuple(dmrs)
        self.conv1 = nn.Conv2d(2, c1, (2, 12), padding="same")
        self.conv2 = nn.Conv2d(c1, c2, (7, 6), padding="same")
        self.drop = nn.Dropout(dropout)
        self.psc = BLOCKS_PER_PRB * BLOCK_SC
        self.embed = nn.Linear(c2 * n_sym * self.psc, d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.se = SE(d)
        self.head = nn.Linear(d, 2 * n_sym * self.psc)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.ls = LSGrid(dmrs, n_sym)

    def forward(self, x):
        b, _, nd, m = x.shape
        base = self.ls(x)
        g = zero_filled_grid(x, self.dmrs, self.n_sym)                 # [B, 2, S, N]
        f = self.drop(F.relu(self.conv2(F.relu(self.conv1(g)))))       # [B, c2, 14, N]
        p = m // BLOCKS_PER_PRB
        t = f.reshape(b, f.shape[1], self.n_sym, p, self.psc).permute(0, 3, 1, 2, 4).reshape(b, p, -1)
        t = self.embed(t)
        t = self.ln(t + self.att(t, t, t, need_weights=False)[0])
        t = self.se(t)
        u = self.head(t).reshape(b, p, 2, self.n_sym, self.psc).permute(0, 2, 3, 1, 4)
        return base + u.reshape(b, 2, self.n_sym, m * BLOCK_SC)


# ----------------------------------------------------------------------------- SwiftChannel
class SPAB(nn.Module):
    """Parameter-free attention block: y = (c3 + x) * (sigmoid(c3) - 0.5)."""

    def __init__(self, c, k=3):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, k, padding=k // 2)
        self.c2 = nn.Conv1d(c, c, k, padding=k // 2)
        self.c3 = nn.Conv1d(c, c, k, padding=k // 2)

    def forward(self, x):
        o = self.c3(F.relu(self.c2(F.relu(self.c1(x)))))
        return (o + x) * (torch.sigmoid(o) - 0.5)


class SwiftChannel(nn.Module):
    """SwiftChannel student (arXiv 2605.01931): conv to 12 channels, 4 SPABs, conv +
    pixel shuffle x4 upscaling, ~8k params.

    Deviations: the paper upsamples a 2D (subcarrier x antenna) SRS grid; here each
    (rx antenna, layer) slice is processed alone, so the network is 1D over DMRS blocks
    with (re/im x DMRS symbol) as input channels and (re/im x 14 symbols) x 4 output
    channels shuffled to subcarriers; residual over LS interpolation; no distillation,
    re-parameterisation or QAT (training recipe only).
    """

    def __init__(self, c=12, n_blocks=4, k=3, nd=2, dmrs=DEFAULT_DMRS, n_sym=NUM_SYM):
        super().__init__()
        self.config = dict(c=c, n_blocks=n_blocks, k=k, nd=nd, dmrs=tuple(dmrs), n_sym=n_sym)
        self.n_sym = n_sym
        self.entry = nn.Conv1d(2 * nd, c, k, padding=k // 2)
        self.blocks = nn.Sequential(*(SPAB(c, k) for _ in range(n_blocks)))
        self.tail = nn.Conv1d(c, 2 * n_sym * BLOCK_SC, k, padding=k // 2)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
        self.ls = LSGrid(dmrs, n_sym)

    def forward(self, x):
        b, _, nd, m = x.shape
        base = self.ls(x)
        h = self.entry(x.reshape(b, 2 * nd, m))
        h = self.blocks(h) + h
        u = self.tail(h).reshape(b, 2 * self.n_sym, BLOCK_SC, m)       # pixel shuffle 1D
        u = u.permute(0, 1, 3, 2).reshape(b, 2, self.n_sym, m * BLOCK_SC)
        return base + u


# ----------------------------------------------------------------------------- EqDeepRx DenoiseNN
class SepRes(nn.Module):
    """Residual block of two depthwise-separable (1 x k) convs along frequency, then a
    symbol mixer (learned Nd x Nd mixing per channel, zero-initialised residual)."""

    def __init__(self, c, nd, k=13):
        super().__init__()
        self.dw1 = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c)
        self.pw1 = nn.Conv2d(c, c, 1)
        self.dw2 = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c)
        self.pw2 = nn.Conv2d(c, c, 1)
        self.mix = nn.Parameter(torch.zeros(c, nd, nd))

    def forward(self, x):                                              # [B, C, Nd, M]
        x = x + self.pw2(self.dw2(F.relu(self.pw1(self.dw1(F.relu(x))))))
        return x + torch.einsum("cij,bcjm->bcim", self.mix, x)


class DenoiseRx(nn.Module):
    """EqDeepRx DenoiseNN (arXiv 2602.11834): ResNet on the pilot positions of each
    RX-TX pair, filters [64, 64, 64, 2], subsampling 1, 4, 2, 1, 13x1 kernels along
    frequency, 1x1 symbol mixer after each block; the denoised pilots are then
    interpolated by the fixed linear LS interpolator (as in the paper's receiver).

    Deviations: depthwise-separable convs (paper: not stated) to keep the parameter
    budget small; the 1/8-resolution path is upsampled back and added to the full-resolution
    features before the 2-channel output (U-Net style skip).
    """

    def __init__(self, c=64, k=13, nd=2, dmrs=DEFAULT_DMRS, n_sym=NUM_SYM):
        super().__init__()
        self.config = dict(c=c, k=k, nd=nd, dmrs=tuple(dmrs), n_sym=n_sym)
        self.entry = nn.Conv2d(2, c, 1)
        self.b1 = SepRes(c, nd, k)          # full resolution
        self.b2 = SepRes(c, nd, k)          # 1/4
        self.b3 = SepRes(c, nd, k)          # 1/8
        self.b4 = SepRes(c, nd, k)          # back at full resolution
        self.out = nn.Conv2d(c, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.ls = LSGrid(dmrs, n_sym)

    def forward(self, x):
        b, _, nd, m = x.shape
        mp = math.ceil(m / 8) * 8
        xp = F.pad(x, (0, mp - m, 0, 0), mode="replicate") if mp > m else x
        h1 = self.b1(self.entry(xp))
        h2 = self.b2(F.avg_pool2d(h1, (1, 4)))
        h3 = self.b3(F.avg_pool2d(h2, (1, 2)))
        up = F.interpolate(h3, size=(nd, mp), mode="bilinear", align_corners=False)
        up = up + F.interpolate(h2, size=(nd, mp), mode="bilinear", align_corners=False)
        h4 = self.b4(h1 + up)
        den = (xp + self.out(h4))[..., :m]
        return self.ls(den)


def time_weights(dmrs, n_sym) -> torch.Tensor:
    """[n_sym, Nd] weights: linear in time between the DMRS symbols, held outside them."""
    d = list(dmrs)
    w = torch.zeros(n_sym, len(d))
    for t in range(n_sym):
        if len(d) == 1 or t <= d[0]:
            w[t, 0] = 1.0
        elif t >= d[-1]:
            w[t, -1] = 1.0
        else:
            j = bisect.bisect_right(d, t) - 1
            a = (t - d[j]) / (d[j + 1] - d[j])
            w[t, j], w[t, j + 1] = 1 - a, a
    return w


class DelayWindow(nn.Module):
    """Delay-domain windowed LS, the ``ls_hard_window`` / ``ls_soft_window`` estimators of
    ``nr_ul_sim/windowed_ce.py`` on one (rx antenna, layer) slice; no trainable weights.

    Per DMRS symbol the M block values go to the delay domain (M-point IDFT), are
    weighted per tap, and come back on all 4M subcarriers (zero-padded 4M-point DFT,
    block value on subcarrier 4m + ``centre``), then linear in time through the DMRS
    symbols, held constant outside them. Tap spacing 1 / (4M * scs); the window keeps
    taps in [-neg_us, +pos_us].

    hard   gain 1 inside the window, 0 outside.
    soft   per-tap Wiener gain max(P_k - a s2, 0) / P_k, P_k the tap power averaged over
           the DMRS symbols, s2 the median tap power outside the window (Wilson-Hilferty
           corrected), a = ``threshold``.

    Difference from nr_ul_sim: there the tap power is also averaged over the receive
    antennas and s2 is bounded below by the LS error variance / M. Here each slice is
    estimated alone (the zoo interface) from normalised input, so only the
    outside-window median sets s2.
    """

    def __init__(self, mode="soft", pos_us=3.0, neg_us=1.0, threshold=1.5, within_window=False,
                 scs_khz=30.0, centre=1, nd=2, dmrs=DEFAULT_DMRS, n_sym=NUM_SYM):
        super().__init__()
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'")
        self.config = dict(mode=mode, pos_us=pos_us, neg_us=neg_us, threshold=threshold,
                           within_window=within_window, scs_khz=scs_khz, centre=centre,
                           nd=nd, dmrs=tuple(dmrs), n_sym=n_sym)
        self.mode, self.threshold, self.within = mode, float(threshold), bool(within_window)
        self.pos_s, self.neg_s, self.scs = pos_us * 1e-6, neg_us * 1e-6, scs_khz * 1e3
        self.centre = int(centre)
        self.register_buffer("w", time_weights(dmrs, n_sym), persistent=False)

    def window(self, m, device):
        n = BLOCK_SC * m
        tap = 1.0 / (n * self.scs)
        half = (m + 1) // 2
        pos = max(0, min(int(math.floor(self.pos_s / tap + 1e-9)), half - 1))
        neg = max(0, min(int(math.floor(self.neg_s / tap + 1e-9)), m - half))
        win = torch.zeros(m, device=device)
        win[:pos + 1] = 1.0
        if neg:
            win[m - neg:] = 1.0
        return win

    def forward(self, x):
        b, _, nd, m = x.shape
        n = BLOCK_SC * m
        h = torch.complex(x[:, 0].float(), x[:, 1].float())                 # [B, Nd, M]
        taps = torch.fft.ifft(h, dim=-1)
        win = self.window(m, x.device)
        if self.mode == "hard":
            gain = win.expand(b, 1, m)
        else:
            power = taps.abs().square().mean(1, keepdim=True)              # [B, 1, M]
            out = win == 0
            if int(out.sum()) >= 8:
                corr = (1.0 - 1.0 / (9.0 * nd)) ** 3
                s2 = power[..., out].quantile(0.5, dim=-1, keepdim=True) / corr
            else:
                s2 = torch.zeros_like(power[..., :1])
            gain = ((power - self.threshold * s2) / power.clamp(min=1e-30)).clamp(0.0, 1.0)
            if self.within:
                gain = gain * win
        taps = taps * gain
        k = torch.arange(m, device=x.device)
        half = (m + 1) // 2
        idx = torch.where(k < half, k, n - (m - k))
        padded = taps.new_zeros(b, nd, n)
        padded[..., idx] = taps
        g = torch.roll(torch.fft.fft(padded, dim=-1), shifts=self.centre, dims=-1)   # [B, Nd, N]
        y = torch.einsum("tn,bnf->btf", self.w.to(g.real.dtype).to(torch.complex64), g)
        return torch.stack([y.real, y.imag], 1)


class LmmseExp(nn.Module):
    """Frequency-domain LMMSE with an exponential power-delay-profile prior, the
    ``lmmse_exp`` idea of ``nr_ul_sim`` on one (rx antenna, layer) slice; no weights.

    Each DMRS symbol is interpolated from its M block values to all 4M subcarriers with
    W = R_fp (R_pp + s2 I)^-1, R(df) = 1 / (1 + j 2 pi df scs tau), then linear in time as
    ``DelayWindow``. The noise-to-signal ratio s2 comes from the slice itself (second
    differences of neighbouring blocks), since the zoo input carries no noise level.

    derotate=True first removes the mean delay (block-to-block phase slope) and puts it
    back afterwards; the over-the-air replay used that with tau = 100 ns to undo the
    receiver's early FFT window. In simulation (ideal timing) it hurts: the centred channel
    has energy at negative delays, which the causal prior suppresses.

    Difference from nr_ul_sim: there the prior delay spread is the scenario's (CDL) or
    1 us (UMi/UMa), the noise variance is known, and time uses a Jakes covariance. Here
    the slice does not say which channel it came from, so tau is fixed.
    """

    def __init__(self, tau_ns=300.0, derotate=False, scs_khz=30.0, centre=1, nd=2, dmrs=DEFAULT_DMRS,
                 n_sym=NUM_SYM):
        super().__init__()
        self.config = dict(tau_ns=tau_ns, derotate=derotate, scs_khz=scs_khz, centre=centre, nd=nd,
                           dmrs=tuple(dmrs), n_sym=n_sym)
        self.tau, self.scs, self.centre = tau_ns * 1e-9, scs_khz * 1e3, int(centre)
        self.derotate = bool(derotate)
        self.register_buffer("w", time_weights(dmrs, n_sym), persistent=False)
        self._cache = {}

    def _prior(self, m, device):
        key = (m, str(device))
        if key not in self._cache:
            pos = torch.arange(m, dtype=torch.float64) * BLOCK_SC + self.centre
            k = torch.arange(BLOCK_SC * m, dtype=torch.float64)

            def r(df):
                return 1.0 / (1.0 + 2j * math.pi * df * self.scs * self.tau)

            lam, u = torch.linalg.eigh(r(pos[:, None] - pos[None, :]))
            rfu = r(k[:, None] - pos[None, :]) @ u
            self._cache[key] = (lam.clamp(min=0).float().to(device), u.to(torch.complex64).to(device),
                                rfu.to(torch.complex64).to(device), pos.float().to(device),
                                k.float().to(device))
        return self._cache[key]

    def forward(self, x):
        b, _, nd, m = x.shape
        h = torch.complex(x[:, 0].float(), x[:, 1].float())                 # [B, Nd, M]
        lam, u, rfu, pos, k = self._prior(m, x.device)
        if self.derotate:
            slope = torch.angle((h[..., 1:] * h[..., :-1].conj()).sum((1, 2))) / BLOCK_SC   # [B]
            h = h * torch.exp(-1j * slope[:, None, None] * pos)
        sd = h[..., 1:-1] - 0.5 * (h[..., :-2] + h[..., 2:])
        noise = sd.abs().square().mean((1, 2)) / 1.5
        p_sig = (h.abs().square().mean((1, 2)) - noise).clamp(min=1e-12)
        nvar = (noise / p_sig).clamp(min=1e-4)                              # [B]
        c = torch.einsum("mi,bdm->bdi", u.conj(), h) / (lam[None, None] + nvar[:, None, None])
        g = torch.einsum("fi,bdi->bdf", rfu, c)                             # [B, Nd, 4M]
        if self.derotate:
            g = g * torch.exp(1j * slope[:, None, None] * k)
        y = torch.einsum("tn,bnf->btf", self.w.to(torch.complex64), g)
        return torch.stack([y.real, y.imag], 1)


# ----------------------------------------------------------------------------- registry
MODELS = {
    "chea": lambda **kw: CHEA(d=16, **kw),
    "chea_xl": lambda **kw: CHEA(d=64, **kw),
    "helena": HELENA,
    "swift": SwiftChannel,
    "denoiserx": DenoiseRx,
    "hard_window": lambda **kw: DelayWindow(mode="hard", **kw),
    "soft_window": lambda **kw: DelayWindow(mode="soft", **kw),
    "lmmse_exp": LmmseExp,
}
CLASSES = {"CHEA": CHEA, "HELENA": HELENA, "SwiftChannel": SwiftChannel, "DenoiseRx": DenoiseRx,
           "DelayWindow": DelayWindow, "LmmseExp": LmmseExp}


def build(name: str, **kw) -> nn.Module:
    return MODELS[name](**kw)


def from_checkpoint(state: dict) -> nn.Module:
    net = CLASSES[state["class"]](**state["config"])
    net.load_state_dict(state["state_dict"])
    return net.eval()


@torch.no_grad()
def estimate_slices(net: nn.Module, blk: torch.Tensor, max_batch: int = 2048) -> torch.Tensor:
    """LS blocks [N, Nd, M] complex -> channel [N, 14, 4M] complex (whole band at once)."""
    outs = []
    for i in range(0, len(blk), max_batch):
        x, scale = to_input(blk[i:i + max_batch])
        y = net(x)
        outs.append(torch.complex(y[:, 0], y[:, 1]) * scale[:, None, None])
    return torch.cat(outs, 0)


if __name__ == "__main__":
    for name in MODELS:
        for kw, nd, ns in (({}, 2, NUM_SYM), (dict(nd=3, dmrs=(0, 5, 10), n_sym=13), 3, 13)):
            net = build(name, **kw)
            n = sum(p.numel() for p in net.parameters())
            for m in (12, 72, 204):
                y = net(torch.randn(3, 2, nd, m))
                assert y.shape == (3, 2, ns, 4 * m), (name, y.shape)
            print(f"{name:10s} nd={nd} n_sym={ns} params {n:7d}  ok")
