"""Channelformer (Luan & Thompson, IEEE TWC 2023, arXiv 2302.04368), offline variant,
adapted to NR PUSCH DMRS type 1 on a 68-PRB grid.

Paper setting: 72 subcarriers x 14 symbols, 2 DMRS symbols, 72 LS pilot values
(2 x 36) -> full 14 x 72 slot. Here one sample is a 12-PRB window (144 subcarriers)
of one (rx antenna, layer) slice: the OCC-despread DMRS gives one LS value per
4-subcarrier block, i.e. 36 blocks x 2 DMRS symbols = 72 values, exactly the
paper's input length. The network outputs the 14 x 144 window; a 68-PRB slot is
assembled from 17 overlapping windows, keeping the central 4 PRB of each
(windows are clamped at the band edges).

Layer by layer (paper Sec. IV; dims not given in the paper are marked *):
  encoder  FC 72->216 along the pilot axis (per re/im), split into Q, K, V;
           2 heads = 2 DMRS symbols (36 pilots each); softmax(QK^T/sqrt(36)) V;
           concat, FC 72->72, add input, LayerNorm;
           pre-network conv 2x2 (1->5) GELU(tanh) conv 2x2 (5->1), add, LayerNorm.
  decoder  conv 5x5 (1->12); 3 residual blocks conv5x5-ReLU-conv5x5 + skip + LN*;
           FC 72 -> 14*144 along the pilot axis per channel; conv 5x5 (12->1).
  loss     Huber(delta=1); Adam lr 2e-3, halved every 50 epochs, 100 epochs,
           batch 128, L2 1e-7.
"""

from __future__ import annotations

import math

import torch
from torch import nn

WIN_BLOCKS = 36       # 12 PRB of 4-subcarrier DMRS blocks
CORE_BLOCKS = 12      # 4 PRB kept from every window
BLOCK_SC = 4
NUM_DMRS = 2
NUM_SYM = 14


class Channelformer(nn.Module):
    def __init__(self, n_tok: int = NUM_DMRS * WIN_BLOCKS, n_heads: int = NUM_DMRS,
                 n_enc: int = 5, n_dec: int = 12, n_res: int = 3, k_dec: int = 5,
                 out_sym: int = NUM_SYM, out_sc: int = WIN_BLOCKS * BLOCK_SC):
        super().__init__()
        self.config = dict(n_tok=n_tok, n_heads=n_heads, n_enc=n_enc, n_dec=n_dec,
                           n_res=n_res, k_dec=k_dec, out_sym=out_sym, out_sc=out_sc)
        assert n_tok % n_heads == 0
        self.n_tok, self.n_heads = n_tok, n_heads
        self.d_k = n_tok // n_heads
        self.out_sym, self.out_sc = out_sym, out_sc
        # encoder
        self.fc_qkv = nn.Linear(n_tok, 3 * n_tok)
        self.fc_out = nn.Linear(n_tok, n_tok)
        self.ln1 = nn.LayerNorm([n_tok, 2], eps=1e-5)
        self.pre1 = nn.Conv2d(1, n_enc, 2, padding="same")
        self.pre2 = nn.Conv2d(n_enc, 1, 2, padding="same")
        self.act = nn.GELU(approximate="tanh")
        self.ln2 = nn.LayerNorm([n_tok, 2], eps=1e-5)
        # decoder
        self.entry = nn.Conv2d(1, n_dec, k_dec, padding="same")
        self.res = nn.ModuleList()
        self.res_ln = nn.ModuleList()
        for _ in range(n_res):
            self.res.append(nn.Sequential(
                nn.Conv2d(n_dec, n_dec, k_dec, padding="same"), nn.ReLU(),
                nn.Conv2d(n_dec, n_dec, k_dec, padding="same")))
            self.res_ln.append(nn.LayerNorm([n_tok, 2], eps=1e-5))
        self.fc_up = nn.Linear(n_tok, out_sym * out_sc)
        self.out_conv = nn.Conv2d(n_dec, 1, k_dec, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, n_tok, 2] (re/im, DMRS-symbol major) -> [B, out_sym, out_sc, 2]."""
        b = x.shape[0]
        qkv = self.fc_qkv(x.transpose(1, 2))                      # [B, 2, 3T]
        q, k, v = qkv.chunk(3, dim=-1)                            # [B, 2, T]

        def heads(t):                                             # -> [B, H, T/H, 2]
            return t.reshape(b, 2, self.n_heads, self.d_k).permute(0, 2, 3, 1)

        q, k, v = heads(q), heads(k), heads(v)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.d_k), dim=-1)
        z = (att @ v).reshape(b, self.n_tok, 2)                   # concat heads on the pilot axis
        z = self.fc_out(z.transpose(1, 2)).transpose(1, 2)
        h = self.ln1(x + z)
        p = self.pre2(self.act(self.pre1(h.unsqueeze(1)))).squeeze(1)
        h = self.ln2(h + p)
        # decoder
        d = self.entry(h.unsqueeze(1))                            # [B, C, T, 2]
        for blk, ln in zip(self.res, self.res_ln):
            d = ln(d + blk(d))
        u = self.fc_up(d.transpose(-1, -2))                       # [B, C, 2, S*F]
        u = self.out_conv(u.transpose(-1, -2))                    # [B, 1, S*F, 2]
        return u.reshape(b, self.out_sym, self.out_sc, 2)


def window_starts(num_blocks: int) -> list[int]:
    """Window start (in blocks) for every 4-PRB core chunk, clamped to the band."""
    n_core = num_blocks // CORE_BLOCKS
    if n_core * CORE_BLOCKS != num_blocks or num_blocks < WIN_BLOCKS:
        raise ValueError("Channelformer needs a multiple of 4 PRB and at least 12 PRB")
    margin = (WIN_BLOCKS - CORE_BLOCKS) // 2
    return [min(max(c * CORE_BLOCKS - margin, 0), num_blocks - WIN_BLOCKS) for c in range(n_core)]


def to_tokens(win: torch.Tensor):
    """Complex windows [N, Nd, 36] -> normalised real tokens [N, 72, 2] and the scale [N]."""
    flat = win.reshape(win.shape[0], -1)
    scale = flat.abs().pow(2).mean(dim=1).sqrt().clamp(min=1e-12)
    flat = flat / scale[:, None]
    return torch.stack([flat.real, flat.imag], dim=-1).float(), scale


@torch.no_grad()
def estimate_slices(net: Channelformer, blk: torch.Tensor, max_batch: int = 8192) -> torch.Tensor:
    """LS blocks of N slices [N, Nd, M] (complex) -> channel [N, 14, 4M] (complex)."""
    n, nd, m = blk.shape
    starts = window_starts(m)
    wins = torch.stack([blk[:, :, s: s + WIN_BLOCKS] for s in starts], dim=1)   # [N, C, Nd, 36]
    tok, scale = to_tokens(wins.reshape(-1, nd, WIN_BLOCKS))
    outs = []
    for i in range(0, tok.shape[0], max_batch):
        outs.append(net(tok[i: i + max_batch]))
    y = torch.cat(outs, 0)
    y = torch.complex(y[..., 0], y[..., 1]) * scale[:, None, None].to(y.dtype)
    y = y.reshape(n, len(starts), y.shape[-2], y.shape[-1])
    core_sc = CORE_BLOCKS * BLOCK_SC
    out = torch.empty(n, y.shape[-2], m * BLOCK_SC, dtype=y.dtype, device=y.device)
    for c, s in enumerate(starts):
        off = (c * CORE_BLOCKS - s) * BLOCK_SC
        out[..., c * core_sc:(c + 1) * core_sc] = y[:, c, :, off: off + core_sc]
    return out


if __name__ == "__main__":
    net = Channelformer()
    print("parameters", sum(p.numel() for p in net.parameters()))
    print(net(torch.randn(3, 72, 2)).shape)
    print(estimate_slices(net, torch.randn(5, 2, 204, dtype=torch.complex64)).shape)
