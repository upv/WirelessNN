"""Loss functions for training on complex-valued signals.

    from nr_ul_sim.losses import complex_bce, complex_bce_batched
    loss = complex_bce(x_hat, x, modulation="qam16", noise_var=0.1)
    loss = complex_bce_batched(x_hat, x, ["qpsk", "qam64", "qam16"], noise_var=0.1)

Self-contained: depends on torch only, so the file can be copied into any project.

Square Gray-mapped QAM is separable: even-indexed bits select the real PAM level,
odd-indexed bits the imaginary one, and |x_hat - s|^2 splits into a real and an
imaginary term. The imaginary factor cancels in the LLR of a real-axis bit, so the
exact LLR over M points reduces to an LLR over sqrt(M) PAM levels per axis. Both
losses use that form: 8 levels instead of 64 points for 64QAM.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import math

import torch
import torch.nn.functional as F

MODULATIONS = ("qpsk", "qam16", "qam64")
_BITS_PER_AXIS = {"qpsk": 1, "qam16": 2, "qam64": 3}
_ENERGY = {"qpsk": 2.0, "qam16": 10.0, "qam64": 42.0}
_MAX_LEVELS = 8      # 64QAM: 2**3 PAM levels per axis
_MAX_AXIS_BITS = 3


def complex_bce(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    modulation: str = "qpsk",
    noise_var: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Bit-wise binary cross-entropy between an estimated and a transmitted signal.

    ``x`` holds transmitted constellation symbols (unit average energy, 3GPP 38.211
    Gray mapping); its bits are recovered by nearest-point demapping. ``x_hat`` is
    the estimate; its bit log-likelihood ratios

        LLR_k = log sum_{s: b_k(s)=1} exp(-|x_hat - s|^2 / noise_var)
              - log sum_{s: b_k(s)=0} exp(-|x_hat - s|^2 / noise_var)

    are used as logits, so the loss is -log P(true bit | x_hat), averaged over all
    symbols and bits. ``noise_var`` is the per-symbol complex noise variance the
    demapper assumes (smaller = sharper LLRs); a scalar, or a tensor broadcastable
    to the symbol shape.

    ``x_hat`` and ``x`` may be complex tensors of any (broadcastable) shape, or
    real tensors whose last dimension holds (real, imag).
    """
    logits, target = bit_logits(x_hat, x, modulation, noise_var)
    return F.binary_cross_entropy_with_logits(logits, target)


def complex_bce_batched(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    modulations: Sequence[str] | Sequence[int] | torch.Tensor,
    noise_var: float | torch.Tensor = 1.0,
    compiled: bool = False,
) -> torch.Tensor:
    """``complex_bce`` for a batch whose records use different modulations.

    ``x_hat`` and ``x`` are ``[B, ...]`` (first dimension = record). ``modulations``
    has one entry per record: a modulation name from ``MODULATIONS`` or its index
    in that tuple (0 = qpsk, 1 = qam16, 2 = qam64). ``noise_var`` is a scalar, a
    ``[B]`` tensor with one noise variance per record, or a tensor broadcastable to
    the symbol shape ``[B, ...]`` (per-RE noise estimate).

    The result is the mean over every bit of the batch, so a QAM64 record weighs
    three times a QPSK record of the same length, exactly as if each modulation
    group had been passed to ``complex_bce`` and the sums were pooled. With a
    single modulation it equals ``complex_bce``.

    One vectorised pass over the batch: every record carries a table of up to 8
    PAM levels and 3 bits per axis, padded entries are masked out. No Python loop
    over records or modulation groups, no gather of record subsets. The core works
    on real tensors without host syncs, so ``compiled=True`` runs it through
    ``torch.compile`` (first call compiles, ~10 s; 5-15x faster afterwards on a GPU).
    """
    x_hat = _as_complex(x_hat)
    x = _as_complex(x)
    if x_hat.dim() == 0 or x.dim() == 0:
        raise ValueError("expected batched tensors with a leading record dimension")
    num_records = max(x_hat.shape[0], x.shape[0])
    codes = _modulation_codes(modulations, num_records, x_hat.device)
    nv = _noise_var(noise_var, x_hat, num_records)
    tables = _axis_tables(x_hat.device, x_hat.real.dtype)
    sym_dims = max(x_hat.dim(), x.dim()) - 1
    symbols_per_record = (x_hat.numel() if x_hat.shape[0] == num_records else x.numel()) // num_records
    core = _compiled_core() if compiled else _batched_core
    return core(x_hat.real, x_hat.imag, x.real, x.imag, nv, codes, sym_dims, symbols_per_record, *tables)


def _batched_core(est_re, est_im, tx_re, tx_im, nv, codes, sym_dims, symbols_per_record,
                  levels, level_valid, level_bits, bit_valid):
    """Real-valued, sync-free body of ``complex_bce_batched`` (``torch.compile``-able)."""
    expand = (codes.shape[0],) + (1,) * sym_dims
    lv = levels[codes].reshape(*expand, _MAX_LEVELS)
    lv_valid = level_valid[codes].reshape(*expand, _MAX_LEVELS)
    lb = level_bits[codes].reshape(*expand, _MAX_AXIS_BITS, _MAX_LEVELS)
    b_valid = bit_valid[codes].reshape(*expand, _MAX_AXIS_BITS)

    total = est_re.new_zeros(())
    for est, tx in ((est_re, tx_re), (est_im, tx_im)):
        logits, target = _axis_logits(est, tx, nv, lv, lv_valid, lb)        # [B, ..., 3]
        loss = F.binary_cross_entropy_with_logits(
            torch.where(b_valid, logits, 0.0), target, reduction="none")
        total = total + torch.where(b_valid, loss, 0.0).sum()
    count = 2 * bit_valid[codes].sum() * symbols_per_record                  # bits in the batch
    return total / count


_COMPILED_CORE = None


def _compiled_core():
    global _COMPILED_CORE
    if _COMPILED_CORE is None:
        _COMPILED_CORE = torch.compile(_batched_core, dynamic=True)
    return _COMPILED_CORE


def bit_logits(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    modulation: str,
    noise_var: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact bit LLRs of ``x_hat`` and the transmitted bits of ``x``, both ``[..., k]``.

    Bit ``2j`` is the j-th real-axis bit and bit ``2j+1`` the j-th imaginary-axis
    bit, the 38.211 order (same labels as ``constellation``).
    """
    if modulation not in MODULATIONS:
        raise ValueError(f"modulation must be one of {MODULATIONS}")
    x_hat = _as_complex(x_hat)
    x = _as_complex(x)
    real_dtype = x_hat.real.dtype
    nv = torch.as_tensor(noise_var, dtype=real_dtype, device=x_hat.device)
    levels, level_valid, level_bits, _ = _axis_tables(x_hat.device, real_dtype)
    code = MODULATIONS.index(modulation)
    k = _BITS_PER_AXIS[modulation]
    lv, lv_valid, lb = levels[code, :1 << k], level_valid[code, :1 << k], level_bits[code, :k, :1 << k]

    out = [_axis_logits(est, tx, nv, lv, lv_valid, lb)
           for est, tx in ((x_hat.real, x.real), (x_hat.imag, x.imag))]
    logits = torch.stack([out[0][0], out[1][0]], dim=-1).flatten(-2)      # [..., k] interleaved
    target = torch.stack([out[0][1], out[1][1]], dim=-1).flatten(-2)
    return logits, target


def _axis_logits(est, tx, noise_var, levels, level_valid, level_bits):
    """LLRs and target bits of one PAM axis.

    ``est``/``tx``: real ``[...]``; ``noise_var`` broadcastable to them;
    ``levels``/``level_valid``: ``[..., L]`` (broadcastable); ``level_bits``: ``[..., J, L]``.
    Returns ``(logits, target)`` of shape ``[..., J]``.
    """
    neg_inf = -math.inf
    metric = -(est.unsqueeze(-1) - levels).square() / noise_var.unsqueeze(-1)   # [..., L]
    metric = torch.where(level_valid, metric, neg_inf).unsqueeze(-2)             # [..., 1, L]
    llr_1 = torch.logsumexp(torch.where(level_bits, metric, neg_inf), dim=-1)    # [..., J]
    llr_0 = torch.logsumexp(torch.where(level_bits, neg_inf, metric), dim=-1)
    logits = llr_1 - llr_0

    dist = torch.where(level_valid, (tx.unsqueeze(-1) - levels).abs(), math.inf)
    idx = dist.argmin(dim=-1, keepdim=True).unsqueeze(-2)                        # [..., 1, 1]
    target = torch.gather(level_bits.expand(*logits.shape, level_bits.shape[-1]), -1,
                          idx.expand(*logits.shape, 1)).squeeze(-1)             # [..., J]
    return logits, target.to(logits.dtype)


@lru_cache(maxsize=None)
def _axis_tables(device, dtype):
    """Per-modulation PAM tables padded to 8 levels / 3 bits per axis.

    ``levels [3, 8]`` (normalised PAM levels, padded with 0), ``level_valid [3, 8]``
    bool, ``level_bits [3, 3, 8]`` bool (bit j of level l), ``bit_valid [3, 3]`` bool.
    """
    levels = torch.zeros(len(MODULATIONS), _MAX_LEVELS, dtype=dtype)
    level_valid = torch.zeros(len(MODULATIONS), _MAX_LEVELS, dtype=torch.bool)
    level_bits = torch.zeros(len(MODULATIONS), _MAX_AXIS_BITS, _MAX_LEVELS, dtype=torch.bool)
    bit_valid = torch.zeros(len(MODULATIONS), _MAX_AXIS_BITS, dtype=torch.bool)
    for code, mod in enumerate(MODULATIONS):
        k = _BITS_PER_AXIS[mod]
        n = 1 << k
        bits = ((torch.arange(n).unsqueeze(1) >> torch.arange(k - 1, -1, -1)) & 1).float()   # [n, k]
        levels[code, :n] = _pam(bits) / math.sqrt(_ENERGY[mod])
        level_valid[code, :n] = True
        level_bits[code, :k, :n] = bits.T > 0.5
        bit_valid[code, :k] = True
    return (levels.to(device), level_valid.to(device), level_bits.to(device), bit_valid.to(device))


def constellation(modulation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-energy Gray-mapped constellation per 3GPP 38.211 5.1.

    Returns ``(points, bits)``: complex points ``[M]`` and their labels ``[M, k]``.
    Even-indexed bits drive the real axis, odd-indexed bits the imaginary axis.
    """
    if modulation not in MODULATIONS:
        raise ValueError(f"modulation must be one of {MODULATIONS}")
    k = 2 * _BITS_PER_AXIS[modulation]
    m = 1 << k
    bits = ((torch.arange(m).unsqueeze(1) >> torch.arange(k - 1, -1, -1)) & 1).float()
    re = _pam(bits[:, 0::2])
    im = _pam(bits[:, 1::2])
    points = torch.complex(re, im) / math.sqrt(_ENERGY[modulation])
    return points, bits


def _pam(b: torch.Tensor) -> torch.Tensor:
    """38.211 PAM level from Gray bits [b_0, b_1, ...] (MSB first): b=0 -> positive."""
    s = 1.0 - 2.0 * b
    level = s[:, -1]
    for j in range(b.shape[1] - 2, -1, -1):
        level = s[:, j] * (2.0 ** (b.shape[1] - 1 - j) - level)
    return level


def _as_complex(t) -> torch.Tensor:
    t = torch.as_tensor(t)
    if t.is_complex():
        return t
    if t.shape[-1] != 2:
        raise ValueError("expected a complex tensor or a real tensor with last dim (real, imag)")
    return torch.complex(t[..., 0], t[..., 1])


def _noise_var(noise_var, x_hat, num_records):
    nv = torch.as_tensor(noise_var, dtype=x_hat.real.dtype, device=x_hat.device)
    if nv.dim() == 1 and nv.numel() == num_records and x_hat.dim() > 1:
        nv = nv.reshape(num_records, *([1] * (x_hat.dim() - 1)))         # per record
    elif nv.dim() != 0:
        try:
            torch.broadcast_shapes(nv.shape, x_hat.shape)
        except RuntimeError:
            raise ValueError(
                f"noise_var must be a scalar, {num_records} per-record values, "
                f"or broadcastable to the symbol shape {tuple(x_hat.shape)}") from None
    return nv


def _modulation_codes(modulations, num_records: int, device) -> torch.Tensor:
    if isinstance(modulations, str):
        raise TypeError("modulations must hold one entry per record, not a single name")
    if isinstance(modulations, torch.Tensor):
        codes = modulations.to(torch.long).flatten()
        if codes.numel() != num_records:
            raise ValueError(f"got {codes.numel()} modulations for {num_records} records")
        if codes.numel() and ((codes < 0) | (codes >= len(MODULATIONS))).any():
            raise ValueError(f"modulation code must be in 0..{len(MODULATIONS) - 1}")
        return codes.to(device)
    out = []
    for m in modulations:
        if isinstance(m, str):
            if m not in MODULATIONS:
                raise ValueError(f"modulation must be one of {MODULATIONS}, got {m!r}")
            out.append(MODULATIONS.index(m))
        else:
            code = int(m)
            if not 0 <= code < len(MODULATIONS):
                raise ValueError(f"modulation code must be in 0..{len(MODULATIONS) - 1}, got {code}")
            out.append(code)
    if len(out) != num_records:
        raise ValueError(f"got {len(out)} modulations for {num_records} records")
    return torch.tensor(out, dtype=torch.long, device=device)
