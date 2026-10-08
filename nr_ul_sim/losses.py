"""Loss functions for training on complex-valued signals.

    from nr_ul_sim.losses import complex_bce, complex_bce_batched
    loss = complex_bce(x_hat, x, modulation="qam16", noise_var=0.1)
    loss = complex_bce_batched(x_hat, x, ["qpsk", "qam64", "qam16"], noise_var=0.1)

Self-contained: depends on torch only, so the file can be copied into any project.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

MODULATIONS = ("qpsk", "qam16", "qam64")


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
) -> torch.Tensor:
    """``complex_bce`` for a batch whose records use different modulations.

    ``x_hat`` and ``x`` are ``[B, ...]`` (first dimension = record). ``modulations``
    has one entry per record: a modulation name from ``MODULATIONS`` or its index
    in that tuple (0 = qpsk, 1 = qam16, 2 = qam64). ``noise_var`` is a scalar or a
    ``[B]`` tensor with one noise variance per record.

    The result is the mean over every bit of the batch, so a QAM64 record weighs
    three times a QPSK record of the same length, exactly as if each modulation
    group had been passed to ``complex_bce`` and the sums were pooled. With a
    single modulation it equals ``complex_bce``.
    """
    x_hat = _as_complex(x_hat)
    x = _as_complex(x)
    if x_hat.dim() == 0 or x.dim() == 0:
        raise ValueError("expected batched tensors with a leading record dimension")
    num_records = max(x_hat.shape[0], x.shape[0])
    names = _modulation_names(modulations, num_records)
    noise = torch.as_tensor(noise_var, dtype=x_hat.real.dtype, device=x_hat.device)
    if noise.dim() > 1 or (noise.dim() == 1 and noise.numel() != num_records):
        raise ValueError(f"noise_var must be a scalar or a tensor of {num_records} per-record values")

    total = x_hat.real.new_zeros(())
    count = 0
    for mod in MODULATIONS:
        sel = [i for i, name in enumerate(names) if name == mod]
        if not sel:
            continue
        idx = torch.tensor(sel, device=x_hat.device)
        xh = x_hat[idx] if x_hat.shape[0] == num_records else x_hat
        xt = x[idx] if x.shape[0] == num_records else x
        nv = noise if noise.dim() == 0 else noise[idx].reshape(-1, *([1] * (max(xh.dim(), xt.dim()) - 1)))
        logits, target = bit_logits(xh, xt, mod, nv)
        total = total + F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        count += target.numel()
    return total / count


def bit_logits(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    modulation: str,
    noise_var: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact bit LLRs of ``x_hat`` and the transmitted bits of ``x``, both ``[..., k]``."""
    points, bits = constellation(modulation)
    x_hat = _as_complex(x_hat)
    x = _as_complex(x)
    points = points.to(device=x_hat.device, dtype=x_hat.dtype)
    bits = bits.to(device=x_hat.device, dtype=x_hat.real.dtype)
    noise_var = torch.as_tensor(noise_var, dtype=x_hat.real.dtype, device=x_hat.device)

    # target bits: label of the constellation point nearest to the transmitted symbol
    idx = (x.unsqueeze(-1) - points).abs().argmin(dim=-1)
    target = bits[idx]                                                   # [..., k]

    # exact LLRs of the estimate
    metric = -(x_hat.unsqueeze(-1) - points).abs().square() / noise_var.unsqueeze(-1)  # [..., M]
    metric = metric.unsqueeze(-1)                                        # [..., M, 1]
    neg_inf = torch.tensor(-math.inf, dtype=metric.dtype, device=metric.device)
    llr_1 = torch.logsumexp(torch.where(bits > 0.5, metric, neg_inf), dim=-2)
    llr_0 = torch.logsumexp(torch.where(bits < 0.5, metric, neg_inf), dim=-2)
    logits = llr_1 - llr_0                                               # [..., k]
    target = target.expand_as(logits) if target.shape != logits.shape else target
    return logits, target


def constellation(modulation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-energy Gray-mapped constellation per 3GPP 38.211 5.1.

    Returns ``(points, bits)``: complex points ``[M]`` and their labels ``[M, k]``.
    Even-indexed bits drive the real axis, odd-indexed bits the imaginary axis.
    """
    if modulation not in MODULATIONS:
        raise ValueError(f"modulation must be one of {MODULATIONS}")
    k = {"qpsk": 2, "qam16": 4, "qam64": 6}[modulation]
    m = 1 << k
    bits = ((torch.arange(m).unsqueeze(1) >> torch.arange(k - 1, -1, -1)) & 1).float()
    re = _pam(bits[:, 0::2])
    im = _pam(bits[:, 1::2])
    energy = {"qpsk": 2.0, "qam16": 10.0, "qam64": 42.0}[modulation]
    points = torch.complex(re, im) / math.sqrt(energy)
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


def _modulation_names(modulations, num_records: int) -> list[str]:
    if isinstance(modulations, torch.Tensor):
        modulations = modulations.tolist()
    if isinstance(modulations, str):
        raise TypeError("modulations must hold one entry per record, not a single name")
    names = []
    for m in modulations:
        if isinstance(m, str):
            if m not in MODULATIONS:
                raise ValueError(f"modulation must be one of {MODULATIONS}, got {m!r}")
            names.append(m)
        else:
            code = int(m)
            if not 0 <= code < len(MODULATIONS):
                raise ValueError(f"modulation code must be in 0..{len(MODULATIONS) - 1}, got {code}")
            names.append(MODULATIONS[code])
    if len(names) != num_records:
        raise ValueError(f"got {len(names)} modulations for {num_records} records")
    return names
