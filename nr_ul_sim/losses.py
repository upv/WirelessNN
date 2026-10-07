"""Loss functions for training on complex-valued signals.

    from nr_ul_sim.losses import complex_bce
    loss = complex_bce(x_hat, x)

Self-contained: depends on torch only, so the file can be copied into any project.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def complex_bce(x_hat: torch.Tensor, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Binary cross-entropy between an estimated and a transmitted complex signal.

    Each complex sample carries two bits, the signs of its real and imaginary parts
    (BPSK / QPSK style): bit = 1 where the part is positive, else 0. The real and
    imaginary parts of ``x_hat`` are used as logits for those bits, so the loss is

        BCE(sigmoid(scale * Re x_hat), Re x > 0) + BCE(sigmoid(scale * Im x_hat), Im x > 0)

    averaged over all samples and both bits. ``scale`` sharpens the logits
    (roughly 2 * sqrt(2) / noise_std for a unit-energy QPSK signal); 1.0 is fine
    for a plain training signal.

    ``x_hat`` and ``x`` may be complex tensors of any shape (broadcastable), or
    real tensors whose last dimension holds (real, imag).
    """
    x_hat = _as_complex(x_hat)
    x = _as_complex(x)
    logits = scale * torch.stack([x_hat.real, x_hat.imag], dim=-1)
    bits = torch.stack([x.real > 0, x.imag > 0], dim=-1).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, bits)


def _as_complex(t) -> torch.Tensor:
    t = torch.as_tensor(t)
    if t.is_complex():
        return t
    if t.shape[-1] != 2:
        raise ValueError("expected a complex tensor or a real tensor with last dim (real, imag)")
    return torch.complex(t[..., 0], t[..., 1])
