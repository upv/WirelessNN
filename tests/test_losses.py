import math

import pytest
import torch

from nr_ul_sim.losses import complex_bce


def test_perfect_estimate_has_small_loss_and_wrong_estimate_large():
    x = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / math.sqrt(2)
    good = complex_bce(10 * x, x)
    bad = complex_bce(-10 * x, x)
    assert good < 1e-3
    assert bad > 5.0


def test_zero_estimate_is_ln2():
    x = torch.randn(32, dtype=torch.complex64)
    loss = complex_bce(torch.zeros_like(x), x)
    assert loss.item() == pytest.approx(math.log(2.0), abs=1e-6)


def test_real_pair_input_matches_complex_input():
    x = torch.randn(8, 4, dtype=torch.complex64)
    x_hat = torch.randn(8, 4, dtype=torch.complex64)
    pair = torch.stack([x_hat.real, x_hat.imag], dim=-1)
    assert torch.allclose(complex_bce(pair, x), complex_bce(x_hat, x))


def test_gradient_flows():
    x = torch.randn(16, dtype=torch.complex64)
    x_hat = torch.zeros(16, dtype=torch.complex64, requires_grad=True)
    complex_bce(x_hat, x).backward()
    assert x_hat.grad is not None and torch.isfinite(x_hat.grad).all()
