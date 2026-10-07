import math

import pytest
import torch

from nr_ul_sim.losses import MODULATIONS, complex_bce, constellation


def _bits_to_point(mod, bit_string):
    points, bits = constellation(mod)
    b = torch.tensor([float(c) for c in bit_string])
    idx = (bits == b).all(dim=1).nonzero().item()
    return points[idx]


@pytest.mark.parametrize(
    "mod,bit_string,expected",
    [
        ("qpsk", "00", (1 + 1j) / math.sqrt(2)),
        ("qpsk", "11", (-1 - 1j) / math.sqrt(2)),
        ("qam16", "0000", (1 + 1j) / math.sqrt(10)),
        ("qam16", "1010", (-3 + 1j) / math.sqrt(10)),
        ("qam64", "000000", (3 + 3j) / math.sqrt(42)),
        ("qam64", "111111", (-7 - 7j) / math.sqrt(42)),
        ("qam64", "100101", (-3 + 7j) / math.sqrt(42)),
    ],
)
def test_constellation_matches_38211(mod, bit_string, expected):
    assert torch.allclose(_bits_to_point(mod, bit_string), torch.tensor(expected, dtype=torch.complex64))


@pytest.mark.parametrize("mod", MODULATIONS)
def test_unit_energy_and_unique_points(mod):
    points, bits = constellation(mod)
    assert points.abs().square().mean().item() == pytest.approx(1.0, abs=1e-6)
    assert len(set(points.tolist())) == points.numel()
    assert bits.shape == (points.numel(), int(math.log2(points.numel())))


@pytest.mark.parametrize("mod", MODULATIONS)
def test_perfect_estimate_small_wrong_estimate_large(mod):
    points, _ = constellation(mod)
    x = points[torch.randint(points.numel(), (64,))]
    good = complex_bce(x, x, mod, noise_var=0.01)
    bad = complex_bce(-x, x, mod, noise_var=0.01)
    assert good < 1e-3
    assert bad > 5.0


@pytest.mark.parametrize("mod", MODULATIONS)
def test_loss_decreases_towards_true_symbol(mod):
    points, _ = constellation(mod)
    x = points[torch.randint(points.numel(), (256,))]
    noise = 0.3 * torch.randn_like(x)
    far = complex_bce(x + noise, x, mod, noise_var=0.1)
    near = complex_bce(x + 0.3 * noise, x, mod, noise_var=0.1)
    assert near < far


def test_qpsk_zero_estimate_is_ln2():
    x = torch.randn(32, dtype=torch.complex64)
    loss = complex_bce(torch.zeros_like(x), x, "qpsk")
    assert loss.item() == pytest.approx(math.log(2.0), abs=1e-6)


def test_real_pair_input_matches_complex_input():
    x = torch.randn(8, 4, dtype=torch.complex64)
    x_hat = torch.randn(8, 4, dtype=torch.complex64)
    pair = torch.stack([x_hat.real, x_hat.imag], dim=-1)
    assert torch.allclose(complex_bce(pair, x, "qam16"), complex_bce(x_hat, x, "qam16"))


@pytest.mark.parametrize("mod", MODULATIONS)
def test_gradient_flows(mod):
    x = torch.randn(16, dtype=torch.complex64)
    x_hat = torch.zeros(16, dtype=torch.complex64, requires_grad=True)
    complex_bce(x_hat, x, mod).backward()
    assert x_hat.grad is not None and torch.isfinite(x_hat.grad).all()


def test_unknown_modulation():
    with pytest.raises(ValueError):
        complex_bce(torch.zeros(2, dtype=torch.complex64), torch.zeros(2, dtype=torch.complex64), "qam256")
