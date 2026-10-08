import math

import pytest
import torch

from nr_ul_sim.losses import MODULATIONS, bit_logits, complex_bce, complex_bce_batched, constellation


def _symbols(mod, *shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    points, _ = constellation(mod)
    return points[torch.randint(points.numel(), shape, generator=g)]


def _mixed_batch(num_sym=48, seed=0):
    mods = ["qpsk", "qam64", "qam16", "qam64", "qpsk"]
    x = torch.stack([_symbols(m, num_sym, seed=seed + i) for i, m in enumerate(mods)])
    g = torch.Generator().manual_seed(seed + 100)
    x_hat = x + 0.2 * torch.randn(x.shape, dtype=x.dtype, generator=g)
    return x_hat, x, mods


@pytest.mark.parametrize("mod", MODULATIONS)
def test_single_modulation_matches_complex_bce(mod):
    x = _symbols(mod, 6, 32)
    x_hat = x + 0.2 * torch.randn_like(x)
    batched = complex_bce_batched(x_hat, x, [mod] * 6, noise_var=0.1)
    assert torch.allclose(batched, complex_bce(x_hat, x, mod, noise_var=0.1))


def test_mixed_batch_is_bit_weighted_pool_of_per_modulation_losses():
    x_hat, x, mods = _mixed_batch()
    total, count = 0.0, 0
    for mod in MODULATIONS:
        idx = [i for i, m in enumerate(mods) if m == mod]
        k = int(math.log2(constellation(mod)[0].numel()))
        n_bits = len(idx) * x.shape[1] * k
        total += complex_bce(x_hat[idx], x[idx], mod, noise_var=0.1).item() * n_bits
        count += n_bits
    loss = complex_bce_batched(x_hat, x, mods, noise_var=0.1)
    assert loss.item() == pytest.approx(total / count, rel=1e-5)


def test_record_order_does_not_matter():
    x_hat, x, mods = _mixed_batch()
    perm = torch.tensor([3, 0, 4, 1, 2])
    a = complex_bce_batched(x_hat, x, mods, noise_var=0.1)
    b = complex_bce_batched(x_hat[perm], x[perm], [mods[i] for i in perm.tolist()], noise_var=0.1)
    assert torch.allclose(a, b)


def test_integer_codes_and_tensor_codes_match_names():
    x_hat, x, mods = _mixed_batch()
    codes = [MODULATIONS.index(m) for m in mods]
    ref = complex_bce_batched(x_hat, x, mods)
    assert torch.allclose(complex_bce_batched(x_hat, x, codes), ref)
    assert torch.allclose(complex_bce_batched(x_hat, x, torch.tensor(codes)), ref)


def test_per_record_noise_var():
    x_hat, x, mods = _mixed_batch()
    nv = torch.tensor([0.05, 0.2, 0.1, 0.3, 0.02])
    loss = complex_bce_batched(x_hat, x, mods, noise_var=nv)
    total, count = 0.0, 0
    for i, m in enumerate(mods):
        logits, target = bit_logits(x_hat[i], x[i], m, nv[i])
        total += torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="sum").item()
        count += target.numel()
    assert loss.item() == pytest.approx(total / count, rel=1e-5)


def test_perfect_estimate_small_wrong_estimate_large():
    _, x, mods = _mixed_batch()
    assert complex_bce_batched(x, x, mods, noise_var=0.01) < 1e-3
    assert complex_bce_batched(-x, x, mods, noise_var=0.01) > 5.0


def test_real_pair_input_matches_complex_input():
    x_hat, x, mods = _mixed_batch()
    pair = torch.stack([x_hat.real, x_hat.imag], dim=-1)
    assert torch.allclose(complex_bce_batched(pair, x, mods), complex_bce_batched(x_hat, x, mods))


def test_gradient_flows_to_every_record():
    _, x, mods = _mixed_batch()
    x_hat = torch.zeros_like(x, requires_grad=True)
    complex_bce_batched(x_hat, x, mods).backward()
    assert torch.isfinite(x_hat.grad).all()
    assert (x_hat.grad.abs().sum(dim=1) > 0).all()


def test_bad_inputs():
    x_hat, x, mods = _mixed_batch()
    with pytest.raises(ValueError):
        complex_bce_batched(x_hat, x, mods[:-1])
    with pytest.raises(ValueError):
        complex_bce_batched(x_hat, x, ["qpsk", "qam256", "qpsk", "qpsk", "qpsk"])
    with pytest.raises(ValueError):
        complex_bce_batched(x_hat, x, [0, 1, 2, 3, 0])
    with pytest.raises(TypeError):
        complex_bce_batched(x_hat, x, "qpsk")
    with pytest.raises(ValueError):
        complex_bce_batched(x_hat, x, mods, noise_var=torch.ones(3))
