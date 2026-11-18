"""Differentiable losses for a rank-1, single-target CE model."""
import torch
from nr_ul_sim.losses import complex_bce


def channel_mse(h_pred, h_true):
    """Complex MSE = mean(|Hhat-H|²), NOT mean over separate real/imag entries."""
    return (h_pred - h_true).abs().square().mean()


def symbol_bce(h_pred, y, x_tx, data_mask, modulation, disturbance_variance):
    """CE -> unbiased SIMO combining -> exact QAM bit logits -> BCE.

    Inputs [RX,T,F]; x_tx/data_mask [T,F]. Assumes spatially white I+N.
    disturbance_variance is the *complex* per-RX variance N0+I0. Coded QAM
    bits are inferred from x_tx; transport-block bits are not valid labels here.
    This is a surrogate: channel uncertainty and coloured interference are not
    in the Gaussian likelihood. All operations on h_pred remain differentiable.
    """
    power = h_pred.abs().square().sum(0).clamp_min(1e-10)
    x_hat = (h_pred.conj() * y).sum(0) / power
    nv = torch.as_tensor(disturbance_variance, device=h_pred.device, dtype=h_pred.real.dtype) / power
    return complex_bce(x_hat[data_mask], x_tx[data_mask], modulation, nv[data_mask])
