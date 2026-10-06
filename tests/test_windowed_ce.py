"""Delay-domain windowed channel estimators (68 PRB)."""

import numpy as np
import pytest
import torch

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.windowed_ce import WindowedLSChannelEstimator

ALL = ("perfect", "ls_lin", "ls_hard_window", "ls_soft_window", "lmmse_ce")


def _cfg(snr, **kw):
    base = dict(
        num_prb=68, fft_size=1024, batch_size=2, max_mc_iter=1, num_target_bit_errors=1,
        receivers=("irc",), channel_estimators=ALL, snr_db=[snr], iot_db=[0.0], seed=7,
    )
    base.update(kw)
    return SimConfig(**base)


def _nmse(h_hat, h):
    return float((h_hat - h).abs().pow(2).mean() / h.abs().pow(2).mean())


def _slot(cfg, snr, iot=0.0):
    sim = NRUplinkSimulator(cfg)
    slot = sim.generate_slot(cfg.batch_size, snr, iot)
    sim.estimate_csi(slot)
    return sim, slot


def test_window_geometry_68_prb():
    cfg = _cfg(10.0)
    sim = NRUplinkSimulator(cfg)
    est = sim.rx.estimators["ls_hard_window"]
    assert isinstance(est, WindowedLSChannelEstimator)
    assert est.num_sc == 816 and est.num_blocks == 204 and est.block_spacing == 4
    assert est.dmrs_syms == [2, 11]
    ts = 1.0 / (816 * 30e3)
    assert est.tap_spacing_s == pytest.approx(ts)
    assert est.window_pos_taps == int(3.0e-6 / ts)  # 73 taps
    assert est.window_neg_taps == int(1.0e-6 / ts)  # 24 taps
    assert est.num_kept_taps == est.window_pos_taps + 1 + est.window_neg_taps


def test_hard_window_is_exact_for_noiseless_in_window_channel():
    """A synthetic channel whose taps lie inside the window is reproduced on the DMRS symbols.

    One UE only: with two ports in one CDM group the OCC despreading itself
    leaks (h1(k) - h1(k+2)) / 2 between the ports on a selective channel.
    """
    cfg = _cfg(100.0, num_ue=1, channel_estimators=("perfect", "ls_hard_window"))
    sim = NRUplinkSimulator(cfg)
    est = sim.rx.estimators["ls_hard_window"]
    x, _ = sim.transmit(1)
    n_sc = cfg.num_used_subcarriers
    left, _ = cfg.guard_carriers
    # taps at 0, 5, 12 samples (< window_pos) and -2 (> -window_neg)
    k = torch.arange(n_sc, dtype=torch.float32)
    delays = [0, 5, 12, -2]
    gains = [1.0, 0.5j, -0.3, 0.2]
    h_line = sum(g * torch.exp(-2j * np.pi * d * k / n_sc) for g, d in zip(gains, delays))
    h = torch.zeros(1, 1, cfg.num_rx_ant, cfg.num_ue, 1, 14, cfg.fft_size, dtype=torch.complex64)
    for m in range(cfg.num_rx_ant):
        for t in range(cfg.num_ue):
            h[0, 0, m, t, 0, :, left : left + n_sc] = h_line * torch.exp(1j * torch.tensor(0.7 * m + 1.3 * t))
    h = h.to(x.device)
    y = sim.apply_channel(x, h)
    no = torch.tensor(1e-10, device=y.device)
    h_hat, err_var = est(y, no)
    h_true = sim.true_channel(h)
    for sym in est.dmrs_syms:
        # residual: OCC despreading averages two pilots 2 subcarriers apart,
        # which scales tap d by cos(2 pi d / N) (0.996 for d = 12)
        assert _nmse(h_hat[..., sym, :], h_true[..., sym, :]) < 2e-4
    assert torch.all(err_var >= 0)


@pytest.mark.parametrize("snr", [0.0, 10.0])
def test_windowed_estimators_beat_linear_interpolation(snr):
    cfg = _cfg(snr, channel="cdl-c")
    sim, slot = _slot(cfg, snr)
    h = slot["csi"]["perfect"][0]
    nmse = {n: _nmse(hh, h) for n, (hh, _) in slot["csi"].items()}
    assert nmse["ls_hard_window"] < nmse["ls_lin"]
    assert nmse["ls_soft_window"] < nmse["ls_lin"]
    # windowing rejects ~(1 - kept/M) of the noise: at 0 dB that is a large gain
    if snr == 0.0:
        assert nmse["ls_hard_window"] < 0.6 * nmse["ls_lin"]
        assert nmse["ls_soft_window"] < 0.3 * nmse["ls_lin"]


def test_error_variance_is_calibrated():
    cfg = _cfg(5.0, batch_size=4)
    sim, slot = _slot(cfg, 5.0)
    h = slot["csi"]["perfect"][0]
    for name in ("ls_hard_window", "ls_soft_window", "ls_lin"):
        h_hat, err = slot["csi"][name]
        mse = (h_hat - h).abs().pow(2).mean().item()
        reported = err.mean().item()
        # measured error contains model mismatch; reported variance is noise only
        assert 0.3 < reported / mse < 3.0, (name, mse, reported)


def test_soft_window_gain_reacts_to_snr():
    """More taps survive the Wiener threshold as the SNR grows."""
    kept = {}
    for snr in (-5.0, 25.0):
        sim, slot = _slot(_cfg(snr), snr)
        est = sim.rx.estimators["ls_soft_window"]
        hp, ep = est.estimate_at_pilot_locations(est._extract_pilots(slot["y"]), slot["no"])
        ep = torch.broadcast_to(ep, hp.shape)
        taps = torch.fft.ifft(est._block_estimates(hp), dim=-1)
        gain = est._tap_gains(taps, est._block_estimates(ep.to(hp.dtype)).real)
        # Measure surviving gains directly: reported MSE now includes an
        # empirical disturbance floor and removed channel energy as well.
        kept[snr] = gain.square().mean().item()
    assert kept[-5.0] < 0.25
    assert kept[25.0] > 2 * kept[-5.0]
    assert kept[25.0] <= 1.0


def test_windowed_ce_decodes():
    cfg = _cfg(12.0, channel_estimators=("ls_hard_window", "ls_soft_window"),
               receivers=("lmmse", "irc"), batch_size=2, max_mc_iter=1, num_target_bit_errors=1)
    sim = NRUplinkSimulator(cfg)
    stats = sim.measure_point(12.0, 0.0)
    for key, st in stats.items():
        assert st.ber < 0.05, (key, st.ber)
