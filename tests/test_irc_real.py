"""Received-signal-only covariance and empirical soft-window noise calibration."""

import pytest
import torch

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.cli import build_parser, config_from_args
from nr_ul_sim.plotting import _style_for


def make_sim(**kwargs):
    opts = dict(num_prb=16, batch_size=2, max_mc_iter=1, seed=83,
                receivers=("irc", "irc_real"), channel_estimators=("ls_soft_window",))
    opts.update(kwargs)
    return NRUplinkSimulator(SimConfig(**opts))


def test_real_irc_covariance_does_not_use_true_channels_or_iot():
    sim = make_sim()
    slot = sim.generate_slot(2, 10.0, 20.0)
    sim.estimate_csi(slot)
    # Remove all simulator-only channel information before detection.
    slot.pop("h")
    slot.pop("h_int")
    slot["iot_db"] = -100.0
    bits = sim.detect(slot, "irc_real")
    r = slot["real_covariance"]
    rin = r + slot["no"] * torch.eye(4, device=r.device)
    assert r.shape == (2, 192, 4, 4)
    assert torch.isfinite(r).all()
    assert torch.linalg.eigvalsh(rin).min() > 0
    assert bits.shape == slot["b"].shape
    assert torch.equal(bits, sim.detect(slot, "irc_real"))


def test_genie_detector_still_estimates_real_covariance_from_received_signal():
    sim = make_sim(channel_estimators=("perfect",), receivers=("irc_real",))
    slot = sim.generate_slot(2, 10.0, 10.0)
    sim.estimate_csi(slot)
    sim.detect(slot, "irc_real", "perfect")
    cov = slot["real_covariance"].clone()
    slot["h_perf"] = slot["h_perf"] * 0.1
    slot.pop("real_covariance")
    slot.pop("real_cov_csi")
    sim.detect(slot, "irc_real", "perfect")
    assert torch.equal(cov, slot["real_covariance"])


def test_outside_noise_tracks_interference_and_resists_sparse_leakage():
    sim = make_sim()
    est = sim.rx.estimators["ls_soft_window"]
    # Complex white disturbances with ten times the supplied thermal floor.
    torch.manual_seed(17)
    m = est.num_blocks
    taps = torch.randn(256, 1, 4, 1, 1, 2, m, device=sim.device,
                       dtype=torch.complex64) * (0.1 ** 0.5)
    taps[..., 0] += 5.0
    outside = torch.where(est._window == 0)[0]
    taps[..., outside[:2]] += 3.0
    err = torch.full_like(taps.real, 0.01 * m)
    _, measured = est._tap_statistics(taps, err)
    assert measured.mean().item() == pytest.approx(0.1, rel=0.25)
    est.soft_noise_mode = "thermal"
    _, old = est._tap_statistics(taps, err)
    assert old.mean().item() == pytest.approx(0.01)


def test_noise_floor_falls_back_when_window_covers_all_taps():
    sim = make_sim(ce_window_pos_us=100.0, ce_window_neg_us=100.0)
    est = sim.rx.estimators["ls_soft_window"]
    taps = torch.ones(1, 1, 4, 1, 1, 2, est.num_blocks,
                      dtype=torch.complex64, device=sim.device)
    _, noise = est._tap_statistics(taps, torch.ones_like(taps.real))
    assert noise.mean().item() == pytest.approx(1.0 / est.num_blocks)


def test_empirical_soft_window_improves_csi_with_strong_interference():
    sim = make_sim(num_prb=68, batch_size=16, num_interferers=1)
    slot = sim.generate_slot(16, 10.0, 20.0)
    est = sim.rx.estimators["ls_soft_window"]
    true = sim.true_channel(slot["h"])
    empirical, _ = est(slot["y"], slot["no"])
    est.soft_noise_mode = "thermal"
    thermal, _ = est(slot["y"], slot["no"])
    assert (empirical - true).abs().square().mean() < (thermal - true).abs().square().mean()


def test_real_irc_cli_and_plot_label():
    cfg = config_from_args(build_parser().parse_args(
        ["--receivers", "irc,irc_real", "--ce-soft-noise-mode", "thermal"]))
    assert cfg.receivers == ("irc", "irc_real")
    assert cfg.summary()["ce_soft_noise_mode"] == "thermal"
    assert "DMRS + OAS" in _style_for("irc_real_ls_soft_window")["label"]
    with pytest.raises(ValueError):
        SimConfig(incm_band_sc=0)
