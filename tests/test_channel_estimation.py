import torch

from nr_ul_sim import NRUplinkSimulator, SimConfig


def _nmse(h_hat, h_perf):
    return float(
        (h_hat - h_perf).abs().pow(2).mean() / (h_perf.abs().pow(2).mean() + 1e-12)
    )


def test_dmrs_ls_is_not_true_channel():
    cfg = SimConfig(
        num_prb=8,
        fft_size=1024,
        batch_size=1,
        max_mc_iter=1,
        num_target_bit_errors=1,
        receivers=("lmmse", "ideal_mmse"),
        snr_db=[20.0],
        iot_db=[0.0],
        seed=1,
    )
    sim = NRUplinkSimulator(cfg)
    slot = sim.generate_slot(1, 20.0, 0.0)
    sim.estimate_csi(slot)

    h_hat, h_perf = slot["h_hat"], slot["h_perf"]
    assert h_hat.shape == h_perf.shape
    assert h_hat.shape[-1] == cfg.num_used_subcarriers
    assert 1e-6 < _nmse(h_hat, h_perf) < 0.05
    assert not torch.allclose(h_hat, h_perf)

    b_hat = sim.detect(slot, "lmmse")
    assert b_hat.shape == slot["b"].shape


def test_estimators_order_by_nmse():
    cfg = SimConfig(
        num_prb=8,
        fft_size=1024,
        batch_size=1,
        max_mc_iter=1,
        num_target_bit_errors=1,
        receivers=("irc",),
        channel_estimators=("perfect", "ls_nn", "ls_lin", "ls_lin_time_avg", "lmmse_ce"),
        snr_db=[15.0],
        iot_db=[0.0],
        seed=2,
    )
    sim = NRUplinkSimulator(cfg)
    slot = sim.generate_slot(1, 15.0, 0.0)
    sim.estimate_csi(slot)
    h_perf = slot["csi"]["perfect"][0]
    nmse = {name: _nmse(h, h_perf) for name, (h, _) in slot["csi"].items()}
    assert nmse["perfect"] == 0.0
    assert nmse["lmmse_ce"] < nmse["ls_lin"] < nmse["ls_nn"]
    assert nmse["ls_lin_time_avg"] < nmse["ls_nn"]
    b_hat = sim.detect(slot, "irc", "ls_lin")
    assert b_hat.shape == slot["b"].shape
