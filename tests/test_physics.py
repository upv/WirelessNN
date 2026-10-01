"""Element-by-element physics checks of the link simulator (68 PRB).

Every block is exercised on its own: transmitter (TB size, power, DMRS),
channel (energy, delay spread, time / frequency / spatial correlation), noise
and interference calibration, IRC covariance, receiver identities.
"""

import math

import numpy as np
import pytest
import torch
from scipy.special import j0

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.interference import (
    interference_scale,
    irc_interference_covariance,
    sample_covariance,
    sample_covariance_from_y,
    spatial_covariance_from_channel,
)
from nr_ul_sim.parameters import snrdb_to_noise_var
from nr_ul_sim.pusch import strip_guard_subcarriers

PRB = 68
SC = 12 * PRB


def _sim(**kw):
    base = dict(num_prb=PRB, fft_size=1024, batch_size=2, max_mc_iter=1,
                num_target_bit_errors=1, snr_db=[10.0], iot_db=[0.0], seed=11)
    base.update(kw)
    return NRUplinkSimulator(SimConfig(**base))


def tbs_38214(n_prb, coderate, qm, layers, n_dmrs_re_per_prb=24, n_oh=0):
    """Transport block size, 38.214 §5.1.3.2 (single slot, 14 symbols)."""
    n_re = min(156, 12 * 14 - n_dmrs_re_per_prb - n_oh) * n_prb
    n_info = n_re * coderate * qm * layers
    if n_info <= 3824:
        n = max(3, math.floor(math.log2(n_info)) - 6)
        n_info_q = max(24, 2**n * math.floor(n_info / 2**n))
        table = [24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
                 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384,
                 408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
                 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352, 1416,
                 1480, 1544, 1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280,
                 2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
                 3624, 3752, 3824]
        return next(t for t in table if t >= n_info_q)
    n = math.floor(math.log2(n_info - 24)) - 5
    n_info_q = max(3840, 2**n * round((n_info - 24) / 2**n))
    if coderate <= 0.25:
        c = math.ceil((n_info_q + 24) / 3816)
    elif n_info_q > 8424:
        c = math.ceil((n_info_q + 24) / 8424)
    else:
        return 8 * math.ceil((n_info_q + 24) / 8) - 24
    return 8 * c * math.ceil((n_info_q + 24) / (8 * c)) - 24


# --------------------------------------------------------------------------- #
# transmitter
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("modulation,rank,coderate,qm", [
    ("qpsk", 1, 379 / 1024, 2),
    ("qam16", 2, 553 / 1024, 4),
    ("qam64", 1, 567 / 1024, 6),
])
def test_tb_size_matches_38214(modulation, rank, coderate, qm):
    sim = _sim(modulation=modulation, num_layers=rank)
    tx = sim.transmitter
    assert float(tx._target_coderate) == pytest.approx(coderate, abs=1e-6)
    assert int(tx._tb_size) == tbs_38214(PRB, coderate, qm, rank)
    assert tx._tb_encoder.coderate < 1.0


def test_grid_layout_and_power():
    cfg = SimConfig(num_prb=PRB, num_ue=2, num_layers=1)
    sim = NRUplinkSimulator(cfg)
    rg = sim.transmitter.resource_grid
    assert rg.fft_size == 1024 and rg.num_effective_subcarriers == SC
    assert tuple(rg.num_guard_carriers) == (104, 104)
    x, b = sim.transmit(4)
    assert x.shape == (4, 2, 1, 14, 1024)
    assert torch.all(x[..., :104] == 0) and torch.all(x[..., -104:] == 0)
    used = strip_guard_subcarriers(x, cfg.guard_carriers)
    power = used.abs().pow(2).mean(dim=(0, 2, 3, 4))
    assert torch.allclose(power, torch.ones_like(power), atol=0.03)


def test_rank2_total_ue_power_is_one():
    sim = _sim(num_layers=2, modulation="qam16")
    x, _ = sim.transmit(4)
    used = strip_guard_subcarriers(x, sim.cfg.guard_carriers)
    per_layer = used.abs().pow(2).mean(dim=(0, 1, 3, 4))
    assert per_layer.sum().item() == pytest.approx(1.0, abs=0.05)


def test_dmrs_layout_type1():
    sim = _sim(num_ue=2, num_layers=2, modulation="qam16")
    rx = sim.rx
    grid = rx.pilot_grid  # [Tx, S, Sym, Sc]
    assert rx.dmrs_syms == [2, 11]
    assert grid.shape == (2, 2, 14, SC)
    nz = grid.abs() > 0
    assert nz[:, :, [0, 1] + list(range(3, 11)) + [12, 13]].sum() == 0
    # ports 0,1 on the even comb (CDM group 0), ports 2,3 on the odd comb
    assert nz[0, 0, 2, 0::2].all() and not nz[0, 0, 2, 1::2].any()
    assert nz[1, 0, 2, 1::2].all() and not nz[1, 0, 2, 0::2].any()
    # DMRS power boost with two CDM groups without data: |p|^2 = 2 (3 dB)
    assert grid[0, 0, 2, 0].abs().pow(2).item() == pytest.approx(2.0, rel=1e-4)


# --------------------------------------------------------------------------- #
# channel
# --------------------------------------------------------------------------- #

def _rms_delay_spread(h_used, subcarrier_spacing):
    """RMS delay spread from the PDP of the frequency response over used subcarriers."""
    taps = torch.fft.ifft(h_used, dim=-1)
    n = taps.shape[-1]
    pdp = taps.abs().pow(2).mean(dim=tuple(range(taps.ndim - 1))).cpu().numpy()
    ts = 1.0 / (n * subcarrier_spacing)
    t = np.arange(n) * ts
    t[n // 2 :] -= n * ts  # wrapped negative delays
    order = np.argsort(t)
    t, pdp = t[order], pdp[order]
    pdp = pdp / pdp.sum()
    mean = (pdp * t).sum()
    return float(np.sqrt((pdp * (t - mean) ** 2).sum())), t, pdp


def _cir_rms_ds(link, batch: int = 64) -> float:
    a, tau = link(batch, 1, 1.0)
    p = a.abs().pow(2).mean(dim=(0, 1, 2, 3, 4, 6)).cpu().numpy()
    t = tau[0, 0, 0].cpu().numpy()
    p = p / p.sum()
    mean = (p * t).sum()
    return float(np.sqrt((p * (t - mean) ** 2).sum()))


def _omni_cdl(cfg):
    from sionna.phy.channel.tr38901 import CDL, AntennaArray

    omni = AntennaArray(num_rows=1, num_cols=1, polarization="single", polarization_type="V",
                        antenna_pattern="omni", carrier_frequency=cfg.carrier_frequency)
    return CDL(model={"cdl-b": "B", "cdl-c": "C"}[cfg.channel], delay_spread=cfg.delay_spread,
               carrier_frequency=cfg.carrier_frequency, ut_array=omni, bs_array=omni,
               direction="uplink", min_speed=cfg.speed_mps, max_speed=cfg.speed_mps)


@pytest.mark.parametrize("channel,ds_ns", [("cdl-b", 100.0), ("cdl-c", 300.0)])
def test_cdl_delay_spread_and_energy(channel, ds_ns):
    sim = _sim(channel=channel, delay_spread_ns=ds_ns, batch_size=16, seed=3)
    h = sim.gen_serving(16)
    used = strip_guard_subcarriers(h, sim.cfg.guard_carriers)
    energy = used.abs().pow(2).mean().item()
    assert energy == pytest.approx(1.0, abs=0.08)
    # RMS delay spread from the cluster impulse response (a, tau) of the model.
    # With omnidirectional elements it equals the configured value; the 38.901
    # element pattern of the gNB panel weights the clusters by their angle of
    # arrival and shrinks the effective delay spread (CDL-B 100 -> ~55 ns).
    ds_pattern = _cir_rms_ds(sim.serving_model.links[0])
    ds_omni = _cir_rms_ds(_omni_cdl(sim.cfg))
    assert ds_omni * 1e9 == pytest.approx(ds_ns, rel=0.1)
    assert 0.15 * ds_omni < ds_pattern < ds_omni
    # the band-limited PDP measured from H over the 24.5 MHz band carries the
    # Dirichlet leakage of every cluster, so it can only be checked loosely
    ds_f, _, _ = _rms_delay_spread(used[:, 0, :, 0, 0, 0, :], sim.cfg.subcarrier_spacing)
    assert 0.5 * ds_pattern < ds_f < 4.0 * ds_pattern + 100e-9


def test_cdl_pdp_is_causal_and_inside_cp():
    sim = _sim(channel="cdl-c", delay_spread_ns=300.0, batch_size=16, seed=5)
    h = strip_guard_subcarriers(sim.gen_serving(16), sim.cfg.guard_carriers)
    _, t, pdp = _rms_delay_spread(h[:, 0, :, 0, 0, 0, :], sim.cfg.subcarrier_spacing)
    cp = 2.34e-6
    assert pdp[(t >= 0) & (t <= cp)].sum() > 0.97
    assert pdp[t < -0.3e-6].sum() < 0.01


@pytest.mark.parametrize("speed_kmh", [3.0, 10.0])
def test_time_correlation_follows_jakes(speed_kmh):
    sim = _sim(channel="cdl-c", speed_kmh=speed_kmh, batch_size=32, seed=8)
    h = strip_guard_subcarriers(sim.gen_serving(32), sim.cfg.guard_carriers)
    h = h[:, 0, :, 0, 0]  # [B, RxAnt, Sym, Sc]
    rg = sim.transmitter.resource_grid
    fd = speed_kmh / 3.6 * sim.cfg.carrier_frequency / 299792458.0
    lag = 13
    num = (h[..., lag, :] * h[..., 0, :].conj()).mean()
    den = h[..., 0, :].abs().pow(2).mean()
    rho = (num / den).abs().item()
    expected = float(j0(2 * np.pi * fd * lag * rg.ofdm_symbol_duration))
    assert rho == pytest.approx(expected, abs=0.03)
    assert rho > 0.9


def test_frequency_correlation_decays_with_delay_spread():
    rhos = {}
    for ds in (30.0, 300.0):
        sim = _sim(channel="cdl-c", delay_spread_ns=ds, batch_size=16, seed=9)
        h = strip_guard_subcarriers(sim.gen_serving(16), sim.cfg.guard_carriers)[:, 0, :, 0, 0, 0]
        lag = 40  # 1.2 MHz
        num = (h[..., lag:] * h[..., :-lag].conj()).mean()
        den = h.abs().pow(2).mean()
        rhos[ds] = (num / den).abs().item()
    assert rhos[30.0] > 0.95
    assert rhos[300.0] < rhos[30.0] - 0.1


def test_spatial_correlation_of_cross_polarised_array():
    sim = _sim(channel="cdl-c", batch_size=32, seed=12)
    h = strip_guard_subcarriers(sim.gen_serving(32), sim.cfg.guard_carriers)
    r = spatial_covariance_from_channel(h[..., :1, :1, :, :]).mean(dim=0).cpu()
    d = r.diagonal().real.sqrt()
    rho = (r / torch.outer(d, d)).abs()
    assert torch.allclose(rho, rho.T, atol=1e-5)
    assert torch.all(rho.diagonal() > 0.999)
    # 1x2 dual-polarised panel: elements 0/1 are the two polarisations of column 0
    off = rho[~torch.eye(4, dtype=bool)]
    assert off.max() < 0.95  # no degenerate (fully correlated) antenna pair
    assert off.min() < 0.6   # cross-polar pairs are weakly correlated


def test_system_level_channel_runs_for_umi_uma():
    for name in ("umi", "uma"):
        sim = _sim(channel=name, batch_size=2, num_ue=2, seed=4)
        slot = sim.generate_slot(2, 10.0, 10.0)
        assert slot["h"].shape == (2, 1, 4, 2, 1, 14, 1024)
        assert slot["h_int"].shape == (2, 1, 4, 2, 1, 14, 1024)
        used = strip_guard_subcarriers(slot["h"], sim.cfg.guard_carriers)
        assert used.abs().pow(2).mean().item() == pytest.approx(1.0, abs=0.15)


# --------------------------------------------------------------------------- #
# noise, interference, IRC covariance
# --------------------------------------------------------------------------- #

def test_noise_and_snr_calibration():
    sim = _sim(batch_size=4, seed=21)
    for snr in (-10.0, 0.0, 15.0):
        slot = sim.generate_slot(4, snr, 0.0)
        # noiseless reference: fresh data through the stored channel
        clean = sim.apply_channel(sim.transmit(4)[0], slot["h"])
        no = snrdb_to_noise_var(snr)
        y = strip_guard_subcarriers(slot["y"], sim.cfg.guard_carriers)
        sig = strip_guard_subcarriers(clean, sim.cfg.guard_carriers)
        assert sig.abs().pow(2).mean().item() == pytest.approx(1.0, abs=0.15)
        assert y.abs().pow(2).mean().item() == pytest.approx(1.0 + no, rel=0.1)


@pytest.mark.parametrize("num_int,rank", [(1, 1), (2, 1), (3, 2)])
def test_interference_to_noise_ratio(num_int, rank):
    sim = _sim(num_interferers=num_int, num_layers=rank, modulation="qam16", batch_size=4, seed=31)
    cfg = sim.cfg
    no = snrdb_to_noise_var(10.0)
    h_i = sim.gen_int(4)
    x_i = sim.interferer_grid(4, h_i)
    y_i = strip_guard_subcarriers(sim.apply_channel(x_i, h_i), cfg.guard_carriers)
    for iot in (5.0, 10.0, 20.0):
        p = (interference_scale(no, iot, cfg.num_interferer_streams) * y_i).abs().pow(2).mean().item()
        assert 10 * np.log10(p / no) == pytest.approx(iot, abs=0.6)


def test_sample_covariance_is_not_conjugated():
    rng = torch.Generator().manual_seed(0)
    a = torch.randn(4, 4, generator=rng, dtype=torch.complex64)
    s = torch.randn(1, 40000, 4, generator=rng, dtype=torch.complex64)  # E|s|^2 = 1
    y = s @ a.T  # y_n = a s_n
    r = sample_covariance(y)[0]
    r_true = a @ a.conj().T
    assert (r - r_true).norm() / r_true.norm() < 0.05
    assert (r - r_true.conj()).norm() / r_true.norm() > 0.2  # the conjugate is a different matrix
    y5 = y.reshape(1, 1, 40000, 1, 4).permute(0, 1, 4, 2, 3)  # [B, Rx, M, Sym, Sc]
    assert torch.allclose(sample_covariance_from_y(y5)[0], r, atol=1e-5)


def test_irc_covariances_agree():
    sim = _sim(num_interferers=2, batch_size=1, seed=41, iot_cov="perfect")
    cfg = sim.cfg
    no = snrdb_to_noise_var(20.0)
    slot = sim.generate_slot(1, 20.0, 15.0)
    sim.estimate_csi(slot)
    h_perf = sim.true_channel(slot["h"])
    r_perf = irc_interference_covariance(slot["h_int"], slot["y"], no, 15.0, "perfect")
    r_res = irc_interference_covariance(
        slot["h_int"], slot["y"], no, 15.0, "residual",
        h_hat=h_perf, pilot_grid=sim.rx.pilot_grid, dmrs_syms=sim.rx.dmrs_syms,
        guard_carriers=cfg.guard_carriers,
    )
    # both are the interference covariance; the residual one is a sample
    # estimate over 1632 REs with one realisation of the interferer symbols
    tr_p, tr_r = r_perf[0].diagonal().real.sum().item(), r_res[0].diagonal().real.sum().item()
    assert tr_p == pytest.approx(4 * 10 ** 1.5 * no, rel=0.3)
    assert tr_r == pytest.approx(tr_p, rel=0.3)
    err = (r_res - r_perf).norm().item() / r_perf.norm().item()
    assert err < 0.35


# --------------------------------------------------------------------------- #
# receivers and decoding
# --------------------------------------------------------------------------- #

def test_irc_without_interference_equals_lmmse():
    sim = _sim(receivers=("lmmse", "irc"), batch_size=2, seed=51)
    slot = sim.generate_slot(2, 5.0, 0.0)
    sim.estimate_csi(slot)
    b_lmmse = sim.detect(slot, "lmmse")
    b_irc = sim.detect(slot, "irc")
    assert torch.equal(b_lmmse, b_irc)


def test_ideal_mmse_equals_lmmse_with_perfect_csi():
    sim = _sim(receivers=("lmmse", "ideal_mmse"), channel_estimators=("perfect",), batch_size=2, seed=52)
    slot = sim.generate_slot(2, 3.0, 0.0)
    sim.estimate_csi(slot)
    assert torch.equal(sim.detect(slot, "lmmse", "perfect"), sim.detect(slot, "ideal_mmse"))


def test_all_receivers_error_free_at_high_snr():
    sim = _sim(receivers=("mr", "lmmse", "ideal_mmse", "zf", "irc"), num_ue=2, batch_size=2, seed=53,
               max_mc_iter=2)
    stats = sim.measure_point(30.0, 0.0)
    for name, st in stats.items():
        assert st.ber == 0.0 and st.bler == 0.0, name
        assert st.num_blocks == 8  # 2 iter x batch 2 x 2 UEs


def test_irc_rejects_coloured_interference():
    sim = _sim(receivers=("lmmse", "irc"), num_interferers=1, batch_size=4, seed=54, max_mc_iter=3)
    stats = sim.measure_point(10.0, 20.0)
    assert stats["irc"].ber < stats["lmmse"].ber
