"""Estimators from EqDeepRx (arXiv 2602.11834) and A-MMSE (arXiv 2506.00452)."""

import math

import numpy as np
import pytest
import torch

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.interference import irc_interference_covariance
from nr_ul_sim.paper_ce import (
    DEFAULT_MODEL_DIR,
    AMMSENet,
    DataCovariance,
    DenoiseNN,
    chunk_inputs,
    chunk_targets,
    freq_cov_from_samples,
    linear_interp_matrix,
    lmmse_matrices,
    oas_shrinkage,
    psd_clip,
    smoothing_taps,
)
from nr_ul_sim.parameters import snrdb_to_noise_var

LEARNED = ("denoise_nn", "lmmse_data", "lmmse_data_1d", "a_mmse", "ra_a_mmse")
HAVE_MODELS = all((DEFAULT_MODEL_DIR / f).exists() for f in ("denoise_nn.pt", "data_cov.pt", "a_mmse.pt"))
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="run train_paper_ce.py first")


def _nmse(a, b):
    return float((a - b).abs().pow(2).mean() / b.abs().pow(2).mean())


# --------------------------------------------------------------------------- building blocks

def test_linear_interp_matrix_is_exact_for_linear_functions():
    pos = np.arange(10) * 4 + 1.0
    f = linear_interp_matrix(40, pos)
    assert np.allclose(f.sum(1), 1.0)
    vals = 0.3 * pos - 2.0
    out = f @ vals
    inner = (np.arange(40) >= pos[0]) & (np.arange(40) <= pos[-1])
    assert np.allclose(out[inner], 0.3 * np.arange(40)[inner] - 2.0, atol=1e-5)


def test_fir_passband_follows_the_delay_window():
    df = 4 * 30e3
    g, lp = smoothing_taps(17, df, -1e-6, 3e-6)
    m = np.arange(17) - 8

    def gain(tau):
        return abs(np.sum(g * np.exp(2j * np.pi * m * df * tau)))

    for tau in (-0.5e-6, 0.0, 1.0e-6, 2.5e-6):
        assert gain(tau) == pytest.approx(1.0, abs=0.12)
    assert gain(6e-6) < 0.1
    assert abs(lp.sum() - 1) < 1e-5


def test_oas_shrinkage_limits():
    torch.manual_seed(0)
    p, n = 4, 48
    # white data: the sample covariance is mostly noise -> heavy shrinkage
    x = torch.complex(torch.randn(200, n, p), torch.randn(200, n, p)) / math.sqrt(2)
    s = torch.einsum("bnp,bnq->bpq", x, x.conj()) / n
    assert oas_shrinkage(s, n).mean() > 0.6
    # rank-1 dominated data with many samples: almost no shrinkage
    a = torch.randn(p, dtype=torch.complex64)
    y = torch.randn(200, 4000, 1, dtype=torch.complex64) * 10 * a + x[:, :1].expand(-1, 4000, -1)
    s2 = torch.einsum("bnp,bnq->bpq", y, y.conj()) / 4000
    assert oas_shrinkage(s2, 4000).mean() < 0.05


def test_oas_shrinkage_reduces_covariance_error():
    torch.manual_seed(1)
    p, n = 4, 48
    a = torch.randn(p, 2, dtype=torch.complex64)
    sigma = a @ a.conj().T + 0.5 * torch.eye(p)
    l = torch.linalg.cholesky(sigma)
    x = (torch.randn(500, n, p, dtype=torch.complex64)) @ l.T
    s = torch.einsum("bnp,bnq->bpq", x, x.conj()) / n
    rho = oas_shrinkage(s, n)[:, None, None]
    mu = torch.diagonal(s, dim1=-2, dim2=-1).real.sum(-1)[:, None, None] / p
    shrunk = (1 - rho) * s + rho * mu * torch.eye(p)
    err_s = (s - sigma).abs().pow(2).sum((-2, -1)).mean()
    err_k = (shrunk - sigma).abs().pow(2).sum((-2, -1)).mean()
    assert err_k < err_s


def _synthetic_grids(num, t=14, n=96, seed=0):
    """Random multipath grids, exponential PDP (300 ns), constant over the slot."""
    g = torch.Generator().manual_seed(seed)
    k = torch.arange(n)
    out = torch.zeros(num, t, n, dtype=torch.complex64)
    for i in range(num):
        taus = torch.rand(8, generator=g) * 1.5e-6
        amp = torch.complex(torch.randn(8, generator=g), torch.randn(8, generator=g)) * torch.exp(-taus / 300e-9) / 2
        h = (amp[:, None] * torch.exp(-2j * math.pi * k[None] * 30e3 * taus[:, None])).sum(0)
        out[i] = h[None]
    return out


def test_data_covariance_matches_sample_covariance_of_pilots():
    grids = _synthetic_grids(600, n=384)
    s, t, n = grids.shape
    r = freq_cov_from_samples(grids) / (s * n)
    cov = DataCovariance(r, n)
    syms, m = [2, 11], n // 4
    rxx, rhx = lmmse_matrices(cov, syms, m, 4, 0, t, n, "cpu")
    pos = torch.arange(m) * 4
    x = torch.cat([(grids[:, d, pos] + grids[:, d, pos + 2]) / 2 for d in syms], dim=1)    # [S, L]
    rxx_e = x.T @ x.conj() / s
    # the biased lag estimate tapers large lags; small lags must agree closely
    # (taper 1 - |lag|/N below 4 % for lags under 4 blocks = 16 of 384 subcarriers)
    lag = (torch.arange(2 * m)[:, None] % m - torch.arange(2 * m)[None] % m).abs()
    near = lag < 4
    rel = (rxx[near] - rxx_e[near]).abs() / rxx_e.diagonal().real.mean()
    assert rel.max() < 0.1
    ev = torch.linalg.eigvalsh(psd_clip(rxx))
    assert ev.min() >= -1e-5


def test_data_lmmse_beats_linear_interpolation_on_synthetic_channels():
    """Also catches B A^-T instead of B A^-1 for a complex Hermitian A."""
    train, test = _synthetic_grids(600, seed=1), _synthetic_grids(200, seed=2)
    s, t, n = train.shape
    cov = DataCovariance(freq_cov_from_samples(train) / (s * n), n)
    syms, m = [2, 11], n // 4
    rxx, rhx = lmmse_matrices(cov, syms, m, 4, 0, t, n, "cpu")
    pos = torch.arange(m) * 4
    x = torch.cat([(test[:, d, pos] + test[:, d, pos + 2]) / 2 for d in syms], dim=1)
    var = 0.05
    torch.manual_seed(3)
    xn = x + torch.complex(torch.randn_like(x.real), torch.randn_like(x.real)) * math.sqrt(var / 2)
    w = torch.linalg.solve((psd_clip(rxx) + var * torch.eye(2 * m)).T, rhx.T).T
    est = (xn @ w.T).reshape(-1, t, n)
    f = torch.as_tensor(linear_interp_matrix(n, (pos + 1).numpy().astype(float)), dtype=torch.complex64)
    lin = torch.einsum("nm,sm->sn", f, xn[:, :m])[:, None].expand(-1, t, -1)
    assert _nmse(est, test) < 0.5 * _nmse(lin, test)


def test_chunking_round_trip():
    blk = torch.arange(2 * 24, dtype=torch.float32).reshape(1, 2, 24).to(torch.complex64)
    x = chunk_inputs(blk, 12)
    assert x.shape == (2, 24)
    assert torch.equal(x[1, :12].real, blk[0, 0, 12:].real) and torch.equal(x[1, 12:].real, blk[0, 1, 12:].real)
    grid = torch.arange(14 * 96, dtype=torch.float32).reshape(1, 14, 96).to(torch.complex64)
    y = chunk_targets(grid, 48)
    assert y.shape == (2, 14 * 48)
    assert torch.equal(y[1].reshape(14, 48), grid[0, :, 48:])


def test_ammse_net_outputs_linear_filter_shape():
    net = AMMSENet(num_pilots=24, num_sc=48, num_sym=14)
    x = torch.randn(3, 48)
    w = net(x)
    assert w.shape == (3, 48 * 14, 24) and w.is_complex()


def test_denoise_nn_starts_as_identity():
    net = DenoiseNN()
    x = torch.randn(5, 2, 2, 204)
    assert torch.allclose(net(x), x)


# --------------------------------------------------------------------------- in the simulator

def _slot(estimators, channel="cdl-c", snr=10.0, num_ue=1, rank=1, iot=0.0, seed=5, **kw):
    cfg = SimConfig(channel=channel, num_prb=68, num_ue=num_ue, num_layers=rank, batch_size=2,
                    max_mc_iter=1, num_target_bit_errors=1, receivers=("lmmse", "irc"),
                    channel_estimators=("perfect",) + tuple(estimators), snr_db=[snr], iot_db=[iot],
                    seed=seed, **kw)
    sim = NRUplinkSimulator(cfg)
    slot = sim.generate_slot(2, snr, iot)
    sim.estimate_csi(slot)
    return sim, slot


def test_fir_estimator_beats_linear_interpolation():
    _, slot = _slot(("ls_lin", "ls_fir"), snr=0.0)
    h = slot["csi"]["perfect"][0]
    assert _nmse(slot["csi"]["ls_fir"][0], h) < 0.6 * _nmse(slot["csi"]["ls_lin"][0], h)


@needs_models
@pytest.mark.parametrize("channel", ["uma", "cdl-c"])
def test_learned_estimators_run_and_beat_ls(channel):
    sim, slot = _slot(("ls_nn", "ls_lin") + LEARNED, channel=channel, snr=5.0)
    h = slot["csi"]["perfect"][0]
    for name in LEARNED:
        h_hat, err = slot["csi"][name]
        assert h_hat.shape == h.shape and torch.isfinite(h_hat).all()
        assert torch.all(err > 0)
        assert _nmse(h_hat, h) < _nmse(slot["csi"]["ls_nn"][0], h), name
    stats = sim.measure_point(5.0, 0.0)
    assert all(s.num_bits > 0 for s in stats.values())


@needs_models
def test_learned_estimators_handle_four_layers():
    _, slot = _slot(("ls_lin",) + LEARNED, num_ue=2, rank=2, snr=20.0)
    h = slot["csi"]["perfect"][0]
    for name in LEARNED:
        assert _nmse(slot["csi"][name][0], h) < 0.2, name


def test_incm_oas_is_close_to_the_true_interference_plus_noise():
    sim, slot = _slot(("ls_lin",), snr=15.0, iot=15.0, num_interferers=1)
    no = snrdb_to_noise_var(15.0)
    r_true = irc_interference_covariance(slot["h_int"], slot["y"], no, 15.0, "perfect")
    r = irc_interference_covariance(slot["h_int"], slot["y"], no, 15.0, "incm_oas",
                                    h_hat=slot["h_perf"], pilot_grid=sim.rx.pilot_grid,
                                    dmrs_syms=sim.rx.dmrs_syms, guard_carriers=sim.cfg.guard_carriers)
    assert r.shape == (2, 816, 4, 4)
    herm = (r - r.conj().transpose(-1, -2)).abs().max()
    assert herm < 1e-4
    # wideband truth vs per-2-PRB estimate: same total power, structure within ~40 %
    tr_t = torch.diagonal(r_true, dim1=-2, dim2=-1).real.sum(-1).mean()
    tr_e = torch.diagonal(r, dim1=-2, dim2=-1).real.sum(-1).mean()
    assert float(tr_e / tr_t) == pytest.approx(1.0, abs=0.2)


def test_per_subcarrier_covariance_equals_wideband_when_constant():
    sim, slot = _slot(("ls_lin",), snr=8.0, iot=10.0, num_interferers=1)
    no = snrdb_to_noise_var(8.0)
    r = irc_interference_covariance(slot["h_int"], slot["y"], no, 10.0, "perfect")
    h, err = slot["csi"]["ls_lin"]
    sim.rx.irc_eq.set_covariance(r)
    b_wide = sim.rx.decode(slot["y"], h, err, slot["no"], "irc")
    sim.rx.irc_eq.set_covariance(r[:, None].expand(-1, 816, -1, -1).contiguous())
    b_sc = sim.rx.decode(slot["y"], h, err, slot["no"], "irc")
    assert torch.equal(b_wide, b_sc)


def test_irc_with_incm_oas_rejects_interference():
    cfg = dict(num_interferers=1, iot_cov="incm_oas")
    sim, _ = _slot(("ls_lin",), snr=10.0, iot=20.0, **cfg)
    sim.cfg.max_mc_iter = 3
    stats = sim.measure_point(10.0, 20.0)
    assert stats["irc_ls_lin"].ber < stats["lmmse_ls_lin"].ber
