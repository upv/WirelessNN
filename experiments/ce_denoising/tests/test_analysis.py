import json
import numpy as np
import pytest
import torch

from experiments.ce_denoising.analyze import hard_window, linear_fit, predict, nominal_pilot_variance
from experiments.ce_denoising.generate import run
from experiments.ce_denoising.dataset import CEDataset
from nr_ul_sim import SimConfig
from nr_ul_sim.pusch import build_transmitter
from nr_ul_sim.channel_estimation import build_estimator


def test_complex_linear_fit_orientation_and_regularization():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(250, 8)) + 1j*rng.normal(size=(250, 8))
    w = rng.normal(size=(8, 14)) + 1j*rng.normal(size=(8, 14))
    y = x@w + 1.0-2j
    fit = linear_fit(x[:200], y[:200])
    np.testing.assert_allclose(predict(fit, x[200:], 1e-10), y[200:], rtol=1e-8, atol=1e-8)
    assert np.var(predict(fit, x[200:], 1000), axis=0).mean() < np.var(y[200:], axis=0).mean()/100
    assert nominal_pilot_variance({'snr_db':0,'iot_db':0}) == .25
    assert nominal_pilot_variance({'snr_db':0,'iot_db':10}) == 2.75


def test_fixed_dft_filter_matches_simulator():
    cfg = SimConfig(num_prb=16, fft_size=256, receivers=('lmmse',))
    estimator = build_estimator('ls_hard_window', build_transmitter(cfg), cfg)
    rng = np.random.default_rng(2)
    pilots = (rng.normal(size=(4,2,48)) + 1j*rng.normal(size=(4,2,48))).astype(np.complex64)
    p = torch.from_numpy(pilots).to(estimator.device)[None,None,:,None,None]
    taps = torch.fft.ifft(p, dim=-1) * estimator._window
    d = estimator._to_full_grid(taps, estimator._centre)
    expected = torch.einsum('ld,...dn->...ln', estimator._time_w.to(d.dtype), d)[0,0,:,0,0]
    np.testing.assert_allclose(hard_window(pilots,192,estimator.dmrs_syms), expected.cpu().numpy(), atol=5e-7, rtol=1e-5)


def test_resume_loader_and_corruption_detection(tmp_path):
    dest = tmp_path/'pilot'
    run(dest, counts=(45,45,45), limit=1)
    original = (dest/'records/train_000000/sample.npz').read_bytes()
    run(dest, counts=(45,45,45), limit=1, resume=True)
    assert (dest/'records/train_000000/sample.npz').read_bytes() == original
    ds = CEDataset(dest, include_bce=True)
    sample = ds[0]
    assert sample['input'].shape == sample['target'].shape == (4,14,192)
    assert set(sample) == {'id','input','target','y','x_tx','data_mask','modulation','disturbance_variance'}
    with pytest.raises(ValueError, match='identical'):
        run(dest, counts=(45,45,45), seed=22, limit=1, resume=True)
    p = dest/'records/train_000000/sample.npz'
    p.write_bytes(original[:-16])
    with pytest.raises(ValueError, match='Checksum'):
        run(dest, counts=(45,45,45), limit=1, resume=True)
