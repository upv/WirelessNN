import numpy as np
import pytest
import torch
from experiments.ce_denoising.generate import generate_record, plan, pilot_blocks, validate_record
from experiments.ce_denoising.losses import channel_mse, symbol_bce


def test_plan_split_disjoint_ranges_and_balance():
    rows = plan(counts=(45, 45, 45))
    assert rows == plan(counts=(45, 45, 45))
    assert len({r['seed'] for r in rows}) == len(rows)
    for split in ('train', 'validation', 'test'):
        part = [r for r in rows if r['split'] == split]
        assert len({(r['channel'], r['iot_db'], r['snr_bin_low_db']) for r in part}) == 45
        assert all(2 <= r['num_interferers_configured'] <= 6 and -20 <= r['snr_db'] <= 10 for r in part)
    with pytest.raises(ValueError):
        plan(counts=(1, 45, 45))


@pytest.mark.parametrize('channel,iot', [('cdl-b', 0), ('cdl-c', 10), ('cdl-d', 20), ('uma', 10), ('umi', 20)])
def test_record_physics_and_all_channel_families(channel, iot):
    row = dict(id='test_000000', split='test', seed=319, channel=channel,
               iot_db=iot, snr_db=-5.0, snr_bin_low_db=-10,
               modulation='qam16', num_interferers_configured=3)
    data, params, meta = generate_record(row)
    validate_record(data, meta)
    assert data['h_true'].shape == (4, 14, 192)
    assert data['ls_pilots'].shape == (4, 2, 48)
    assert data['tx_coded_bits'].shape == (2304, 4)
    assert params and not any(a.dtype == object for a in params.values())
    assert abs(meta['measured_snr_db'] - row['snr_db']) < 2
    if iot:
        assert abs(meta['measured_inr_db'] - iot) < 2
    h = data['h_true']
    dmrs = data['dmrs_symbols']
    expected_clean = (h[:, dmrs, 0::4] + h[:, dmrs, 2::4]) / 2
    np.testing.assert_allclose(data['ls_pilots_clean'], expected_clean, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(data['h_ls'][:, dmrs, 0::4], data['ls_pilots'], rtol=2e-6, atol=1e-6)
    from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
    frequencies = subcarrier_frequencies(256, 30e3)
    rebuilt = cir_to_ofdm_channel(frequencies, torch.tensor(data['cir_target_a']),
                                  torch.tensor(data['cir_target_tau_s']), normalize=True)
    np.testing.assert_allclose(rebuilt[0,0,:,0,0,:,32:-32].cpu().numpy(), h, rtol=2e-5, atol=2e-6)
    # Same exact generation recipe reproduces tensor data on the same runtime.
    if channel == 'cdl-b':
        repeat, _, _ = generate_record(row)
        np.testing.assert_array_equal(data['y'], repeat['y'])
    pred = torch.tensor(data['h_ls'], requires_grad=True)
    args = [torch.tensor(data[k]) for k in ('y', 'x_tx', 'data_mask')]
    loss = symbol_bce(pred, *args, row['modulation'], meta['noise_variance'] + meta['interference_variance_nominal'])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_bce_noiseless_true_channel_and_wrong_channel():
    from nr_ul_sim.losses import constellation
    for mod in ('qpsk', 'qam16', 'qam64'):
        x, _ = constellation(mod)
        x = x[None]
        h = torch.ones(4, 1, x.numel(), dtype=torch.complex64)
        mask = torch.ones_like(x, dtype=torch.bool)
        y = h * x
        assert symbol_bce(h, y, x, mask, mod, .001) < .001
        assert symbol_bce(-h, y, x, mask, mod, .001) > 1
    assert channel_mse(torch.ones(2, dtype=torch.complex64) * (1+1j), torch.zeros(2)).item() == pytest.approx(2)
