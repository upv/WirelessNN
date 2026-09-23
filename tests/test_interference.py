import torch

from nr_ul_sim.interference import spatial_covariance_from_channel


def test_spatial_covariance_shape_and_psd():
    b, m, k, nre = 3, 4, 2, 20
    h = torch.randn(b, 1, m, k, 1, nre, 1, dtype=torch.complex64)
    r = spatial_covariance_from_channel(h)
    assert r.shape == (b, m, m)
    assert torch.allclose(r, r.transpose(-1, -2).conj(), atol=1e-5)
    eig = torch.linalg.eigvalsh(r)
    assert torch.all(eig > -1e-4)
