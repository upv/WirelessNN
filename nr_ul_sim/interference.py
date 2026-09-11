"""Other-cell interference (IoT) and IRC spatial covariance."""

from __future__ import annotations

import torch

from .parameters import db_to_lin


def random_ofdm_grid(
    batch_size: int,
    num_tx: int,
    num_tx_ant: int,
    num_ofdm_symbols: int,
    fft_size: int,
    *,
    dtype,
    device,
    guard_carriers: tuple[int, int],
) -> torch.Tensor:
    """Unit-power complex Gaussian symbols on used subcarriers (zeros on guards)."""
    x = torch.randn(
        batch_size,
        num_tx,
        num_tx_ant,
        num_ofdm_symbols,
        fft_size,
        device=device,
        dtype=dtype,
    )
    x = torch.complex(x, torch.randn_like(x)) / (2.0 ** 0.5)
    left, right = guard_carriers
    if left:
        x[..., :left] = 0
    if right:
        x[..., fft_size - right :] = 0
    return x


def spatial_covariance_from_channel(h: torch.Tensor) -> torch.Tensor:
    """Average spatial covariance ``E[H H^H]`` over streams and resource elements.

    Parameters
    ----------
    h:
        Interferer channel ``[B, Rx, RxAnt, Tx, TxAnt, Sym, Sc]``.

    Returns
    -------
    r
        Covariance ``[B, RxAnt, RxAnt]``.
    """
    b, _, rx_ant, num_tx, tx_ant, num_sym, num_sc = h.shape
    # [B, Nre, RxAnt, K]
    h_re = h[:, 0].reshape(b, rx_ant, num_tx * tx_ant, num_sym * num_sc)
    h_re = h_re.permute(0, 3, 1, 2)
    r = torch.matmul(h_re, h_re.conj().transpose(-1, -2))
    return r.mean(dim=1)


def sample_covariance_from_y(y: torch.Tensor) -> torch.Tensor:
    """Sample spatial covariance of a received grid ``[B, Rx, RxAnt, Sym, Sc]``."""
    b, _, rx_ant, num_sym, num_sc = y.shape
    y_mat = y[:, 0].permute(0, 2, 3, 1).reshape(b, num_sym * num_sc, rx_ant)
    r = torch.matmul(y_mat.transpose(-1, -2).conj(), y_mat) / (num_sym * num_sc)
    return r


def project_psd(r: torch.Tensor, min_eig: float = 1e-8) -> torch.Tensor:
    """Hermitian PSD projection with a small eigenvalue floor."""
    r_h = 0.5 * (r + r.transpose(-1, -2).conj())
    eigvals, eigvecs = torch.linalg.eigh(r_h)
    eigvals = torch.clamp(eigvals.real, min=min_eig).to(eigvals.dtype)
    return eigvecs @ torch.diag_embed(eigvals.to(eigvecs.dtype)) @ eigvecs.transpose(-1, -2).conj()


def interference_power_scale(iot_db: float, no: float) -> float:
    """Linear amplitude scale so that I/N equals ``iot_db`` (0 dB → no IoT).

    The platform treats 0 dB as *no neighbouring-cell interference*. Values
    10 dB and 20 dB are interference-to-noise ratios (INR).
    """
    if iot_db <= 0.0:
        return 0.0
    return (db_to_lin(iot_db) * float(no)) ** 0.5


def serving_plus_interference(
    y_s: torch.Tensor,
    y_i: torch.Tensor | None,
    no: float,
    iot_db: float,
) -> torch.Tensor:
    """y = y_serving + sqrt(INR * N0) y_int + w, with 0 dB IoT meaning y_int = 0."""
    scale = interference_power_scale(iot_db, no)
    y = y_s
    if y_i is not None and scale > 0.0:
        y = y + scale * y_i
    std = (float(no) / 2.0) ** 0.5
    noise = torch.complex(torch.randn_like(y.real), torch.randn_like(y.real))
    return y + noise * std


def irc_interference_covariance(
    h_int: torch.Tensor | None,
    y: torch.Tensor,
    no: float,
    iot_db: float,
    method: str = "perfect",
) -> torch.Tensor:
    """Spatial covariance of other-cell interference (no thermal noise).

    Returned shape is ``[B, RxAnt, RxAnt]``. For ``iot_db <= 0`` this is zero.
    """
    b = y.shape[0]
    rx_ant = y.shape[2]
    device = y.device
    dtype = y.dtype
    zero = torch.zeros(b, rx_ant, rx_ant, device=device, dtype=dtype)
    if iot_db <= 0.0:
        return zero

    inr_lin = db_to_lin(iot_db)
    int_power = inr_lin * float(no)

    if method == "perfect":
        if h_int is None:
            raise ValueError("perfect IoT covariance requires interferer channels")
        r = spatial_covariance_from_channel(h_int)
        # spatial_covariance_from_channel uses unscaled h_int (unit energy).
        return project_psd(int_power * r)

    r_yy = sample_covariance_from_y(y)
    # Remove the thermal-noise floor; remaining power is serving + IoT.
    eye = torch.eye(rx_ant, device=device, dtype=dtype).expand(b, -1, -1)
    r_minus_n = r_yy - float(no) * eye
    return project_psd(r_minus_n)
