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
    x = torch.randn(
        batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size,
        device=device, dtype=dtype,
    )
    x = torch.complex(x, torch.randn_like(x)) / (2.0 ** 0.5)
    left, right = guard_carriers
    if left:
        x[..., :left] = 0
    if right:
        x[..., fft_size - right :] = 0
    return x


def spatial_covariance_from_channel(h: torch.Tensor) -> torch.Tensor:
    """E[H H^H] averaged over streams and REs. h: [B, Rx, M, Tx, TxAnt, Sym, Sc]."""
    b, _, rx_ant, num_tx, tx_ant, num_sym, num_sc = h.shape
    h_re = h[:, 0].reshape(b, rx_ant, num_tx * tx_ant, num_sym * num_sc)
    h_re = h_re.permute(0, 3, 1, 2)
    return torch.matmul(h_re, h_re.conj().transpose(-1, -2)).mean(dim=1)


def sample_covariance_from_y(y: torch.Tensor) -> torch.Tensor:
    b, _, rx_ant, num_sym, num_sc = y.shape
    y_mat = y[:, 0].permute(0, 2, 3, 1).reshape(b, num_sym * num_sc, rx_ant)
    return torch.matmul(y_mat.transpose(-1, -2).conj(), y_mat) / (num_sym * num_sc)


def project_psd(r: torch.Tensor, min_eig: float = 1e-8) -> torch.Tensor:
    r_h = 0.5 * (r + r.transpose(-1, -2).conj())
    eigvals, eigvecs = torch.linalg.eigh(r_h)
    eigvals = torch.clamp(eigvals.real, min=min_eig).to(eigvals.dtype)
    return eigvecs @ torch.diag_embed(eigvals.to(eigvecs.dtype)) @ eigvecs.transpose(-1, -2).conj()


def serving_plus_interference(y_s, y_i, no: float, iot_db: float):
    """y = y_s + sqrt(INR * N0) y_i + w. IoT 0 dB → no interferer."""
    y = y_s
    if y_i is not None and iot_db > 0.0:
        y = y + ((db_to_lin(iot_db) * float(no)) ** 0.5) * y_i
    noise = torch.complex(torch.randn_like(y.real), torch.randn_like(y.real))
    return y + noise * ((float(no) / 2.0) ** 0.5)


def irc_interference_covariance(h_int, y, no: float, iot_db: float, method: str = "perfect"):
    b, rx_ant = y.shape[0], y.shape[2]
    if iot_db <= 0.0:
        return torch.zeros(b, rx_ant, rx_ant, device=y.device, dtype=y.dtype)
    if method == "perfect":
        if h_int is None:
            raise ValueError("perfect IoT covariance requires interferer channels")
        return project_psd(db_to_lin(iot_db) * float(no) * spatial_covariance_from_channel(h_int))
    eye = torch.eye(rx_ant, device=y.device, dtype=y.dtype).expand(b, -1, -1)
    return project_psd(sample_covariance_from_y(y) - float(no) * eye)
