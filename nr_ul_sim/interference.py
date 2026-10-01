from __future__ import annotations

import torch

from .parameters import db_to_lin
from .pusch import strip_guard_subcarriers


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
    """E[H H^H] averaged over REs, summed over streams. h: [B, Rx, M, Tx, TxAnt, Sym, Sc]."""
    b, _, rx_ant, num_tx, tx_ant, num_sym, num_sc = h.shape
    h_re = h[:, 0].reshape(b, rx_ant, num_tx * tx_ant, num_sym * num_sc)
    h_re = h_re.permute(0, 3, 1, 2)
    return torch.matmul(h_re, h_re.conj().transpose(-1, -2)).mean(dim=1)


def sample_covariance(samples: torch.Tensor) -> torch.Tensor:
    """R = mean_n s_n s_n^H for samples [B, N, M] -> [B, M, M]."""
    n = samples.shape[1]
    return torch.einsum("bnm,bnk->bmk", samples, samples.conj()) / n


def sample_covariance_from_y(y: torch.Tensor) -> torch.Tensor:
    """R_yy over every RE of the grid. y: [B, Rx, M, Sym, Sc]."""
    b, _, rx_ant, num_sym, num_sc = y.shape
    y_mat = y[:, 0].permute(0, 2, 3, 1).reshape(b, num_sym * num_sc, rx_ant)
    return sample_covariance(y_mat)


def residual_covariance_from_dmrs(
    y: torch.Tensor,
    h_hat: torch.Tensor,
    pilot_grid: torch.Tensor,
    dmrs_syms: list[int],
    guard_carriers: tuple[int, int],
) -> torch.Tensor:
    """Covariance of y - H_hat p on the DMRS resource elements.

    With two CDM groups without data every RE of a DMRS symbol is either a
    pilot or empty, so the residual contains only other-cell interference,
    thermal noise and channel-estimation error.
    y: [B, Rx, M, Sym, FFT]; h_hat: [B, Rx, M, Tx, S, Sym, Sc]; pilot_grid: [Tx, S, Sym, Sc].
    """
    y_eff = strip_guard_subcarriers(y, guard_carriers)[:, :, :, dmrs_syms, :]
    p = pilot_grid[:, :, dmrs_syms, :].to(device=y.device, dtype=y.dtype)
    h = h_hat[:, :, :, :, :, dmrs_syms, :]
    y_hat = torch.einsum("brmtsdn,tsdn->brmdn", h, p)
    e = y_eff - y_hat
    b, _, rx_ant, num_d, num_sc = e.shape
    e = e[:, 0].permute(0, 2, 3, 1).reshape(b, num_d * num_sc, rx_ant)
    return sample_covariance(e)


def project_psd(r: torch.Tensor, min_eig: float = 1e-8) -> torch.Tensor:
    r_h = 0.5 * (r + r.transpose(-1, -2).conj())
    eigvals, eigvecs = torch.linalg.eigh(r_h)
    eigvals = torch.clamp(eigvals.real, min=min_eig).to(eigvals.dtype)
    return eigvecs @ torch.diag_embed(eigvals.to(eigvecs.dtype)) @ eigvecs.transpose(-1, -2).conj()


def interference_scale(no: float, iot_db: float, num_streams: int) -> float:
    """Amplitude of the interferer grid so that the total INR per antenna is ``iot_db``."""
    return (db_to_lin(iot_db) * float(no) / max(int(num_streams), 1)) ** 0.5


def serving_plus_interference(y_s, y_i, no: float, iot_db: float, num_streams: int = 1):
    """y = y_s + sqrt(INR * N0 / num_streams) y_i + w. IoT 0 dB → no interferer.

    Every interferer stream has unit symbol power and a unit-energy channel, so
    dividing by the number of interferer streams makes the *total* interference
    power per receive antenna equal to ``INR * N0``.
    """
    y = y_s
    if y_i is not None and iot_db > 0.0:
        y = y + interference_scale(no, iot_db, num_streams) * y_i
    noise = torch.complex(torch.randn_like(y.real), torch.randn_like(y.real))
    return y + noise * ((float(no) / 2.0) ** 0.5)


def irc_interference_covariance(
    h_int,
    y,
    no: float,
    iot_db: float,
    method: str = "perfect",
    *,
    h_hat=None,
    pilot_grid=None,
    dmrs_syms=None,
    guard_carriers=(0, 0),
):
    """Other-cell interference covariance R_iot used by the IRC receiver.

    perfect   -- INR * N0 * E[H_int H_int^H] / num_streams from the true interferer channels
    estimated -- sample covariance of the whole received grid minus N0 I
                 (contains the serving users' signal, kept for backwards compatibility)
    residual  -- covariance of the DMRS residual y - H_hat p minus N0 I
    """
    b, rx_ant = y.shape[0], y.shape[2]
    if iot_db <= 0.0 and method == "perfect":
        return torch.zeros(b, rx_ant, rx_ant, device=y.device, dtype=y.dtype)
    eye = torch.eye(rx_ant, device=y.device, dtype=y.dtype).expand(b, -1, -1)
    if method == "perfect":
        if h_int is None:
            raise ValueError("perfect IoT covariance requires interferer channels")
        num_streams = h_int.shape[3] * h_int.shape[4]
        scale = interference_scale(no, iot_db, num_streams) ** 2
        return project_psd(scale * spatial_covariance_from_channel(h_int))
    if method == "estimated":
        return project_psd(sample_covariance_from_y(y) - float(no) * eye)
    if method == "residual":
        if h_hat is None or pilot_grid is None or dmrs_syms is None:
            raise ValueError("residual IoT covariance needs h_hat, pilot_grid and dmrs_syms")
        r = residual_covariance_from_dmrs(y, h_hat, pilot_grid, dmrs_syms, guard_carriers)
        return project_psd(r - float(no) * eye)
    raise ValueError(f"unknown IoT covariance method {method!r}")
