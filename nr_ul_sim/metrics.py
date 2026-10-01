from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ErrorStats:
    bit_errors: int = 0
    num_bits: int = 0
    block_errors: int = 0
    num_blocks: int = 0

    def update(self, b, b_hat) -> None:
        diff = b != b_hat
        self.bit_errors += int(diff.sum())
        self.num_bits += int(diff.size)
        block_err = diff.reshape(diff.shape[0] * diff.shape[1], -1).any(axis=1)
        self.block_errors += int(block_err.sum())
        self.num_blocks += int(block_err.size)

    @property
    def ber(self) -> float:
        return 1.0 if self.num_bits == 0 else self.bit_errors / self.num_bits

    @property
    def bler(self) -> float:
        return 1.0 if self.num_blocks == 0 else self.block_errors / self.num_blocks

    def as_dict(self) -> dict:
        return {
            "ber": self.ber,
            "bler": self.bler,
            "bit_errors": self.bit_errors,
            "num_bits": self.num_bits,
            "block_errors": self.block_errors,
            "num_blocks": self.num_blocks,
        }


@dataclass
class SweepResult:
    snr_db: list[float] = field(default_factory=list)
    ber: list[float] = field(default_factory=list)
    bler: list[float] = field(default_factory=list)
    stats: list[dict] = field(default_factory=list)
    working_point_db: float | None = None
    working_point_bler_db: float | None = None

    def as_dict(self) -> dict:
        return {
            "snr_db": self.snr_db,
            "ber": self.ber,
            "bler": self.bler,
            "stats": self.stats,
            "working_point_db": self.working_point_db,
            "working_point_bler_db": self.working_point_bler_db,
        }


def working_point_snr(snr_db, ber, target: float = 0.01) -> float | None:
    """SNR where a BER or BLER curve crosses ``target`` (log interpolation).

    None if never crossed. Works for any monotone-in-expectation error rate.
    """
    snr = np.asarray(snr_db, dtype=float)
    ber = np.clip(np.asarray(ber, dtype=float), 1e-12, 1.0)
    if snr.size == 0:
        return None
    order = np.argsort(snr)
    snr, ber = snr[order], ber[order]
    if np.all(ber > target):
        return None
    if np.all(ber <= target):
        return float(snr[0])
    log_t = np.log10(target)
    log_ber = np.log10(ber)
    for i in range(len(snr) - 1):
        b0, b1 = log_ber[i], log_ber[i + 1]
        if (b0 - log_t) * (b1 - log_t) <= 0:
            if b1 == b0:
                return float(snr[i])
            t = (log_t - b0) / (b1 - b0)
            return float(snr[i] + t * (snr[i + 1] - snr[i]))
    return float(snr[np.argmin(np.abs(ber - target))])
