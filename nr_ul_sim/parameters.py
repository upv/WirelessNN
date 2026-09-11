"""Scenario parameters for NR PUSCH uplink simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


RECEIVERS = ("mr", "lmmse", "zf", "irc")
CHANNELS = ("cdl-b", "cdl-c", "umi", "uma")
MODULATIONS = ("qpsk", "qam16")

# TS 38.214 MCS Table 1
MCS_PRESETS = {
    "qpsk": {"mcs_table": 1, "mcs_index": 4, "num_bits_per_symbol": 2},
    "qam16": {"mcs_table": 1, "mcs_index": 14, "num_bits_per_symbol": 4},
}

# 3GPP TR 38.901 suggested RMS delay-spread values [s]
DEFAULT_DELAY_SPREAD = {
    "cdl-b": 100e-9,  # nominal
    "cdl-c": 300e-9,  # long / typical for CDL-C
    "umi": 129e-9,  # UMi street-canyon, ~3.5 GHz, normal DS (documentation only)
    "uma": 363e-9,  # UMa, ~3.5 GHz, normal DS (documentation only)
}

# 68 PRB BWP, configuration 1 → RBG size 4 → 17 RBGs (TS 38.214)
RBG_SIZE_CONFIG1 = 4


def kmh_to_mps(speed_kmh: float) -> float:
    """Convert kilometres per hour to metres per second."""
    return float(speed_kmh) / 3.6


def db_to_lin(value_db: float) -> float:
    """Convert a dB value to linear scale."""
    return 10.0 ** (float(value_db) / 10.0)


def lin_to_db(value: float) -> float:
    """Convert a linear value to dB."""
    import math

    return 10.0 * math.log10(max(float(value), 1e-30))


def snrdb_to_noise_var(snr_db: float) -> float:
    """Per-antenna / per-RE noise variance for a normalized channel."""
    return 10.0 ** (-float(snr_db) / 10.0)


def as_float_list(values: Iterable[float] | float) -> list[float]:
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(v) for v in values]


def parse_csv_floats(text: str) -> list[float]:
    """Parse a comma-separated list, a start:stop:step range, or a single float."""
    text = text.strip()
    if ":" in text and "," not in text:
        parts = [float(p) for p in text.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1.0
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Invalid range '{text}'. Use start:stop:step.")
        if step == 0:
            raise ValueError("Range step must be non-zero.")
        values = []
        x = start
        if step > 0:
            while x <= stop + 1e-9:
                values.append(round(x, 10))
                x += step
        else:
            while x >= stop - 1e-9:
                values.append(round(x, 10))
                x += step
        return values
    return [float(p.strip()) for p in text.split(",") if p.strip()]


@dataclass
class SimConfig:
    """Complete configuration of one NR uplink campaign or a single drop."""

    channel: str = "cdl-c"
    delay_spread_ns: float | None = None
    num_ue: int = 1
    num_rx_ant: int = 4
    num_ue_ant: int | None = None
    num_layers: int = 1
    speed_kmh: float = 3.0
    snr_db: list[float] = field(default_factory=lambda: list(range(-20, 21, 2)))
    iot_db: list[float] = field(default_factory=lambda: [0.0])
    num_interferers: int = 2
    modulation: str = "qpsk"
    mcs_table: int | None = None
    mcs_index: int | None = None
    receivers: tuple[str, ...] = RECEIVERS
    perfect_csi: bool = False
    iot_cov: str = "perfect"
    carrier_frequency: float = 3.5e9
    subcarrier_spacing_khz: float = 30.0
    num_prb: int = 68
    fft_size: int = 1024
    num_ofdm_symbols: int = 14
    mapping_type: str = "A"
    dmrs_type: int = 1
    dmrs_additional_position: int = 1
    dmrs_length: int | None = None
    num_cdm_groups_without_data: int = 2
    batch_size: int = 4
    max_mc_iter: int = 50
    num_target_bit_errors: int = 200
    target_ber: float = 0.01
    seed: int = 42
    domain: str = "freq"
    o2i_model: str = "low"
    enable_pathloss: bool = False
    enable_shadow_fading: bool = False

    def __post_init__(self) -> None:
        self.channel = self.channel.lower()
        self.modulation = self.modulation.lower()
        self.iot_cov = self.iot_cov.lower()
        self.domain = self.domain.lower()
        self.receivers = tuple(r.lower() for r in self.receivers)
        self.snr_db = as_float_list(self.snr_db)
        self.iot_db = as_float_list(self.iot_db)

        if self.channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {self.channel}")
        if self.modulation not in MODULATIONS:
            raise ValueError(f"modulation must be one of {MODULATIONS}")
        if not 1 <= self.num_ue <= 4:
            raise ValueError("num_ue must be in 1..4")
        if not 1 <= self.num_layers <= 2:
            raise ValueError("num_layers (rank) must be 1 or 2")
        if self.num_rx_ant < 1:
            raise ValueError("num_rx_ant must be positive")
        if self.dmrs_type != 1:
            raise ValueError("Only DMRS Type-1 is supported in this platform")
        if self.domain != "freq":
            raise ValueError("Only frequency-domain simulations are supported")
        if self.iot_cov not in ("perfect", "estimated"):
            raise ValueError("iot_cov must be 'perfect' or 'estimated'")
        unknown = [r for r in self.receivers if r not in RECEIVERS]
        if unknown:
            raise ValueError(f"Unknown receivers {unknown}; choose from {RECEIVERS}")
        if self.num_prb * 12 > self.fft_size:
            raise ValueError(
                f"{self.num_prb} PRBs need {self.num_prb * 12} subcarriers, "
                f"larger than FFT size {self.fft_size}"
            )
        total_layers = self.num_ue * self.num_layers
        max_ports = 8 if self.resolved_dmrs_length == 2 else 4
        if total_layers > max_ports:
            raise ValueError(
                f"Total layers {total_layers} exceed DMRS Type-1 ports ({max_ports}). "
                "Reduce users/rank or use dmrs_length=2."
            )

    @property
    def num_used_subcarriers(self) -> int:
        return int(self.num_prb) * 12

    @property
    def num_rbg(self) -> int:
        return int(self.num_prb) // RBG_SIZE_CONFIG1

    @property
    def rbg_size(self) -> int:
        return RBG_SIZE_CONFIG1

    @property
    def guard_carriers(self) -> tuple[int, int]:
        unused = self.fft_size - self.num_used_subcarriers
        left = unused // 2
        return left, unused - left

    @property
    def speed_mps(self) -> float:
        return kmh_to_mps(self.speed_kmh)

    @property
    def delay_spread(self) -> float:
        if self.delay_spread_ns is not None:
            return float(self.delay_spread_ns) * 1e-9
        return DEFAULT_DELAY_SPREAD[self.channel]

    @property
    def resolved_ue_ant(self) -> int:
        if self.num_ue_ant is not None:
            return int(self.num_ue_ant)
        return 2 if self.num_layers == 2 else 1

    @property
    def resolved_dmrs_length(self) -> int:
        if self.dmrs_length is not None:
            return int(self.dmrs_length)
        return 2 if self.num_ue * self.num_layers > 4 else 1

    @property
    def resolved_mcs(self) -> tuple[int, int]:
        preset = MCS_PRESETS[self.modulation]
        table = self.mcs_table if self.mcs_table is not None else preset["mcs_table"]
        index = self.mcs_index if self.mcs_index is not None else preset["mcs_index"]
        return int(table), int(index)

    @property
    def subcarrier_spacing(self) -> float:
        return float(self.subcarrier_spacing_khz) * 1e3

    @property
    def sampling_rate(self) -> float:
        return self.fft_size * self.subcarrier_spacing

    def summary(self) -> dict[str, Any]:
        table, index = self.resolved_mcs
        left, right = self.guard_carriers
        return {
            "channel": self.channel,
            "delay_spread_ns": self.delay_spread * 1e9,
            "num_ue": self.num_ue,
            "num_rx_ant": self.num_rx_ant,
            "num_ue_ant": self.resolved_ue_ant,
            "rank": self.num_layers,
            "speed_kmh": self.speed_kmh,
            "snr_db": self.snr_db,
            "iot_db": self.iot_db,
            "num_interferers": self.num_interferers,
            "modulation": self.modulation,
            "mcs_table": table,
            "mcs_index": index,
            "receivers": list(self.receivers),
            "perfect_csi": self.perfect_csi,
            "iot_cov": self.iot_cov,
            "carrier_frequency_hz": self.carrier_frequency,
            "subcarrier_spacing_khz": self.subcarrier_spacing_khz,
            "num_prb": self.num_prb,
            "num_used_subcarriers": self.num_used_subcarriers,
            "fft_size": self.fft_size,
            "guard_carriers": [left, right],
            "rbg_size": self.rbg_size,
            "num_rbg": self.num_rbg,
            "dmrs_type": self.dmrs_type,
            "dmrs_length": self.resolved_dmrs_length,
            "dmrs_additional_position": self.dmrs_additional_position,
            "target_ber": self.target_ber,
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["summary"] = self.summary()
        return data


def validate_receiver_list(receivers: Sequence[str]) -> tuple[str, ...]:
    names = tuple(r.lower() for r in receivers)
    unknown = [r for r in names if r not in RECEIVERS]
    if unknown:
        raise ValueError(f"Unknown receivers {unknown}; choose from {RECEIVERS}")
    return names
