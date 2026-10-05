"""Simulation configuration: the names the simulator understands and ``SimConfig``.

Everything a run depends on is a field of :class:`SimConfig`; the command line
(:mod:`nr_ul_sim.cli`) maps onto it one to one. The tuples at the top are the
single source of truth for the allowed channels, modulations, receivers and
channel estimators — the CLI choices, the validation below and ``--list`` all
read them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


RECEIVERS = ("mr", "lmmse", "ideal_mmse", "zf", "irc")
RECEIVER_ALIASES = {
    "mmse_ideal": "ideal_mmse",
    "lmmse_ideal": "ideal_mmse",
    "idealmmse": "ideal_mmse",
    "perfect_mmse": "ideal_mmse",
    "mmse_perfect": "ideal_mmse",
}
RECEIVER_INFO = {
    "mr": "matched filter (maximum ratio); ignores all interference",
    "lmmse": "MMSE with S = N0 I; other-cell interference treated as white noise",
    "ideal_mmse": "the lmmse combiner with the true channel (genie bound)",
    "zf": "zero forcing among the co-scheduled layers",
    "irc": "MMSE with S = R_iot + N0 I; rejects coloured other-cell interference",
}
CHANNELS = ("cdl-b", "cdl-c", "cdl-d", "umi", "uma")
CHANNEL_INFO = {
    "cdl-b": "3GPP CDL-B, NLoS clusters, default RMS delay spread 100 ns",
    "cdl-c": "3GPP CDL-C, NLoS clusters, default RMS delay spread 300 ns",
    "cdl-d": "3GPP CDL-D, LoS + clusters, default RMS delay spread 100 ns",
    "umi": "3GPP 38.901 urban micro drop; delay spread drawn per drop",
    "uma": "3GPP 38.901 urban macro drop; delay spread drawn per drop",
}
MODULATIONS = ("qpsk", "qam16", "qam64")
# 38.214 table 5.1.3.1-1. QPSK uses index 5 (R = 379/1024): index 4 (R = 0.30)
# selects LDPC base graph 1 for TBs above 3824 bits, and BG1 below rate 1/3
# needs repetition, which Sionna does not implement.
MCS_PRESETS = {
    "qpsk": (1, 5),   # QPSK, coderate ≈ 0.37
    "qam16": (1, 14),  # 16QAM, coderate ≈ 0.54
    "qam64": (1, 20),  # 64QAM, coderate ≈ 0.55
}
CHANNEL_ESTIMATORS = (
    "perfect",
    "ls_nn",
    "ls_lin",
    "ls_lin_time_avg",
    "lmmse_ce",
    "lmmse_exp",
    "ls_hard_window",
    "ls_soft_window",
    "ls_fir",
    "denoise_nn",
    "lmmse_data",
    "lmmse_data_1d",
    "a_mmse",
    "ra_a_mmse",
)
ESTIMATOR_INFO = {
    "perfect": "true channel (genie)",
    "ls_nn": "DMRS least squares, nearest-neighbour interpolation",
    "ls_lin": "DMRS least squares, linear interpolation",
    "ls_lin_time_avg": "DMRS least squares, linear in frequency, averaged over the DMRS symbols",
    "lmmse_ce": "LMMSE with a TDL prior matched to the scenario (genie prior on CDL)",
    "lmmse_exp": "LMMSE with an exponential-PDP prior (robust, realistic)",
    "ls_hard_window": "LS denoised by a rectangular window in the delay domain",
    "ls_soft_window": "LS denoised by per-tap Wiener weights in the delay domain",
    "ls_fir": "LS + static 17-tap frequency FIR (EqDeepRx baseline)",
    "denoise_nn": "EqDeepRx DenoiseNN [trained]",
    "lmmse_data": "2D LMMSE with a covariance measured on training channels [trained]",
    "lmmse_data_1d": "1D (frequency) LMMSE with the measured covariance [trained]",
    "a_mmse": "A-MMSE: attention-learned fixed linear filters [trained]",
    "ra_a_mmse": "rank-adaptive A-MMSE, rank 6 [trained]",
}
# estimators that need trained weights / statistics from ce_model_dir
LEARNED_ESTIMATORS = ("denoise_nn", "lmmse_data", "lmmse_data_1d", "a_mmse", "ra_a_mmse")
CHANNEL_ESTIMATOR_ALIASES = {
    "ls": "ls_lin",
    "lin": "ls_lin",
    "ls_linear": "ls_lin",
    "nn": "ls_nn",
    "nearest": "ls_nn",
    "lin_time_avg": "ls_lin_time_avg",
    "lmmse": "lmmse_ce",
    "lmmse_tdl": "lmmse_ce",
    "lmmse_robust": "lmmse_exp",
    "ideal": "perfect",
    "genie": "perfect",
    "true": "perfect",
    "hard_window": "ls_hard_window",
    "hard": "ls_hard_window",
    "hw": "ls_hard_window",
    "window": "ls_hard_window",
    "soft_window": "ls_soft_window",
    "soft": "ls_soft_window",
    "sw": "ls_soft_window",
}
IOT_COV_METHODS = ("perfect", "estimated", "residual", "incm_oas")
IOT_COV_INFO = {
    "perfect": "true interferer channels (genie)",
    "estimated": "sample covariance of the whole received grid minus N0 I (includes the serving signal)",
    "residual": "covariance of the DMRS residual y - H_hat p minus N0 I, wideband",
    "incm_oas": "DMRS residual covariance per band with OAS shrinkage (EqDeepRx), per subcarrier",
}
TX_POWER_NORMS = ("per_ue", "per_layer")
DEFAULT_DELAY_SPREAD = {
    "cdl-b": 100e-9,
    "cdl-c": 300e-9,
    "cdl-d": 100e-9,
    "umi": 129e-9,
    "uma": 363e-9,
}
RBG_SIZE = 4


def db_to_lin(value_db: float) -> float:
    return 10.0 ** (float(value_db) / 10.0)


def snrdb_to_noise_var(snr_db: float) -> float:
    return 10.0 ** (-float(snr_db) / 10.0)


def as_float_list(values: Iterable[float] | float) -> list[float]:
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(v) for v in values]


def canonicalize_receiver_name(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    return RECEIVER_ALIASES.get(n, n)


def canonicalize_estimator_name(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    return CHANNEL_ESTIMATOR_ALIASES.get(n, n)


def receiver_uses_perfect_csi(name: str, perfect_csi: bool = False) -> bool:
    return bool(perfect_csi) or canonicalize_receiver_name(name) == "ideal_mmse"


def detection_keys(
    receivers: Iterable[str], estimators: Iterable[str]
) -> list[tuple[str, str, str]]:
    """Map a sweep onto ``(result_key, receiver, estimator)``.

    One estimator → keys stay the receiver names (old campaigns). One receiver
    and several estimators → keys are the estimator names (CE study). Both
    vary → ``{receiver}_{estimator}``.
    """
    recs = tuple(canonicalize_receiver_name(r) for r in receivers)
    ests = tuple(canonicalize_estimator_name(e) for e in estimators) or ("ls_lin",)
    if len(ests) == 1:
        return [(r, r, ests[0]) for r in recs]
    if len(recs) == 1:
        return [(e, recs[0], e) for e in ests]
    return [(f"{r}_{e}", r, e) for r in recs for e in ests]


def parse_csv_floats(text: str) -> list[float]:
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
    """One link-level scenario and how to simulate it.

    SNR is per receive antenna and resource element for one UE (the channel has
    unit mean energy, ``N0 = 10^(-SNR/10)``). IoT is the total other-cell
    interference power per antenna relative to ``N0``; ``0`` means none.
    Invalid values raise ``ValueError`` on construction.
    """

    # --- scenario ------------------------------------------------------------
    channel: str = "cdl-c"                 # one of CHANNELS
    delay_spread_ns: float | None = None   # CDL RMS delay spread; None -> DEFAULT_DELAY_SPREAD
    num_ue: int = 1                        # co-scheduled UEs, 1..4
    num_rx_ant: int = 4                    # gNB antennas: 1, 2, 4 or 8
    num_ue_ant: int | None = None          # UE antenna ports 1/2/4; None -> equal to the rank
    num_layers: int = 1                    # rank per UE, 1..2
    speed_kmh: float = 3.0
    # --- sweep ---------------------------------------------------------------
    snr_db: list[float] = field(default_factory=lambda: list(range(-20, 31, 2)))
    iot_db: list[float] = field(default_factory=lambda: [0.0])   # one SNR sweep per value
    num_interferers: int = 2               # interfering UEs sharing the IoT power
    # --- modulation and coding -------------------------------------------------
    modulation: str = "qpsk"               # one of MODULATIONS; selects MCS_PRESETS
    mcs_table: int | None = None           # override the preset table ...
    mcs_index: int | None = None           # ... and index (38.214 5.1.3.1)
    # --- receiver --------------------------------------------------------------
    receivers: tuple[str, ...] = RECEIVERS
    channel_estimators: tuple[str, ...] = ("ls_lin",)
    perfect_csi: bool = False              # True: every receiver gets the true channel
    iot_cov: str = "perfect"               # one of IOT_COV_METHODS (IRC only)
    # "per_ue": every UE radiates unit power in total, split over its layers, so
    # the per-antenna SNR is the SNR of one UE. "per_layer": unit power per layer.
    tx_power_norm: str = "per_ue"
    # Delay-domain window of the ls_hard_window / ls_soft_window estimators
    ce_window_pos_us: float = 3.0
    ce_window_neg_us: float = 1.0
    ce_soft_threshold: float = 1.5
    ce_soft_within_window: bool = False
    ce_time_interp: str = "linear"
    # RMS delay spread of the exponential PDP prior of lmmse_exp [ns]; None -> scenario value
    ce_lmmse_prior_ds_ns: float | None = None
    # trained models of the paper estimators (None -> <repo>/models/paper_ce)
    ce_model_dir: str | None = None
    # INCM estimation band of --iot-cov incm_oas [subcarriers] (EqDeepRx: 2 PRB)
    incm_band_sc: int = 24
    # --- carrier and resource grid ---------------------------------------------
    carrier_frequency: float = 3.5e9       # [Hz]
    subcarrier_spacing_khz: float = 30.0
    num_prb: int = 68
    fft_size: int = 1024
    num_ofdm_symbols: int = 14
    mapping_type: str = "A"
    dmrs_additional_position: int = 1
    dmrs_length: int | None = None
    num_cdm_groups_without_data: int = 2
    # --- Monte Carlo -------------------------------------------------------------
    # An SNR point stops when every curve has both error targets, or at max_mc_iter.
    batch_size: int = 4                    # slots per iteration
    max_mc_iter: int = 50
    num_target_bit_errors: int = 200
    num_target_block_errors: int = 20
    target_ber: float = 0.01               # working point: SNR where BER crosses this
    target_bler: float = 0.1               # BLER working point
    seed: int = 42
    # --- UMi / UMa drops -----------------------------------------------------------
    o2i_model: str = "low"
    enable_pathloss: bool = False
    enable_shadow_fading: bool = False

    def __post_init__(self) -> None:
        self.channel = self.channel.lower()
        self.modulation = self.modulation.lower()
        self.iot_cov = self.iot_cov.lower()
        self.tx_power_norm = self.tx_power_norm.lower()
        self.ce_time_interp = self.ce_time_interp.lower()
        self.receivers = tuple(canonicalize_receiver_name(r) for r in self.receivers)
        self.channel_estimators = tuple(
            canonicalize_estimator_name(e) for e in self.channel_estimators
        ) or ("ls_lin",)
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
        if self.iot_cov not in IOT_COV_METHODS:
            raise ValueError(f"iot_cov must be one of {IOT_COV_METHODS}")
        if self.tx_power_norm not in TX_POWER_NORMS:
            raise ValueError(f"tx_power_norm must be one of {TX_POWER_NORMS}")
        if self.ce_time_interp not in ("linear", "avg"):
            raise ValueError("ce_time_interp must be 'linear' or 'avg'")
        if self.ce_window_pos_us < 0 or self.ce_window_neg_us < 0:
            raise ValueError("CE window lengths must be non-negative")
        unknown = [r for r in self.receivers if r not in RECEIVERS]
        if unknown:
            raise ValueError(f"Unknown receivers {unknown}; choose from {RECEIVERS}")
        unknown_ce = [e for e in self.channel_estimators if e not in CHANNEL_ESTIMATORS]
        if unknown_ce:
            raise ValueError(
                f"Unknown channel estimators {unknown_ce}; choose from {CHANNEL_ESTIMATORS}"
            )
        if self.num_prb * 12 > self.fft_size:
            raise ValueError(
                f"{self.num_prb} PRBs need {self.num_prb * 12} subcarriers, "
                f"larger than FFT size {self.fft_size}"
            )
        max_ports = 8 if self.resolved_dmrs_length == 2 else 4
        if self.num_ue * self.num_layers > max_ports:
            raise ValueError(
                f"Total layers {self.num_ue * self.num_layers} exceed "
                f"DMRS Type-1 ports ({max_ports})."
            )

    @property
    def num_used_subcarriers(self) -> int:
        return int(self.num_prb) * 12

    @property
    def num_rbg(self) -> int:
        return int(self.num_prb) // RBG_SIZE

    @property
    def guard_carriers(self) -> tuple[int, int]:
        unused = self.fft_size - self.num_used_subcarriers
        left = unused // 2
        return left, unused - left

    @property
    def speed_mps(self) -> float:
        return float(self.speed_kmh) / 3.6

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
        table, index = MCS_PRESETS[self.modulation]
        if self.mcs_table is not None:
            table = self.mcs_table
        if self.mcs_index is not None:
            index = self.mcs_index
        return int(table), int(index)

    @property
    def subcarrier_spacing(self) -> float:
        return float(self.subcarrier_spacing_khz) * 1e3

    @property
    def layer_power_scale(self) -> float:
        """Amplitude scale applied to every layer of a UE (unit total UE power)."""
        if self.tx_power_norm == "per_ue":
            return 1.0 / float(self.num_layers) ** 0.5
        return 1.0

    @property
    def num_interferer_streams(self) -> int:
        return int(self.num_interferers) * int(self.resolved_ue_ant)

    def summary(self) -> dict[str, Any]:
        """Resolved configuration, stored as ``campaign["config"]`` in every result."""
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
            "channel_estimators": list(self.channel_estimators),
            "channel_estimator": (
                "perfect" if self.perfect_csi else ",".join(self.channel_estimators)
            ),
            "iot_cov": self.iot_cov,
            "tx_power_norm": self.tx_power_norm,
            "ce_window_pos_us": self.ce_window_pos_us,
            "ce_window_neg_us": self.ce_window_neg_us,
            "ce_soft_threshold": self.ce_soft_threshold,
            "ce_soft_within_window": self.ce_soft_within_window,
            "ce_time_interp": self.ce_time_interp,
            "ce_lmmse_prior_ds_ns": self.ce_lmmse_prior_ds_ns,
            "ce_model_dir": self.ce_model_dir,
            "incm_band_sc": self.incm_band_sc,
            "carrier_frequency_hz": self.carrier_frequency,
            "subcarrier_spacing_khz": self.subcarrier_spacing_khz,
            "num_prb": self.num_prb,
            "num_used_subcarriers": self.num_used_subcarriers,
            "fft_size": self.fft_size,
            "guard_carriers": [left, right],
            "rbg_size": RBG_SIZE,
            "num_rbg": self.num_rbg,
            "dmrs_type": 1,
            "dmrs_length": self.resolved_dmrs_length,
            "dmrs_additional_position": self.dmrs_additional_position,
            "target_ber": self.target_ber,
            "target_bler": self.target_bler,
            "seed": self.seed,
        }
