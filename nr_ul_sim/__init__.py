"""5G NR PUSCH uplink link-level simulator built on NVIDIA Sionna.

    from nr_ul_sim import NRUplinkSimulator, SimConfig
    campaign = NRUplinkSimulator(SimConfig(channel="cdl-c", modulation="qam16")).run()

Modules, in the order a slot flows through them:

    parameters          SimConfig and the names of channels, receivers, estimators
    pusch               PUSCH transmitter on the FFT grid
    channels            CDL links and UMi/UMa drops
    interference        other-cell interference, noise, IRC covariance
    channel_estimation  estimator factory (LS, LMMSE)
      windowed_ce         delay-domain windowed LS
      paper_ce            EqDeepRx / A-MMSE estimators (trained)
    receivers           MR / ZF / L-MMSE / IRC detection and TB decoding
    simulator           NRUplinkSimulator: one slot, one SNR point, a full sweep
    metrics, plotting   error statistics, working point, figures
    cli                 python -m nr_ul_sim

Built on top of the simulator:

    dataset, ridge                 channel tensor -> working point dataset and baseline
    random_channels, random_campaign, wp_search
                                   random-channel campaign with censored WP search

docs/TUTORIAL.md and docs/ARCHITECTURE.md describe all of it.
"""

from .metrics import working_point_snr
from .parameters import CHANNEL_ESTIMATORS, CHANNELS, MODULATIONS, RECEIVERS, SimConfig
from .simulator import NRUplinkSimulator

__all__ = [
    "NRUplinkSimulator",
    "SimConfig",
    "working_point_snr",
    "CHANNELS",
    "MODULATIONS",
    "RECEIVERS",
    "CHANNEL_ESTIMATORS",
]
