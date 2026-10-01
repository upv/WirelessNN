"""CLI parsing, BLER working point and config additions (no Sionna)."""

import numpy as np
import pytest

from nr_ul_sim.cli import build_parser, config_from_args, join_signed_values
from nr_ul_sim.metrics import working_point_snr
from nr_ul_sim.parameters import CHANNEL_ESTIMATORS, SimConfig


def test_negative_range_argument_is_accepted():
    argv = join_signed_values(["--snr-db", "-20:30:2", "--iot-db", "0,10", "--speed-kmh", "-3"])
    assert argv == ["--snr-db=-20:30:2", "--iot-db", "0,10", "--speed-kmh=-3"]
    args = build_parser().parse_args(join_signed_values(["--snr-db", "-20:0:10"]))
    cfg = config_from_args(args)
    assert cfg.snr_db == [-20.0, -10.0, 0.0]
    assert cfg.num_prb == 68


def test_bler_working_point():
    snr = [0.0, 2.0, 4.0, 6.0]
    bler = [1.0, 0.6, 0.05, 0.0]
    wp = working_point_snr(snr, bler, target=0.1)
    assert 2.0 < wp < 4.0
    assert working_point_snr(snr, [1, 1, 0.9, 0.5], 0.1) is None


def test_new_estimators_and_aliases():
    cfg = SimConfig(channel_estimators=("hard", "sw", "lmmse", "lmmse_robust"))
    assert cfg.channel_estimators == ("ls_hard_window", "ls_soft_window", "lmmse_ce", "lmmse_exp")
    assert set(("ls_hard_window", "ls_soft_window")) <= set(CHANNEL_ESTIMATORS)
    with pytest.raises(ValueError):
        SimConfig(iot_cov="magic")
    with pytest.raises(ValueError):
        SimConfig(ce_window_pos_us=-1.0)


def test_power_normalisation_scale():
    assert SimConfig(num_layers=1).layer_power_scale == 1.0
    assert SimConfig(num_layers=2).layer_power_scale == pytest.approx(1 / np.sqrt(2))
    assert SimConfig(num_layers=2, tx_power_norm="per_layer").layer_power_scale == 1.0
    assert SimConfig(num_interferers=3, num_layers=2).num_interferer_streams == 6


def test_qpsk_preset_is_bg1_safe():
    assert SimConfig(modulation="qpsk").resolved_mcs == (1, 5)
    assert SimConfig(modulation="qpsk", mcs_index=4).resolved_mcs == (1, 4)
