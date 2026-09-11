"""Unit tests that do not require Sionna."""

import math

import numpy as np

from nr_ul_sim.metrics import working_point_snr
from nr_ul_sim.parameters import SimConfig, parse_csv_floats, snrdb_to_noise_var


def test_parse_range():
    values = parse_csv_floats("-20:20:10")
    assert values == [-20.0, -10.0, 0.0, 10.0, 20.0]


def test_parse_csv():
    assert parse_csv_floats("0,10,20") == [0.0, 10.0, 20.0]


def test_snr_to_no():
    assert math.isclose(snrdb_to_noise_var(0.0), 1.0)
    assert math.isclose(snrdb_to_noise_var(20.0), 0.01)


def test_grid_geometry():
    cfg = SimConfig(num_prb=68, fft_size=1024, num_ue=4, num_layers=2)
    assert cfg.num_used_subcarriers == 816
    assert cfg.guard_carriers == (104, 104)
    assert cfg.num_rbg == 17
    assert cfg.resolved_dmrs_length == 2  # 8 layers need Type-1 length 2


def test_working_point_interpolation():
    snr = np.array([-4.0, 0.0, 4.0, 8.0])
    ber = np.array([0.2, 0.05, 0.005, 0.0004])
    wp = working_point_snr(snr, ber, target=0.01)
    assert wp is not None
    assert 0.0 < wp < 4.0


def test_working_point_not_reached():
    snr = np.array([0.0, 5.0])
    ber = np.array([0.2, 0.05])
    assert working_point_snr(snr, ber, target=0.01) is None
