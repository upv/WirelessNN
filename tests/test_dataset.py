"""Dataset builder tests that do not run a Sionna link simulation."""

import json

import numpy as np
import pytest
import torch

from nr_ul_sim.dataset import (
    FEATURE_NAMES,
    BerProbe,
    DatasetConfig,
    channel_features,
    compact_channel,
    interference_covariance,
    pack_dataset,
    run_scenario,
    sample_scenario,
    save_channel_arrays,
    search_working_point,
)
from nr_ul_sim.parameters import SimConfig


class FakeProbe(BerProbe):
    """BER curve with a waterfall at ``crossing`` dB, no simulator involved."""

    def __init__(self, crossing: float, receivers=("lmmse",)):
        self.crossing = crossing
        self.receivers = receivers
        self.calls = 0
        self.ber = {}

    def __call__(self, snr_db):
        key = round(float(snr_db), 3)
        if key not in self.ber:
            self.calls += 1
            value = 10.0 ** (-2.0 - 0.5 * (key - self.crossing))
            self.ber[key] = {r: min(value, 0.5) for r in self.receivers}
        return self.ber[key]


SEARCH = dict(target=0.01, snr_min=-20.0, snr_max=30.0, coarse_step=4.0, refine_db=0.5)


@pytest.mark.parametrize("crossing", [-11.3, 0.0, 7.7, 21.2])
def test_search_finds_crossing(crossing):
    probe = FakeProbe(crossing)
    wp, status = search_working_point(probe, "lmmse", **SEARCH)
    assert status == "ok"
    assert wp == pytest.approx(crossing, abs=0.5)
    assert probe.calls < 20


def test_search_hint_saves_evaluations():
    cold = FakeProbe(12.0)
    search_working_point(cold, "lmmse", **SEARCH)
    warm = FakeProbe(12.0)
    search_working_point(warm, "lmmse", start_db=11.0, **SEARCH)
    assert warm.calls < cold.calls


def test_search_out_of_range():
    wp, status = search_working_point(FakeProbe(60.0), "lmmse", **SEARCH)
    assert wp is None and status == "not_reached"

    wp, status = search_working_point(FakeProbe(-40.0), "lmmse", **SEARCH)
    assert wp == -20.0 and status == "below_range"


def test_compact_channel_shape_and_values():
    cfg = SimConfig(num_prb=8, fft_size=1024, num_ue=2, num_layers=1)
    dcfg = DatasetConfig(num_freq_bins=16, num_symbol_bins=4)
    h = torch.zeros(1, 1, 4, 2, 1, 14, 1024, dtype=torch.complex64)
    left, _ = cfg.guard_carriers
    h[..., left : left + cfg.num_used_subcarriers] = 2 + 1j
    out = compact_channel(h, cfg, dcfg)
    assert out.shape == (4, 2, 4, 16)
    assert out.dtype == np.complex64
    assert np.allclose(out, 2 + 1j)


def test_interference_covariance_is_normalized():
    h = torch.randn(1, 1, 4, 2, 1, 8, 4, dtype=torch.complex64)
    r = interference_covariance(h)
    assert r.shape == (4, 4)
    assert np.trace(r).real == pytest.approx(4.0, rel=1e-4)
    assert interference_covariance(None) is None


def test_channel_features():
    rng = np.random.default_rng(0)
    h = (rng.normal(size=(4, 2, 4, 16)) + 1j * rng.normal(size=(4, 2, 4, 16))) / np.sqrt(2)
    features = channel_features(h.astype(np.complex64))
    assert set(features) == set(FEATURE_NAMES)
    assert features["gain_db"] == pytest.approx(0.0, abs=1.0)
    assert features["h_rms"] == pytest.approx(1.0, abs=0.2)
    assert features["cond_db"] > 0.0
    assert features["h_rank"] >= 1.0
    assert features["ruu_rank"] == 0.0
    assert 0.0 <= features["freq_corr"] <= 1.0


def test_channel_features_rank_and_ruu():
    h = np.zeros((4, 2, 3, 8), dtype=np.complex64)
    h[0, 0] = 1.0
    h[1, 0] = 0.5
    features = channel_features(h)
    assert features["h_rank"] == pytest.approx(1.0)
    assert features["h_eff_rank"] == pytest.approx(1.0, abs=0.05)
    assert features["h_rms"] > 0.0
    assert features["rms_delay_samp"] >= 0.0

    r_iot = np.zeros((4, 4), dtype=np.complex64)
    r_iot[0, 0] = 1.0
    r_iot[1, 1] = 1e-6
    ruu = channel_features(h, r_iot)
    assert ruu["ruu_rank"] == pytest.approx(1.0)
    assert ruu["ruu_eff_rank"] == pytest.approx(1.0, abs=0.05)
    assert ruu["ruu_dom_frac"] > 0.9
    assert ruu["ruu_cond_db"] > 20.0


def test_sampled_scenarios_are_valid_configs():
    dcfg = DatasetConfig(max_total_layers=8, rank_choices=(1, 2))
    for index in range(30):
        scenario = sample_scenario(np.random.default_rng([0, index]), dcfg)
        cfg = SimConfig(
            channel=scenario["channel"],
            delay_spread_ns=scenario["delay_spread_ns"],
            num_ue=scenario["num_ue"],
            num_layers=scenario["rank"],
            num_rx_ant=scenario["num_rx_ant"],
            speed_kmh=scenario["speed_kmh"],
            num_interferers=scenario["num_interferers"],
        )
        assert cfg.num_ue * cfg.num_layers <= 8
        assert scenario["iot_db"] > 0.0 or scenario["num_interferers"] == 0


def test_scenario_is_reproducible_regardless_of_warm_start():
    """Each SNR point owns its random stream, so the search path cannot bias the label."""
    dcfg = DatasetConfig(
        channels=("cdl-c",), modulations=("qpsk",), receivers=("lmmse",),
        channel_estimators=("ls_lin",),
        num_ue_choices=(1,), rank_choices=(1,), num_prb=4, fft_size=256,
        num_freq_bins=16, batch_size=1, max_mc_iter=2, num_target_bit_errors=20,
        coarse_step=6.0, refine_db=2.0,
    )
    cold, arrays = run_scenario(dcfg, 3)
    warm, _ = run_scenario(dcfg, 3, {"qpsk": 25.0})
    assert arrays["h"].shape == (4, 1, 4, 16)
    assert cold[0]["working_point_db"] == warm[0]["working_point_db"]
    assert warm[0]["num_ber_points"] <= cold[0]["num_ber_points"]

    curves = [dict(zip(r[0]["ber_curve"]["snr_db"], r[0]["ber_curve"]["ber"]["lmmse"]))
              for r in (cold, warm)]
    shared = set(curves[0]) & set(curves[1])
    assert shared
    assert all(curves[0][snr] == curves[1][snr] for snr in shared)


def test_pack_dataset_pads_and_stacks(tmp_path):
    records = []
    for i, streams in enumerate((1, 3)):
        rec = {
            "scenario_id": i,
            "channel": "cdl-c",
            "channel_code": 1,
            "modulation": "qpsk",
            "modulation_code": 0,
            "iot_db": 10.0,
            "num_interferers": 2,
            "num_ue": streams,
            "rank": 1,
            "num_layers_total": streams,
            "num_rx_ant": 4,
            "num_ue_ant": 1,
            "num_prb": 8,
            "mcs_table": 1,
            "mcs_index": 4,
            "num_bits_per_symbol": 2,
            "tb_size": 1416,
            "speed_kmh": 3.0,
            "delay_spread_ns": None if i else 100.0,
            "carrier_frequency_hz": 3.5e9,
            "target_ber": 0.01,
            "target_coderate": 0.3,
            "working_point_db": {"lmmse": None if i else 4.0},
            "working_point_status": {"lmmse": "not_reached" if i else "ok"},
            "features": {name: float(i) for name in FEATURE_NAMES},
            "channel_shape": [4, streams, 4, 16],
        }
        arrays = {
            "h": np.ones((4, streams, 4, 16), dtype=np.complex64) * (i + 1),
            "r_iot": np.eye(4, dtype=np.complex64),
        }
        rec.update(save_channel_arrays(tmp_path, i, arrays, [rec]))
        records.append(rec)
    (tmp_path / "meta.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    assert (tmp_path / "channels" / "000000" / "H.npy").exists()
    assert json.loads((tmp_path / "channels" / "000000" / "meta.json").read_text())["iot_db"] == 10.0

    pack_dataset(tmp_path, verbose=False)
    with np.load(tmp_path / "dataset.npz") as data:
        assert data["H"].shape == (2, 4, 3, 4, 16)
        assert np.all(data["H"][0, :, 1:] == 0)  # padded streams
        assert np.all(data["H"][1] == 2)
        assert data["R_iot"].shape == (2, 4, 4)
        assert np.isnan(data["wp_lmmse"][1])
        assert np.isnan(data["delay_spread_ns"][1])
        assert list(data["wp_status_lmmse"]) == ["ok", "not_reached"]
    assert (tmp_path / "dataset.csv").exists()
