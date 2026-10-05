import numpy as np
import pytest

from nr_ul_sim.paper_ce import DEFAULT_MODEL_DIR
from nr_ul_sim.parameters import CHANNEL_ESTIMATORS
from nr_ul_sim.random_campaign import CELLS, parse_estimators, plan_configs, run_config, start_snr
from nr_ul_sim.random_channels import (
    circular_mean_spread,
    linear_mean_spread,
    randomize_cdl_table,
    sample_cdl_link,
    scale_azimuth,
    scale_zenith,
    wrap180,
)


def test_azimuth_scaling_sets_mean_and_scales_spread():
    rng = np.random.default_rng(0)
    ang = rng.normal(30.0, 8.0, 20)
    pw = rng.uniform(0.1, 1.0, 20)
    _, sp0 = circular_mean_spread(ang, pw)
    new = scale_azimuth(ang, pw, 1.5, -100.0)
    mu, sp = circular_mean_spread(new, pw)
    assert abs(wrap180(mu + 100.0)) < 0.05  # circular mean shifts slightly when scaling
    assert sp == pytest.approx(1.5 * sp0, rel=0.03)


def test_azimuth_scaling_across_wrap():
    ang = np.array([175.0, -175.0, 179.0, -179.0])
    pw = np.ones(4)
    mu, _ = circular_mean_spread(ang, pw)
    assert abs(abs(mu) - 180.0) < 1e-6
    new = scale_azimuth(ang, pw, 1.0, 0.0)
    assert np.all(np.abs(new) < 10.0)


def test_zenith_scaling_is_clipped():
    ang = np.array([60.0, 90.0, 120.0])
    pw = np.ones(3)
    new = scale_zenith(ang, pw, 4.0, 90.0)
    assert new.min() >= 0.0 and new.max() <= 180.0
    mu, _ = linear_mean_spread(scale_zenith(ang, pw, 0.5, 100.0), pw)
    assert mu == pytest.approx(100.0)


def _toy_table(los=False):
    return {
        "los": los, "num_clusters": 4,
        "delays": [0.0, 0.3, 1.1, 2.4],
        "powers": [0.0, -3.0, -6.0, -10.0],
        "aod": [-40.0, 10.0, 60.0, 120.0], "aoa": [100.0, -20.0, 170.0, -150.0],
        "zod": [95.0, 100.0, 92.0, 105.0], "zoa": [70.0, 85.0, 95.0, 110.0],
    }


def test_randomized_table_moves_means_as_requested():
    link = sample_cdl_link(np.random.default_rng(3), "cdl-c")
    rnd = randomize_cdl_table(_toy_table(), link)
    st = rnd["stats"]
    # jitter is a few degrees; the requested mean must be reached closely
    assert abs(wrap180(st["bs_as_mean_deg"] - link["bs_az_mean_deg"])) < 8.0
    assert abs(st["bs_zs_mean_deg"] - link["bs_zen_mean_deg"]) < 4.0
    o = rnd["override"]
    assert len(o["delays"]) == 4 and o["delays"][0] == 0.0
    assert all(d > 0 for d in o["delays"][1:])
    assert sum(rnd["powers_lin"]) == pytest.approx(1.0)


def test_cdl_link_sampling_ranges_and_k_factor():
    rng = np.random.default_rng(1)
    for _ in range(200):
        c = sample_cdl_link(rng, "cdl-c")
        assert 20.0 <= c["ds_ns"] <= 1000.0 and 0.5 <= c["speed_kmh"] <= 120.0
        assert "k_factor_db" not in c
    d = sample_cdl_link(rng, "cdl-d")
    assert 0.0 <= d["k_factor_db"] <= 15.0


def test_plan_is_balanced_and_deterministic():
    rows = plan_configs(120, ["cdl-b", "umi"], seed=7)
    assert rows == plan_configs(120, ["cdl-b", "umi"], seed=7)
    assert [r["id"] for r in rows] == list(range(240))
    for ch in ("cdl-b", "umi"):
        counts = {}
        for r in rows:
            if r["channel"] == ch:
                key = (r["num_ue"], r["rank"], r["modulation"])
                counts[key] = counts.get(key, 0) + 1
        assert set(counts) == set(CELLS) and set(counts.values()) == {10}
    assert len({r["seed"] for r in rows}) == 240


def test_start_snr_inside_range():
    for mod in ("qpsk", "qam16", "qam64"):
        for ue, r in ((1, 1), (2, 2)):
            for iot in (0.0, 20.0):
                assert -16.0 <= start_snr(mod, ue, r, iot) <= 48.0


def test_parse_estimators():
    assert parse_estimators("") == CHANNEL_ESTIMATORS
    assert parse_estimators("a_mmse, ls_lin,perfect,a_mmse") == ("perfect", "a_mmse", "ls_lin")
    with pytest.raises(ValueError, match="nope"):
        parse_estimators("ls_lin,nope")


@pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "a_mmse.pt").exists(), reason="run train_paper_ce.py first")
@pytest.mark.parametrize("channel,num_ue,rank", [("cdl-c", 1, 1), ("uma", 2, 2)])
def test_run_config_with_a_mmse(channel, num_ue, rank):
    row = {"id": 0, "seed": 7, "channel": channel, "num_ue": num_ue, "rank": rank, "modulation": "qpsk"}
    est = parse_estimators("ls_lin,lmmse_data,a_mmse")
    rec, _ = run_config(row, num_prb=8, batch_size=2, max_mc_iter=2, target_bit_errors=50,
                        target_block_errors=4, max_points=3, estimators=est)
    assert set(rec["estimators"]) == set(est)
    for name, r in rec["estimators"].items():
        assert r["status"] in ("ok", "right_censored", "left_censored"), name
        if r["status"] == "ok":
            assert rec["search"]["lo_db"] <= r["wp_db"] <= rec["search"]["hi_db"]
