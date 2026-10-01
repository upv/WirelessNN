import math

import pytest

from nr_ul_sim.wp_search import classify, is_floor, search_working_point

NUM_BITS = 100_000


def waterfall(wp, slope_db=1.5, floor=0.0):
    """Smooth BER curve that crosses 1e-2 at ``wp``."""
    def ber(s):
        b = 0.5 / (1.0 + 49.0 * 10 ** ((s - wp) / slope_db))
        return max(b, floor)
    return ber


def make_measure(curves, calls=None):
    def measure(s):
        if calls is not None:
            calls.append(s)
        out = {}
        for name, f in curves.items():
            b = f(s)
            errs = int(round(b * NUM_BITS))
            out[name] = {"ber": errs / NUM_BITS, "bler": min(1.0, 10 * b), "bit_errors": errs,
                         "num_bits": NUM_BITS, "block_errors": 0, "num_blocks": 64}
        return out
    return measure


def test_brackets_and_interpolates_every_estimator():
    curves = {"perfect": waterfall(5.0), "ls_lin": waterfall(7.3), "ls_nn": waterfall(8.1)}
    calls = []
    res = search_working_point(make_measure(curves, calls), list(curves),
                               lo=-10, hi=40, start=0.0, max_points=12)
    for name, true_wp in (("perfect", 5.0), ("ls_lin", 7.3), ("ls_nn", 8.1)):
        r = res["estimators"][name]
        assert r["status"] == "ok"
        assert r["wp_upper_db"] - r["wp_lower_db"] <= 2.0 + 1e-9
        assert r["wp_lower_db"] < r["wp_db"] <= r["wp_upper_db"]
        assert abs(r["wp_db"] - true_wp) < 0.6
    assert len(calls) == len(set(calls)) <= 12


def test_right_censored_at_upper_limit():
    curves = {"perfect": waterfall(10.0), "bad": waterfall(60.0)}
    res = search_working_point(make_measure(curves), list(curves), lo=-10, hi=30, start=8.0,
                               max_points=15, max_excess_db=100.0)
    bad = res["estimators"]["bad"]
    assert bad["status"] == "right_censored"
    assert bad["reason"] == "hi_limit"
    assert bad["wp_lower_db"] == 30.0 and bad["wp_upper_db"] is None
    assert res["estimators"]["perfect"]["status"] == "ok"


def test_error_floor_stops_climbing():
    curves = {"perfect": waterfall(5.0), "floored": waterfall(5.0, floor=0.03)}
    res = search_working_point(make_measure(curves), list(curves), lo=-10, hi=45, start=4.0,
                               max_points=15)
    fl = res["estimators"]["floored"]
    assert fl["status"] == "right_censored"
    assert fl["reason"] == "floor"
    assert max(res["snr_db"]) < 45.0


def test_left_censored_at_lower_limit():
    curves = {"easy": waterfall(-30.0)}
    res = search_working_point(make_measure(curves), ["easy"], lo=-12, hi=30, start=5.0,
                               max_points=15)
    r = res["estimators"]["easy"]
    assert r["status"] == "left_censored" and r["reason"] == "lo_limit"
    assert r["wp_upper_db"] == -12.0 and r["wp_lower_db"] is None


def test_budget_censoring_reports_bound():
    curves = {"far": waterfall(35.0)}
    res = search_working_point(make_measure(curves), ["far"], lo=-10, hi=48, start=0.0,
                               max_points=3)
    r = res["estimators"]["far"]
    assert res["num_points"] == 3
    assert r["status"] == "right_censored" and r["reason"] == "budget"
    assert r["wp_lower_db"] == max(res["snr_db"])


def test_nonmonotone_takes_highest_crossing():
    pts = [(0.0, {"ber": 0.2}), (2.0, {"ber": 0.005}), (4.0, {"ber": 0.02}), (6.0, {"ber": 0.001})]
    c = classify(pts, 1e-2)
    assert c["kind"] == "bracket" and (c["a"], c["b"]) == (4.0, 6.0) and c["nonmonotone"]


def test_floor_needs_tail_and_flat_slope():
    steep = [(10.0, {"ber": 0.04}), (14.0, {"ber": 0.012})]
    flat = [(10.0, {"ber": 0.03}), (14.0, {"ber": 0.028})]
    plateau_high = [(0.0, {"ber": 0.3}), (4.0, {"ber": 0.28})]
    assert not is_floor(steep, 1e-2)
    assert is_floor(flat, 1e-2)
    assert not is_floor(plateau_high, 1e-2)


def test_zero_error_point_interpolates_finitely():
    curves = {"x": lambda s: 0.3 if s < 5 else 0.0}
    res = search_working_point(make_measure(curves), ["x"], lo=-10, hi=30, start=4.0, max_points=8)
    r = res["estimators"]["x"]
    assert r["status"] == "ok" and math.isfinite(r["wp_db"])


def test_flat_floor_at_high_ber_stops_when_reference_is_below():
    curves = {"perfect": waterfall(13.0),
              "leaky": lambda s: 0.3 if s < 12 else 0.17}
    calls = []
    res = search_working_point(make_measure(curves, calls), list(curves), lo=-10, hi=42,
                               start=11.0, max_points=10)
    r = res["estimators"]["leaky"]
    assert r["status"] == "right_censored" and r["reason"] == "floor"
    assert res["estimators"]["perfect"]["status"] == "ok"
    assert len(calls) <= 6 and max(calls) < 30


def test_low_snr_plateau_is_not_a_floor():
    # every estimator on the high-BER plateau: keep climbing until the waterfall
    curves = {"perfect": waterfall(20.0, slope_db=1.0), "ls": waterfall(24.0, slope_db=1.0)}
    res = search_working_point(make_measure(curves), list(curves), lo=-10, hi=42,
                               start=-8.0, max_points=15)
    assert all(v["status"] == "ok" for v in res["estimators"].values())


def test_excess_loss_cap():
    curves = {"perfect": waterfall(5.0), "awful": waterfall(35.0, slope_db=0.5)}
    res = search_working_point(make_measure(curves), list(curves), lo=-10, hi=48,
                               start=4.0, max_points=15, max_excess_db=15.0)
    r = res["estimators"]["awful"]
    assert r["status"] == "right_censored" and r["reason"] == "excess_loss"
    assert r["wp_lower_db"] >= res["estimators"]["perfect"]["wp_upper_db"] + 15.0 - 1e-9
    assert max(res["snr_db"]) <= 5.0 + 2.0 + 15.0 + 6.0
