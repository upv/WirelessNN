"""Plot helpers from a synthetic campaign (no Sionna)."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

from nr_ul_sim.plotting import plot_campaign, plot_working_points, save_campaign_plots


def _curve(snr, ber, bler, wp):
    return {
        "snr_db": list(snr),
        "ber": list(ber),
        "bler": list(bler),
        "working_point_db": wp,
    }


def sample_campaign():
    snr = [-10.0, -5.0, 0.0, 5.0, 10.0]
    return {
        "config": {
            "channel": "cdl-c",
            "modulation": "qpsk",
            "num_ue": 1,
            "rank": 1,
            "num_rx_ant": 4,
            "speed_kmh": 3.0,
            "target_ber": 0.01,
        },
        "iot": {
            "0.0": {
                "mr": _curve(snr, [0.3, 0.12, 0.04, 0.009, 0.001], [1, 0.9, 0.5, 0.12, 0.02], 4.6),
                "lmmse": _curve(snr, [0.25, 0.08, 0.02, 0.004, 4e-4], [1, 0.7, 0.3, 0.05, 0.01], 2.1),
                "ideal_mmse": _curve(snr, [0.2, 0.05, 0.01, 0.002, 2e-4], [0.95, 0.55, 0.18, 0.03, 0.006], 0.0),
                "zf": _curve(snr, [0.28, 0.1, 0.03, 0.007, 8e-4], [1, 0.8, 0.4, 0.08, 0.015], 3.4),
                "irc": _curve(snr, [0.24, 0.07, 0.018, 0.003, 3e-4], [1, 0.65, 0.25, 0.04, 0.008], 1.8),
            },
            "10.0": {
                "mr": _curve(snr, [0.4, 0.22, 0.09, 0.03, 0.008], [1, 1, 0.8, 0.35, 0.1], 8.4),
                "lmmse": _curve(snr, [0.35, 0.16, 0.05, 0.012, 0.003], [1, 0.95, 0.55, 0.18, 0.04], 5.6),
                "ideal_mmse": _curve(snr, [0.22, 0.08, 0.02, 0.005, 8e-4], [1, 0.7, 0.28, 0.06, 0.012], 2.5),
                "zf": _curve(snr, [0.38, 0.2, 0.07, 0.02, 0.006], [1, 0.98, 0.7, 0.25, 0.07], 7.1),
                "irc": _curve(snr, [0.26, 0.09, 0.025, 0.006, 0.001], [1, 0.75, 0.32, 0.07, 0.015], 3.2),
            },
        },
    }


def test_plot_campaign_writes_png(tmp_path: Path):
    out = plot_campaign(sample_campaign(), tmp_path / "ber_bler.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_working_points_writes_png(tmp_path: Path):
    out = plot_working_points(sample_campaign(), tmp_path / "wp.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_save_campaign_plots(tmp_path: Path):
    plots = save_campaign_plots(sample_campaign(), tmp_path, "demo")
    assert [p.name for p in plots] == ["demo.png", "demo_working_points.png"]
    assert all(p.is_file() for p in plots)
