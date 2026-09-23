import numpy as np

from nr_ul_sim.ridge import RidgePipeline, design_matrix


def _toy(n=80, seed=0):
    rng = np.random.default_rng(seed)
    iot = rng.uniform(0, 20, n)
    bits = rng.choice([2, 4], n)
    y = 4.0 + 0.6 * iot + 2.0 * bits + rng.normal(0, 0.3, n)
    features = np.column_stack([
        rng.normal(0, 1, n), rng.normal(8, 2, n), rng.normal(4, 1, n),
        np.ones(n) * 0.99, np.ones(n), np.ones(n) * 4, np.ones(n) * 2,
    ]).astype(np.float32)
    return {
        "features": features,
        "feature_names": np.array([
            "gain_db", "cond_db", "capacity_bpcu_0db", "freq_corr",
            "time_corr", "num_rx_ant", "num_streams",
        ]),
        "iot_db": iot.astype(np.float32),
        "num_layers_total": np.ones(n, np.int32) * 2,
        "num_ue": np.ones(n, np.int32) * 2,
        "rank": np.ones(n, np.int32),
        "speed_kmh": np.full(n, 5.0, np.float32),
        "delay_spread_ns": np.full(n, 100.0, np.float32),
        "num_interferers": np.ones(n, np.int32),
        "num_bits_per_symbol": bits.astype(np.int32),
        "target_coderate": np.where(bits == 2, 0.3, 0.54).astype(np.float32),
        "channel_code": rng.integers(0, 4, n).astype(np.int32),
        "channel_names": np.array(["cdl-b", "cdl-c", "umi", "uma"]),
        "modulation_code": (bits == 4).astype(np.int32),
        "scenario_id": np.arange(n, dtype=np.int64),
        "receivers": np.array(["lmmse"]),
        "wp_lmmse": y.astype(np.float32),
    }


def test_design_matrix_shape():
    data = _toy()
    x, names, fill = design_matrix(data)
    assert x.shape == (80, 7 + 10 + 4)
    assert fill == 100.0
    assert names[-1] == "is_uma"


def test_fit_predict_save_load(tmp_path):
    data = _toy()
    pipe = RidgePipeline.fit(data, alpha=1.0, test_fraction=0.2, seed=1)
    pred = pipe.predict(data, "lmmse")
    assert pred.shape == (80,)
    metrics = pipe.evaluate(data)
    assert metrics["receivers"]["lmmse"]["test"]["rmse"] < 1.0

    pipe.save(tmp_path)
    loaded = RidgePipeline.load(tmp_path)
    assert np.allclose(loaded.predict(data, "lmmse"), pred)
    assert (tmp_path / "coefficients.csv").exists()
