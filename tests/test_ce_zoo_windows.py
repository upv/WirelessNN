"""ce_zoo DelayWindow reproduces nr_ul_sim ls_hard_window / ls_soft_window on one rx antenna."""

import sys
from pathlib import Path

import pytest
import torch

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.channel_estimation import estimator_args
from nr_ul_sim.paper_ce import BlockEstimator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "ce_zoo"))
from models import DelayWindow, estimate_slices  # noqa: E402


@pytest.mark.parametrize("snr", [0.0, 20.0])
def test_delay_window_matches_simulator(snr):
    cfg = SimConfig(num_prb=24, fft_size=512, batch_size=2, max_mc_iter=1, num_target_bit_errors=1,
                    receivers=("lmmse",), channel_estimators=("ls_hard_window", "ls_soft_window"),
                    snr_db=[snr], iot_db=[0.0], seed=3, num_rx_ant=1)
    sim = NRUplinkSimulator(cfg)
    slot = sim.generate_slot(2, snr, 0.0)
    sim.estimate_csi(slot)
    est = BlockEstimator(*estimator_args(sim.transmitter))
    h_blk, _ = est.blocks(slot["y"], slot["no"])
    shape = h_blk.shape
    blk = h_blk.reshape(-1, shape[-2], shape[-1])
    for mode in ("hard", "soft"):
        net = DelayWindow(mode=mode, dmrs=tuple(est.dmrs_syms), nd=shape[-2], n_sym=est.num_sym)
        out = estimate_slices(net, blk)
        ref = slot["csi"][f"ls_{mode}_window"][0].reshape(out.shape).to(out.dtype)
        rel = (out - ref).abs().pow(2).mean() / ref.abs().pow(2).mean()
        assert float(rel) < 1e-8, (mode, float(rel))
