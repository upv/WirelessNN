"""nr_ul_sim estimator plugin for the ce_zoo models:

    --ce-plugin experiments/ce_zoo/plugin.py \
    --estimators ...,zoo_chea,zoo_chea_xl,zoo_helena,zoo_swift,zoo_denoiserx

Weights: <dir>/<name>.pt with <dir> = $CE_ZOO_WEIGHTS, else weights/ next to this file.
A checkpoint written by train.py (runs/<name>/model.pt) works as well.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from models import estimate_slices, from_checkpoint
from nr_ul_sim.channel_estimation import estimator_args
from nr_ul_sim.paper_ce import BlockEstimator, ErrVarTable
from nr_ul_sim.parameters import register_estimator

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("CE_ZOO_WEIGHTS", HERE / "weights"))
NAMES = {
    "chea": "CHEA d=16 (arXiv 2607.16462)",
    "chea_xl": "CHEA-XL d=64 (arXiv 2607.16462)",
    "helena": "HELENA (arXiv 2506.13408)",
    "swift": "SwiftChannel student (arXiv 2605.01931)",
    "denoiserx": "EqDeepRx DenoiseNN + linear interpolation (arXiv 2602.11834)",
}


def checkpoint_path(name: str) -> Path:
    for p in (WEIGHTS / f"{name}.pt", WEIGHTS / name / "model.pt"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no checkpoint for {name} in {WEIGHTS}")


class ZooEstimator(BlockEstimator):
    def __init__(self, *args, name: str, **kwargs):
        super().__init__(*args, **kwargs)
        state = torch.load(checkpoint_path(name), map_location="cpu", weights_only=False)
        self.net = from_checkpoint(state)
        dmrs = tuple(state["config"]["dmrs"])
        if tuple(int(s) for s in self.dmrs_syms) != dmrs:
            raise ValueError(f"{name} trained for DMRS symbols {dmrs}, simulator uses {tuple(self.dmrs_syms)}")
        self.err_table = ErrVarTable(state["errvar_snr_db"], state["errvar_mse"])

    @torch.no_grad()
    def process(self, h_blk, var):
        net = self.net.to(h_blk.device)
        shape = h_blk.shape                                   # [B,1,Rx,Tx,S,Nd,M]
        out = estimate_slices(net, h_blk.reshape(-1, shape[-2], shape[-1]))
        h_hat = out.reshape(*shape[:-2], out.shape[-2], out.shape[-1]).to(h_blk.dtype)
        err = self.err_table(self.snr_db_from_block_var(var)).reshape(-1, 1, 1, 1, 1, 1, 1)
        return h_hat, torch.broadcast_to(err, h_hat.shape).contiguous()


def _factory_for(name):
    def _factory(transmitter, cfg):
        return ZooEstimator(*estimator_args(transmitter), name=name)
    return _factory


for _n, _d in NAMES.items():
    register_estimator(f"zoo_{_n}", _factory_for(_n), f"{_d} [plugin, trained]")
