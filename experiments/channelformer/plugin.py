"""nr_ul_sim estimator plugin: ``--ce-plugin channelformer/plugin.py --estimators ...,channelformer``.

Weights: $CHANNELFORMER_CKPT, else weights/channelformer_4x2.pt next to this file.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

from model import Channelformer, estimate_slices
from nr_ul_sim.channel_estimation import estimator_args
from nr_ul_sim.paper_ce import BlockEstimator, ErrVarTable
from nr_ul_sim.parameters import register_estimator

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "weights" / "channelformer_4x2.pt"


class ChannelformerEstimator(BlockEstimator):
    """Channelformer on the despread DMRS LS blocks of every (rx antenna, layer)."""

    def __init__(self, *args, ckpt=None, **kwargs):
        super().__init__(*args, **kwargs)
        path = Path(ckpt or os.environ.get("CHANNELFORMER_CKPT", DEFAULT_CKPT))
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.net = Channelformer(**state["config"])
        self.net.load_state_dict(state["state_dict"])
        self.net.eval()
        self.err_table = ErrVarTable(state["errvar_snr_db"], state["errvar_mse"])
        if len(self.dmrs_syms) != 2:
            raise ValueError("Channelformer is trained for 2 DMRS symbols (additional position 1)")
        if set(self.comb.ravel().tolist()) != {0}:
            warnings.warn("Channelformer was trained on CDM group 0 only")

    @torch.no_grad()
    def process(self, h_blk, var):
        net = self.net.to(h_blk.device)
        shape = h_blk.shape                               # [B,1,Rx,Tx,S,Nd,M]
        out = estimate_slices(net, h_blk.reshape(-1, shape[-2], shape[-1]))
        h_hat = out.reshape(*shape[:-2], out.shape[-2], out.shape[-1]).to(h_blk.dtype)
        err = self.err_table(self.snr_db_from_block_var(var)).reshape(-1, 1, 1, 1, 1, 1, 1)
        return h_hat, torch.broadcast_to(err, h_hat.shape).contiguous()


def _factory(transmitter, cfg):
    return ChannelformerEstimator(*estimator_args(transmitter))


register_estimator(
    "channelformer", _factory,
    "Channelformer (Luan & Thompson, TWC 2023), 12-PRB windows [plugin, trained]",
)
