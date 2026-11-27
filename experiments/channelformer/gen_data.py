"""Training data for Channelformer from the link simulator (4 gNB ant x 2 UE ant, rank 2).

Every chunk draws a scenario (channel, delay spread, speed) and simulates
``--chunk-slots`` slots without interference at uniform random SNR. Stored per
(rx antenna, layer) slice: despread DMRS LS blocks [2, M] complex64, the true
effective channel [14, N] as float16 re/im, the nominal SNR and the scenario.

    PYTHONPATH=.:experiments/channelformer python experiments/channelformer/gen_data.py --out data/cf_4x2 --slots 12000 --val-slots 1200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sionna.phy
import torch

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.channel_estimation import estimator_args
from nr_ul_sim.paper_ce import BlockEstimator

CHANNELS = ("cdl-b", "cdl-c", "cdl-d", "umi", "uma")
CDL_DS_NS = (30.0, 100.0, 300.0)
SPEEDS = (3.0, 30.0, 60.0, 120.0)


def scenario(rng):
    ch = str(rng.choice(CHANNELS))
    ds = float(rng.choice(CDL_DS_NS)) if ch.startswith("cdl") else None
    return ch, ds, float(rng.choice(SPEEDS))


def make_split(out: Path, name: str, n_slots: int, args, seed: int):
    rng = np.random.default_rng(seed)
    blks, tgts, snrs, meta = [], [], [], []
    done, chunk_id, t0 = 0, 0, time.time()
    while done < n_slots:
        ch, ds, speed = scenario(rng)
        cfg = SimConfig(channel=ch, delay_spread_ns=ds, speed_kmh=speed, num_ue=1,
                        num_rx_ant=args.rx_ant, num_ue_ant=args.ue_ant, num_layers=args.rank,
                        num_prb=args.prb, fft_size=args.fft_size, carrier_frequency=args.carrier_freq,
                        mapping_type=args.mapping_type, num_ofdm_symbols=args.num_symbols,
                        dmrs_additional_position=args.dmrs_addpos,
                        num_interferers=0, receivers=("lmmse",),
                        channel_estimators=("ls_lin",), iot_db=[0.0],
                        seed=int(rng.integers(1 << 30)))
        sim = NRUplinkSimulator(cfg)
        est = BlockEstimator(*estimator_args(sim.transmitter))
        n_chunk = min(args.chunk_slots, n_slots - done)
        for _ in range(0, n_chunk, args.batch):
            snr = float(rng.uniform(args.snr_lo, args.snr_hi))
            with torch.no_grad():
                slot = sim.generate_slot(args.batch, snr, 0.0)
                no = torch.tensor(slot["no"], device=slot["y"].device, dtype=slot["y"].real.dtype)
                h_blk, _ = est.blocks(slot["y"], no)                 # [B,1,Rx,Tx,S,Nd,M]
                h = sim.true_channel(slot["h"])                      # [B,1,Rx,Tx,S,sym,N]
            nd, m = h_blk.shape[-2:]
            blk = h_blk.reshape(-1, nd, m).cpu().numpy().astype(np.complex64)
            tgt = h.reshape(-1, h.shape[-2], h.shape[-1])
            tgt = torch.stack([tgt.real, tgt.imag], -1).cpu().numpy().astype(np.float16)
            blks.append(blk)
            tgts.append(tgt)
            snrs.append(np.full(len(blk), snr, np.float32))
            meta += [(ch, ds or 0.0, speed)] * len(blk)
        done += n_chunk
        chunk_id += 1
        print(f"{name}: {done}/{n_slots} slots, last {ch} ds={ds} v={speed} "
              f"[{time.time() - t0:.0f} s]", flush=True)
        del sim
    np.save(out / f"{name}_blk.npy", np.concatenate(blks))
    np.save(out / f"{name}_tgt.npy", np.concatenate(tgts))
    np.save(out / f"{name}_snr.npy", np.concatenate(snrs))
    (out / f"{name}_meta.json").write_text(json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slots", type=int, default=12000)
    ap.add_argument("--val-slots", type=int, default=1200)
    ap.add_argument("--chunk-slots", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--snr-lo", type=float, default=-10.0)
    ap.add_argument("--snr-hi", type=float, default=30.0)
    ap.add_argument("--rx-ant", type=int, default=4)
    ap.add_argument("--ue-ant", type=int, default=2)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--prb", type=int, default=68)
    ap.add_argument("--fft-size", type=int, default=1024)
    ap.add_argument("--carrier-freq", type=float, default=3.5e9)
    ap.add_argument("--mapping-type", choices=("A", "B"), default="A")
    ap.add_argument("--num-symbols", type=int, default=14, help="PUSCH symbols per slot")
    ap.add_argument("--dmrs-addpos", type=int, default=1, help="DMRS additional position")
    ap.add_argument("--seed", type=int, default=2023)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    make_split(args.out, "val", args.val_slots, args, args.seed + 1)
    make_split(args.out, "train", args.slots, args, args.seed)


if __name__ == "__main__":
    main()
