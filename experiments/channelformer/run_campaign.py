"""4x2 MIMO PUSCH campaign: BLER/throughput and CE NMSE of every estimator.

One JSON per (channel, speed, modulation) under --out/bler, one per (channel,
speed) under --out/nmse; existing files are skipped (resumable). Each JSON
carries the full resolved SimConfig, TB size, git commit and package versions.

    PYTHONPATH=.:experiments/channelformer python experiments/channelformer/run_campaign.py --out <dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import sionna
import sionna.phy
import torch

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.parameters import CHANNEL_ESTIMATORS, parse_csv_floats
from nr_ul_sim.pusch import build_pusch_configs

HERE = Path(__file__).resolve().parent
PLUGIN = str(HERE / "plugin.py")
ESTIMATORS = CHANNEL_ESTIMATORS + ("channelformer",)
CHANNELS = ("cdl-b", "cdl-c", "cdl-d", "umi", "uma")
# BLER sweep per modulation (the NMSE pass always uses --snr)
SNR_PER_MOD = {"qpsk": "-10:16:2", "qam16": "-4:24:2", "qam64": "2:30:2"}


def provenance(repo: Path) -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()
        except Exception:
            return None
    ckpt = torch.load(os.environ.get("CHANNELFORMER_CKPT", HERE / "weights" / "channelformer_4x2.pt"),
                      map_location="cpu", weights_only=False)
    return {"repo_commit": git("rev-parse", "HEAD"), "repo_dirty": bool(git("status", "--porcelain")),
            "torch": torch.__version__, "sionna": sionna.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "channelformer_epoch": ckpt.get("epoch"), "channelformer_val_nmse_db": ckpt.get("val_nmse_db")}


def base_cfg(args, ch, speed, mod, ests, snr=None):
    return SimConfig(
        channel=ch, speed_kmh=speed, num_ue=1, num_rx_ant=4, num_ue_ant=2, num_layers=2,
        snr_db=args.snr if snr is None else snr, iot_db=[0.0], num_interferers=0, modulation=mod,
        receivers=(args.receiver,), channel_estimators=ests, ce_plugins=(PLUGIN,),
        num_prb=68, subcarrier_spacing_khz=30.0, carrier_frequency=3.5e9, fft_size=1024,
        dmrs_additional_position=1, batch_size=args.batch, max_mc_iter=args.max_iter,
        num_target_bit_errors=args.bit_errors, num_target_block_errors=args.block_errors,
        seed=args.seed)


def nmse_pass(cfg, slots, batch):
    sim = NRUplinkSimulator(cfg)
    ests = [e for e in cfg.channel_estimators if e != "perfect"]
    out = {e: [] for e in ests}
    for snr in cfg.snr_db:
        sionna.phy.config.seed = cfg.seed            # common random numbers over SNR
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        err = {e: 0.0 for e in ests}
        pw = 0.0
        for _ in range(max(1, slots // batch)):
            with torch.no_grad():
                slot = sim.generate_slot(batch, snr, 0.0)
                sim.estimate_csi(slot)
            h = slot["csi"]["perfect"][0]
            pw += float(h.abs().pow(2).mean())
            for e in ests:
                err[e] += float((slot["csi"][e][0] - h).abs().pow(2).mean())
        for e in ests:
            out[e].append(10 * math.log10(err[e] / pw))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--channels", default=",".join(CHANNELS))
    ap.add_argument("--speeds", default="3,30,120")
    ap.add_argument("--modulations", default="qpsk,qam16,qam64")
    ap.add_argument("--snr", default="-10:30:2", help="NMSE sweep (and BLER sweep with --no-snr-per-mod)")
    ap.add_argument("--no-snr-per-mod", action="store_true", help="BLER over --snr for every modulation")
    ap.add_argument("--receiver", default="lmmse")
    ap.add_argument("--estimators", default=",".join(ESTIMATORS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=25)
    ap.add_argument("--bit-errors", type=int, default=1000)
    ap.add_argument("--block-errors", type=int, default=100)
    ap.add_argument("--nmse-slots", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="one SNR point per scenario, for timing")
    args = ap.parse_args()
    args.snr = parse_csv_floats(args.snr)
    ests = tuple(e.strip() for e in args.estimators.split(",") if e.strip())
    if args.dry_run:
        args.snr = [10.0]
    (args.out / "bler").mkdir(parents=True, exist_ok=True)
    (args.out / "nmse").mkdir(parents=True, exist_ok=True)
    prov = provenance(args.repo)
    t_all = time.time()
    for ch in args.channels.split(","):
        for speed in [float(s) for s in args.speeds.split(",")]:
            tag = f"{ch}_v{speed:g}"
            f = args.out / "nmse" / f"{tag}.json"
            if not f.exists():
                t0 = time.time()
                cfg = base_cfg(args, ch, speed, "qpsk", ests if "perfect" in ests else ("perfect",) + ests)
                res = nmse_pass(cfg, args.nmse_slots if not args.dry_run else 16, args.batch)
                f.write_text(json.dumps({"config": cfg.summary(), "provenance": prov, "snr_db": cfg.snr_db,
                                         "nmse_db": res, "slots_per_point": args.nmse_slots,
                                         "duration_s": time.time() - t0}, indent=1))
                print(f"{time.strftime('%H:%M:%S')} NMSE {tag} [{time.time() - t0:.0f} s]", flush=True)
            for mod in args.modulations.split(","):
                f = args.out / "bler" / f"{tag}_{mod}.json"
                if f.exists():
                    continue
                snr = None
                if not args.no_snr_per_mod and not args.dry_run:
                    snr = parse_csv_floats(SNR_PER_MOD[mod])
                cfg = base_cfg(args, ch, speed, mod, ests, snr)
                tbs = int(build_pusch_configs(cfg)[0].tb_size)
                sim = NRUplinkSimulator(cfg)
                camp = sim.run(verbose=False)
                slot_s = 1e-3 / (cfg.subcarrier_spacing_khz / 15.0)
                for curve in camp["iot"]["0.0"].values():
                    curve["throughput_mbps"] = [(1 - b) * tbs / slot_s / 1e6 for b in curve["bler"]]
                camp["tb_size_bits"] = tbs
                camp["slot_duration_s"] = slot_s
                camp["provenance"] = prov
                f.write_text(json.dumps(camp, indent=1))
                print(f"{time.strftime('%H:%M:%S')} BLER {tag} {mod} TBS {tbs} "
                      f"[{camp['duration_s']:.0f} s]", flush=True)
                del sim
                torch.cuda.empty_cache()
    print(f"total {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
