"""Quick BLER working points of the ce_zoo estimators on the 4x2 PUSCH link.

Same link as experiments/channelformer/run_campaign.py (4 gNB x 2 UE antennas, rank 2,
68 PRB, LMMSE receiver, MCS 5/14/20), one speed, a 1 dB SNR grid around the 10 % BLER
point and short Monte Carlo runs. perfect, lmmse_exp and channelformer run on the same
slots as anchors. One JSON per (channel, modulation); existing files are skipped.

    export PYTHONPATH=.:experiments/channelformer:experiments/ce_zoo
    CE_ZOO_WEIGHTS=runs/zoo python experiments/ce_zoo/bler.py --out results/zoo_bler
    python experiments/ce_zoo/bler.py --out results/zoo_bler --summary runs/zoo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLUGINS = (str(REPO / "experiments/channelformer/plugin.py"), str(REPO / "experiments/ce_zoo/plugin.py"))
ZOO = ("zoo_chea", "zoo_chea_xl", "zoo_helena", "zoo_swift", "zoo_denoiserx")
ESTIMATORS = ("perfect", "lmmse_exp", "channelformer") + ZOO
CHANNELS = ("cdl-b", "cdl-c", "cdl-d", "umi", "uma")
MODS = ("qpsk", "qam16", "qam64")
SNR_PER_MOD = {"qpsk": "-5:3:1", "qam16": "3:14:1", "qam64": "8:24:1"}


def cfg_for(args, ch, mod, snr):
    from nr_ul_sim import SimConfig
    return SimConfig(
        channel=ch, speed_kmh=args.speed, num_ue=1, num_rx_ant=4, num_ue_ant=2, num_layers=2,
        snr_db=snr, iot_db=[0.0], num_interferers=0, modulation=mod,
        receivers=("lmmse",), channel_estimators=tuple(args.estimators.split(",")), ce_plugins=PLUGINS,
        num_prb=68, subcarrier_spacing_khz=30.0, carrier_frequency=3.5e9, fft_size=1024,
        mapping_type="A", num_ofdm_symbols=14, dmrs_additional_position=1,
        batch_size=args.batch, max_mc_iter=args.max_iter,
        num_target_bit_errors=1000, num_target_block_errors=args.block_errors, seed=42)


def simulate(args):
    import torch
    from nr_ul_sim import NRUplinkSimulator
    from nr_ul_sim.parameters import parse_csv_floats
    from nr_ul_sim.pusch import build_pusch_configs
    args.out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for ch in args.channels.split(","):
        for mod in args.modulations.split(","):
            f = args.out / f"{ch}_v{args.speed:g}_{mod}.json"
            if f.exists():
                continue
            cfg = cfg_for(args, ch, mod, parse_csv_floats(SNR_PER_MOD[mod]))
            sim = NRUplinkSimulator(cfg)
            camp = sim.run(verbose=False)
            camp["tb_size_bits"] = int(build_pusch_configs(cfg)[0].tb_size)
            camp["mc"] = {"batch": args.batch, "max_iter": args.max_iter, "block_errors": args.block_errors}
            f.write_text(json.dumps(camp, indent=1))
            print(f"{time.strftime('%H:%M:%S')} {ch} {mod} [{camp['duration_s']:.0f} s]", flush=True)
            del sim
            torch.cuda.empty_cache()
    print(f"total {(time.time() - t_all) / 60:.1f} min", flush=True)


def summary(args):
    """One markdown table: params, val NMSE, mean SNR at 10 % BLER over the channels."""
    import torch
    wp = {}
    for f in sorted(args.out.glob("*.json")):
        ch, _, mod = f.stem.split("_")
        for est, curve in json.loads(f.read_text())["iot"]["0.0"].items():
            wp.setdefault(est, {})[(ch, mod)] = curve["working_point_bler_db"]
    meta = {"perfect": ("-", None), "lmmse_exp": ("0", None)}
    cf = torch.load(REPO / "experiments/channelformer/weights/channelformer_4x2.pt", map_location="cpu",
                    weights_only=False)
    n_cf = sum(p.numel() for p in cf["state_dict"].values())
    meta["channelformer"] = (f"{n_cf:,}".replace(",", " "), cf.get("val_nmse_db"))
    for z in ZOO:
        s = torch.load(args.summary / z.removeprefix("zoo_") / "model.pt", map_location="cpu", weights_only=False)
        meta[z] = (f"{s['params']:,}".replace(",", " "), s["val_nmse_db"])
    keys = [(c, m) for m in MODS for c in CHANNELS]
    ref = wp["perfect"]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    print("| estimator | params | val NMSE [dB] | QPSK | 16QAM | 64QAM | loss vs perfect CSI [dB] |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for est, w in wp.items():
        per_mod = [mean([w.get((c, m)) for c in CHANNELS]) for m in MODS]
        loss = mean([w[k] - ref[k] for k in keys if w.get(k) is not None and ref.get(k) is not None])
        rows.append((loss if est != "perfect" else -1, est, per_mod))
    for loss, est, per_mod in sorted(rows):
        p, n = meta.get(est, ("", None))
        print(f"| {est} | {p} | {'' if n is None else f'{n:.2f}'} | "
              + " | ".join(f"{v:.1f}" for v in per_mod) + f" | {'' if est == 'perfect' else f'{loss:.2f}'} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--channels", default=",".join(CHANNELS))
    ap.add_argument("--modulations", default=",".join(MODS))
    ap.add_argument("--speed", type=float, default=30.0)
    ap.add_argument("--estimators", default=",".join(ESTIMATORS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--block-errors", type=int, default=50)
    ap.add_argument("--summary", type=Path, default=None, help="runs dir of train.py: print the table only")
    args = ap.parse_args()
    if args.summary:
        summary(args)
    else:
        simulate(args)


if __name__ == "__main__":
    main()
