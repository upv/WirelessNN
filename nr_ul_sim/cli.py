from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from .parameters import (
    CHANNEL_ESTIMATORS,
    IOT_COV_METHODS,
    RECEIVERS,
    TX_POWER_NORMS,
    SimConfig,
    canonicalize_estimator_name,
    canonicalize_receiver_name,
    parse_csv_floats,
)
from .plotting import save_campaign_plots
from .simulator import NRUplinkSimulator

# Options whose value may start with a minus sign ("--snr-db -20:30:2").
# argparse would read "-20:30:2" as an option, so the pair is joined with "=".
_SIGNED_VALUE_OPTIONS = {
    "--snr-db", "--iot-db", "--delay-spread-ns", "--speed-kmh", "--target-ber",
    "--target-bler", "--carrier-ghz", "--ce-window-pos-us", "--ce-window-neg-us",
    "--ce-soft-threshold",
}


def join_signed_values(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if arg in _SIGNED_VALUE_OPTIONS and nxt is not None and re.match(r"^-\d", nxt):
            out.append(f"{arg}={nxt}")
            skip = True
        else:
            out.append(arg)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="5G NR PUSCH uplink BER (Sionna)")
    p.add_argument("--channel", default="cdl-c", choices=["cdl-b", "cdl-c", "cdl-d", "umi", "uma"])
    p.add_argument("--delay-spread-ns", type=float, default=None, help="CDL RMS delay spread [ns]")
    p.add_argument("--num-ue", type=int, default=1)
    p.add_argument("--num-rx-ant", type=int, default=4)
    p.add_argument("--num-ue-ant", type=int, default=None, help="UE ports 1/2/4; default from rank")
    p.add_argument("--rank", type=int, default=1, dest="num_layers")
    p.add_argument("--speed-kmh", type=float, default=3.0)
    p.add_argument("--snr-db", default="-20:30:2", help="Comma list or start:stop:step")
    p.add_argument("--iot-db", default="0,10,20", help="INR dB; 0 = no interferer")
    p.add_argument("--num-interferers", type=int, default=2)
    p.add_argument("--modulation", default="qpsk", choices=["qpsk", "qam16", "qam64"])
    p.add_argument("--mcs-index", type=int, default=None)
    p.add_argument("--mcs-table", type=int, default=None)
    p.add_argument("--receivers", default="mr,lmmse,ideal_mmse,zf,irc")
    p.add_argument("--estimators", default="ls_lin", dest="channel_estimators",
                   help="Comma list: " + ",".join(CHANNEL_ESTIMATORS))
    p.add_argument("--perfect-csi", action="store_true")
    p.add_argument("--iot-cov", default="perfect", choices=list(IOT_COV_METHODS))
    p.add_argument("--tx-power-norm", default="per_ue", choices=list(TX_POWER_NORMS),
                   help="per_ue: unit total power per UE; per_layer: unit power per layer")
    p.add_argument("--ce-window-pos-us", type=float, default=3.0,
                   help="Delay-domain window after the first tap [us] (windowed CE)")
    p.add_argument("--ce-window-neg-us", type=float, default=1.0,
                   help="Delay-domain window before the first tap [us] (windowed CE)")
    p.add_argument("--ce-soft-threshold", type=float, default=1.5,
                   help="Soft window: tap kept when its power exceeds this multiple of the noise")
    p.add_argument("--ce-soft-within-window", action="store_true",
                   help="Soft window: also zero every tap outside the hard window")
    p.add_argument("--ce-time-interp", default="linear", choices=["linear", "avg"])
    p.add_argument("--ce-model-dir", default=None,
                   help="Trained paper estimators (default: models/paper_ce)")
    p.add_argument("--incm-band-sc", type=int, default=24,
                   help="Subcarriers per INCM band for --iot-cov incm_oas")
    p.add_argument("--ce-lmmse-prior-ds-ns", type=float, default=None,
                   help="RMS delay spread of the exponential-PDP prior of lmmse_exp (default: scenario value)")
    p.add_argument("--carrier-ghz", type=float, default=3.5)
    p.add_argument("--num-prb", type=int, default=68)
    p.add_argument("--fft-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-mc-iter", type=int, default=20)
    p.add_argument("--num-target-bit-errors", type=int, default=200)
    p.add_argument("--num-target-block-errors", type=int, default=20)
    p.add_argument("--target-ber", type=float, default=0.01)
    p.add_argument("--target-bler", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results")
    p.add_argument("--quick", action="store_true", help="Tiny smoke run")
    return p


def config_from_args(args: argparse.Namespace) -> SimConfig:
    receivers = tuple(
        canonicalize_receiver_name(r) for r in args.receivers.split(",") if r.strip()
    )
    unknown = [r for r in receivers if r not in RECEIVERS]
    if unknown:
        raise SystemExit(f"Unknown receivers {unknown}")
    estimators = tuple(
        canonicalize_estimator_name(e)
        for e in args.channel_estimators.split(",") if e.strip()
    )
    unknown_ce = [e for e in estimators if e not in CHANNEL_ESTIMATORS]
    if unknown_ce:
        raise SystemExit(f"Unknown channel estimators {unknown_ce}")

    cfg = SimConfig(
        channel=args.channel,
        delay_spread_ns=args.delay_spread_ns,
        num_ue=args.num_ue,
        num_rx_ant=args.num_rx_ant,
        num_ue_ant=args.num_ue_ant,
        num_layers=args.num_layers,
        speed_kmh=args.speed_kmh,
        snr_db=parse_csv_floats(args.snr_db),
        iot_db=parse_csv_floats(args.iot_db),
        num_interferers=args.num_interferers,
        modulation=args.modulation,
        mcs_table=args.mcs_table,
        mcs_index=args.mcs_index,
        receivers=receivers,
        channel_estimators=estimators,
        perfect_csi=args.perfect_csi,
        iot_cov=args.iot_cov,
        tx_power_norm=args.tx_power_norm,
        ce_window_pos_us=args.ce_window_pos_us,
        ce_window_neg_us=args.ce_window_neg_us,
        ce_soft_threshold=args.ce_soft_threshold,
        ce_soft_within_window=args.ce_soft_within_window,
        ce_time_interp=args.ce_time_interp,
        ce_lmmse_prior_ds_ns=args.ce_lmmse_prior_ds_ns,
        ce_model_dir=args.ce_model_dir,
        incm_band_sc=args.incm_band_sc,
        carrier_frequency=args.carrier_ghz * 1e9,
        num_prb=args.num_prb,
        fft_size=args.fft_size,
        batch_size=args.batch_size,
        max_mc_iter=args.max_mc_iter,
        num_target_bit_errors=args.num_target_bit_errors,
        num_target_block_errors=args.num_target_block_errors,
        target_ber=args.target_ber,
        target_bler=args.target_bler,
        seed=args.seed,
    )
    if args.quick:
        cfg.num_prb = 8
        cfg.snr_db = [-10.0, 0.0, 10.0]
        cfg.iot_db = [0.0]
        cfg.batch_size = 1
        cfg.max_mc_iter = 2
        cfg.num_target_bit_errors = 1
        cfg.num_target_block_errors = 1
    return cfg


def save_campaign(campaign: dict, outdir: Path) -> tuple[Path, list[Path]]:
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = campaign["config"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{cfg['channel']}_{cfg['modulation']}_ue{cfg['num_ue']}_r{cfg['rank']}_{stamp}"
    json_path = outdir / f"{stem}.json"
    json_path.write_text(json.dumps(campaign, indent=2))
    return json_path, save_campaign_plots(campaign, outdir, stem)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(join_signed_values(argv))
    cfg = config_from_args(args)
    print("NR PUSCH uplink simulation")
    for key, value in cfg.summary().items():
        print(f"  {key}: {value}")
    json_path, plots = save_campaign(NRUplinkSimulator(cfg).run(), Path(args.outdir))
    print(f"\nWrote {json_path}")
    for png_path in plots:
        print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
