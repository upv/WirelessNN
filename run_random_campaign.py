#!/usr/bin/env python
"""Random-channel working-point campaign.

    .venv/bin/python run_random_campaign.py --out /root/reports/random_ch --per-model 1000 --workers 2

Writes into ``--out``:

    campaign_config.json   run settings and parameter ranges
    plan.jsonl             every configuration (id, seed, model, #UE, rank, modulation)
    results/worker_K.jsonl one JSON record per finished configuration
    arrays/ID.npz          cluster tables / drop geometry of every configuration
    errors.jsonl           configurations that raised (retried on the next start)
    logs/worker_K.log      one line per configuration
    DONE                   written when every planned configuration has a record

Re-running the same command resumes: finished ids are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def done_ids(out: Path) -> set[int]:
    ids = set()
    for f in (out / "results").glob("worker_*.jsonl"):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(int(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    pass  # a line cut by a kill is simply redone
    return ids


def load_plan(out: Path) -> list[dict]:
    with open(out / "plan.jsonl") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_plan(args, out: Path) -> None:
    from nr_ul_sim import random_channels as rc
    from nr_ul_sim import random_campaign as rcamp

    if (out / "plan.jsonl").exists():
        return
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    rows = rcamp.plan_configs(args.per_model, channels, args.seed)
    with open(out / "plan.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ranges = {k: getattr(rc, k) for k in dir(rc) if k.isupper()}
    ranges.update({k: getattr(rcamp, k) for k in
                   ("SNR_RANGE_DB", "BASE_WP_DB", "IOT_ZERO_PROBABILITY", "IOT_DB_RANGE")})
    cfg = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
        "args": vars(args),
        "num_configs": len(rows),
        "channels": channels,
        "receiver": "irc",
        "estimators": args.estimators.split(","),
        "iot_cov": "perfect",
        "working_point": "BER",
        "parameter_ranges": ranges,
        "notes": rcamp.__doc__,
    }
    (out / "campaign_config.json").write_text(json.dumps(cfg, indent=2, default=_json_default))


def worker(args, out: Path, k: int) -> None:
    import sionna.phy
    import torch

    from nr_ul_sim.random_campaign import run_config

    sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    plan = [r for r in load_plan(out) if r["id"] % args.workers == k]
    if args.limit:
        plan = plan[: args.limit]
    finished = done_ids(out)
    todo = [r for r in plan if r["id"] not in finished]
    res_path = out / "results" / f"worker_{k}.jsonl"
    log = open(out / "logs" / f"worker_{k}.log", "a", buffering=1)
    log.write(f"# {time.strftime('%F %T')} worker {k}: {len(todo)} of {len(plan)} to do\n")
    for row in todo:
        try:
            rec, arrays = run_config(
                row, num_prb=args.num_prb, fft_size=args.fft_size, batch_size=args.batch_size,
                max_mc_iter=args.max_mc_iter, target_bit_errors=args.target_bit_errors,
                target_block_errors=args.target_block_errors, target_ber=args.target_ber,
                max_points=args.max_points, estimators=tuple(args.estimators.split(",")),
            )
            np.savez_compressed(out / "arrays" / f"{row['id']:06d}.npz",
                                **{kk: np.asarray(v) for kk, v in arrays.items()})
            with open(res_path, "a") as fh:
                fh.write(json.dumps(rec, default=_json_default) + "\n")
            st = rec["estimators"]
            wps = " ".join(
                f"{n}={v['wp_db']:.1f}" if v["status"] == "ok"
                else f"{n}={'>' if v['status'] == 'right_censored' else '<'}"
                     f"{v['wp_lower_db'] if v['status'] == 'right_censored' else v['wp_upper_db']:.0f}"
                for n, v in st.items()
            )
            log.write(
                f"{time.strftime('%T')} id={row['id']} {row['channel']} {row['modulation']} "
                f"ue{row['num_ue']}r{row['rank']} iot={rec['iot_db']:.1f} "
                f"pts={rec['search']['num_points']} t={rec['timing_s']['total']:.1f}s | {wps}\n"
            )
        except Exception as exc:  # keep the campaign going; retried on restart
            with open(out / "errors.jsonl", "a") as fh:
                fh.write(json.dumps({"id": row["id"], "row": row, "time": time.strftime("%F %T"),
                                     "error": repr(exc), "traceback": traceback.format_exc()}) + "\n")
            log.write(f"{time.strftime('%T')} id={row['id']} ERROR {exc!r}\n")
            torch.cuda.empty_cache()
    log.write(f"# {time.strftime('%F %T')} worker {k} finished\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--per-model", type=int, default=1000)
    p.add_argument("--channels", default="cdl-b,cdl-c,cdl-d,umi,uma")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--limit", type=int, default=0, help="per-worker cap (testing)")
    p.add_argument("--num-prb", type=int, default=16)
    p.add_argument("--fft-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-mc-iter", type=int, default=8)
    p.add_argument("--target-bit-errors", type=int, default=300)
    p.add_argument("--target-block-errors", type=int, default=20)
    p.add_argument("--target-ber", type=float, default=1e-2)
    p.add_argument("--max-points", type=int, default=10)
    p.add_argument("--estimators", default="",
                   help="comma-separated channel estimators (default: all in CHANNEL_ESTIMATORS); "
                        "perfect is always added")
    args = p.parse_args()
    from nr_ul_sim.random_campaign import parse_estimators
    try:
        args.estimators = ",".join(parse_estimators(args.estimators))
    except ValueError as exc:
        p.error(str(exc))

    out = Path(args.out)
    for d in ("results", "arrays", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)

    if args.worker is not None:
        worker(args, out, args.worker)
        return 0

    write_plan(args, out)
    # the launcher itself never gets --worker, so its argv is passed through as is
    base = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
    procs = [subprocess.Popen(base + ["--worker", str(k)]) for k in range(args.workers)]
    codes = [pr.wait() for pr in procs]
    total = len(load_plan(out))
    n_done = len(done_ids(out))
    print(f"workers exited {codes}; {n_done}/{total} configurations done")
    if n_done >= total:
        (out / "DONE").write_text(time.strftime("%F %T") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
