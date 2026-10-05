#!/usr/bin/env python
"""Flatten a random-channel campaign into tables for the working-point model.

    .venv/bin/python summarize_random_campaign.py /root/reports/random_ch_2026-09-30

Writes into the campaign directory:

    wp_wide.csv   one row per configuration: parameters, measured channel
                  features, and per estimator status / wp / bracket / CE loss
    wp_long.csv   one row per (configuration, estimator): the censored label
                  (wp_lower_db, wp_upper_db; empty = open end) for an
                  interval-censored regression, plus the parameters
    summary.md    progress, ETA, censoring rates, CE loss per model and estimator
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nr_ul_sim.parameters import CHANNEL_ESTIMATORS  # noqa: E402

ESTIMATORS = list(CHANNEL_ESTIMATORS)  # estimators missing from a record are skipped
BASE_KEYS = ["id", "seed", "channel", "num_ue", "rank", "num_layers_total", "modulation",
             "iot_db", "num_interferers", "num_prb", "num_rx_ant", "num_ue_ant",
             "prior_ds_ns", "prior_speed_kmh"]


def load_records(out: Path) -> list[dict]:
    recs = {}
    for f in sorted((out / "results").glob("worker_*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[r["id"]] = r
    return [recs[k] for k in sorted(recs)]


def scalar_items(d: dict, prefix: str):
    for k, v in d.items():
        if isinstance(v, bool):
            yield f"{prefix}{k}", int(v)
        elif isinstance(v, (int, float)) or v is None:
            yield f"{prefix}{k}", v


def base_row(r: dict) -> dict:
    row = {k: r.get(k) for k in BASE_KEYS}
    row["mcs_index"] = r["mcs"][1]
    ues = r["ue"]
    ds = [u.get("ds_ns", u.get("lsp_ds_ns")) for u in ues]
    row["ue_ds_ns_max"] = max(ds)
    row["ue_ds_ns_mean"] = float(np.mean(ds))
    row["ue_speed_kmh_max"] = max(u["speed_kmh"] for u in ues)
    row["ue_speed_kmh_mean"] = float(np.mean([u["speed_kmh"] for u in ues]))
    for i, u in enumerate(ues):
        row.update(scalar_items(u, f"ue{i}_"))
    for i, u in enumerate(r["interferers"]):
        row.update(scalar_items(u, f"int{i}_"))
    row.update(scalar_items(r["features"], "feat_"))
    row["search_points"] = r["search"]["num_points"]
    row["time_s"] = r["timing_s"]["total"]
    return row


def main(out: Path) -> int:
    recs = load_records(out)
    plan_n = sum(1 for _ in open(out / "plan.jsonl"))
    if not recs:
        print("no records yet")
        return 0
    global ESTIMATORS
    ESTIMATORS = [n for n in ESTIMATORS if any(n in r["estimators"] for r in recs)]

    wide, long_rows = [], []
    for r in recs:
        row = base_row(r)
        est = r["estimators"]
        perfect = est.get("perfect", {})
        for n in ESTIMATORS:
            if n not in est:
                continue
            e = est[n]
            row[f"{n}_status"] = e["status"]
            row[f"{n}_reason"] = e["reason"]
            row[f"{n}_wp_db"] = e["wp_db"]
            row[f"{n}_wp_lower_db"] = e["wp_lower_db"]
            row[f"{n}_wp_upper_db"] = e["wp_upper_db"]
            row[f"{n}_nonmonotone"] = int(e["nonmonotone"])
            row[f"{n}_ce_loss_db"] = (
                e["wp_db"] - perfect["wp_db"]
                if e["status"] == "ok" and perfect.get("status") == "ok" and n != "perfect" else None
            )
            lr = {k: row[k] for k in BASE_KEYS + ["mcs_index", "ue_ds_ns_max", "ue_speed_kmh_max"]}
            lr.update({"estimator": n, "status": e["status"], "reason": e["reason"],
                       "wp_db": e["wp_db"], "wp_lower_db": e["wp_lower_db"],
                       "wp_upper_db": e["wp_upper_db"],
                       "perfect_wp_db": perfect.get("wp_db"),
                       "ce_loss_db": row[f"{n}_ce_loss_db"]})
            lr.update({k: v for k, v in row.items() if k.startswith("feat_")})
            long_rows.append(lr)
        wide.append(row)

    for name, rows in (("wp_wide.csv", wide), ("wp_long.csv", long_rows)):
        cols = []
        for rr in rows:
            for k in rr:
                if k not in cols:
                    cols.append(k)
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for rr in rows:
                w.writerow({k: ("" if rr.get(k) is None else rr.get(k)) for k in cols})

    # ---------------- report
    lines = [f"# Random-channel campaign: {len(recs)} / {plan_n} configurations",
             "", f"Generated {time.strftime('%F %T')}.", ""]
    times = np.array([r["timing_s"]["total"] for r in recs])
    logs = sorted((out / "logs").glob("worker_*.log"))
    workers = max(len(logs), 1)
    eta_h = (plan_n - len(recs)) * times.mean() / workers / 3600
    lines += [f"Mean time per configuration {times.mean():.1f} s per worker, {workers} workers, "
              f"remaining about {eta_h:.1f} h.", ""]

    by_ch = defaultdict(list)
    for r in recs:
        by_ch[r["channel"]].append(r)
    lines += ["## Censoring by estimator", "",
              "| model | estimator | ok | right | left | floor | excess | hi_limit |",
              "|---|---|---|---|---|---|---|---|"]
    for ch in sorted(by_ch):
        for n in ESTIMATORS:
            st = [r["estimators"][n] for r in by_ch[ch] if n in r["estimators"]]
            c = lambda f: sum(1 for e in st if f(e))
            lines.append(
                f"| {ch} | {n} | {c(lambda e: e['status'] == 'ok')} "
                f"| {c(lambda e: e['status'] == 'right_censored')} "
                f"| {c(lambda e: e['status'] == 'left_censored')} "
                f"| {c(lambda e: e['reason'] == 'floor')} "
                f"| {c(lambda e: e['reason'] == 'excess_loss')} "
                f"| {c(lambda e: e['reason'] == 'hi_limit')} |")
    lines += ["", "## CE loss vs perfect CSI, dB (configurations where both are ok)", "",
              "| model | " + " | ".join(ESTIMATORS[1:]) + " |",
              "|---|" + "---|" * (len(ESTIMATORS) - 1)]
    for ch in sorted(by_ch):
        cells = []
        for n in ESTIMATORS[1:]:
            v = [row[f"{n}_ce_loss_db"] for row in wide
                 if row["channel"] == ch and row.get(f"{n}_ce_loss_db") is not None]
            cells.append(f"{np.median(v):.2f} (n={len(v)})" if v else "-")
        lines.append(f"| {ch} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print(f"wrote {out / 'wp_wide.csv'}, {out / 'wp_long.csv'}, {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
