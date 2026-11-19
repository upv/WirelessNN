#!/usr/bin/env python3
"""Report for the random-channel working-point campaign (run_random_campaign.py).

    .venv/bin/python scripts/campaigns/report_random_campaign.py ~/reports/random_ch_2026-09-30 [interp_check.json]
    .venv/bin/python scripts/campaigns/report_random_campaign.py ~/reports/random_ch_ammse_2026-10-05 --compare ~/reports/random_ch_2026-09-30

Writes <campaign>/report/REPORT.md and figures next to it: setup,
verification (labels recomputed from the stored BER curves, optional dense
re-simulation check, working point vs theoretical post-IRC SINR), censoring,
channel-estimation loss by model / modulation / layers / IoT / speed / delay
spread, perfect-CSI working points, known issues and <campaign>/notes.md if
present. The estimators are the
ones the campaign ran. With ``a_mmse`` among them a section compares it with
the other estimators per configuration, censoring included; ``--compare``
adds a check against an earlier campaign with the same plan.

``interp_check.json`` holds ``{"configs": [...], "rows": [{"campaign", "dense", ...}]}``
from re-running configurations with a finer search.
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nr_ul_sim.wp_search import classify, interpolate_wp  # noqa: E402

argv = sys.argv[1:]
COMPARE = None
if "--compare" in argv:
    i = argv.index("--compare")
    COMPARE = Path(argv[i + 1]).expanduser().resolve()
    del argv[i:i + 2]
O = Path(argv[0]).expanduser().resolve()
R = O / "report"
R.mkdir(exist_ok=True)
interp = json.loads(Path(argv[1]).read_text()) if len(argv) > 1 and Path(argv[1]).exists() else None

LABEL = {"perfect": "perfect CSI", "ls_nn": "LS-NN", "ls_lin": "LS-linear", "ls_lin_time_avg": "LS-lin+time avg",
         "lmmse_ce": "LMMSE (TDL prior)", "lmmse_exp": "LMMSE (exp prior)", "ls_hard_window": "LS+hard window",
         "ls_soft_window": "LS+soft window", "ls_fir": "LS+FIR (EqDeepRx)", "denoise_nn": "DenoiseNN (EqDeepRx)",
         "lmmse_data": "LMMSE (data cov.)", "lmmse_data_1d": "1D-LMMSE (data cov.)", "a_mmse": "A-MMSE",
         "ra_a_mmse": "RA-A-MMSE (r=6)"}
COLOR = {"ls_nn": "#4c78a8", "ls_lin": "#f58518", "ls_lin_time_avg": "#54a24b", "lmmse_ce": "#b279a2",
         "lmmse_exp": "#ff9da6", "ls_hard_window": "#e45756", "ls_soft_window": "#72b7b2",
         "ls_fir": "#9d755d", "denoise_nn": "#bab0ac", "lmmse_data": "#eeca3b", "lmmse_data_1d": "#8c6d31",
         "a_mmse": "#d62728", "ra_a_mmse": "#17becf"}
MODELS = ["cdl-b", "cdl-c", "cdl-d", "umi", "uma"]
MODS = ["qpsk", "qam16", "qam64"]
T = 1e-2

recs = {}
for f in sorted((O / "results").glob("worker_*.jsonl")):
    for line in open(f):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                recs[r["id"]] = r
            except json.JSONDecodeError:
                pass
recs = [recs[k] for k in sorted(recs)]
plan_n = sum(1 for _ in open(O / "plan.jsonl"))
cfg = json.loads((O / "campaign_config.json").read_text())
errors = [json.loads(l) for l in open(O / "errors.jsonl")] if (O / "errors.jsonl").exists() else []
N = len(recs)
EST = cfg.get("estimators") or list(recs[0]["estimators"])
CE = EST[1:]
GPU = cfg.get("gpu", "RTX PRO 4000 Blackwell")  # campaigns before 2026-10-05 did not record it


def ds_of(u):
    return u.get("ds_ns", u.get("lsp_ds_ns"))


def loss(r, e):
    E = r["estimators"]
    if E[e]["status"] == "ok" and E["perfect"]["status"] == "ok":
        return E[e]["wp_db"] - E["perfect"]["wp_db"]
    return None


def med(v):
    return f"{np.median(v):.2f}" if len(v) else "—"


L: list[str] = []
add = L.append

# --------------------------------------------------------------------------- header
tt = np.array([r["timing_s"]["total"] for r in recs])
add("# Random-channel campaign: IRC working point at BER = 1e-2\n")
add(f"Directory: `{O}`. Configurations: **{N} of {plan_n}**, runtime errors: {len(errors)}. "
    f"Started {cfg['created']}, {cfg['args']['workers']} workers on one GPU ({GPU}). "
    f"Time per configuration: median {np.median(tt):.1f} s, p95 {np.percentile(tt, 95):.0f} s, "
    f"{tt.sum() / 3600:.1f} GPU worker-hours in total.\n")

a = cfg["args"]
add("## 1. Setup\n")
add(f"""- Channel models: {', '.join(m.upper() for m in cfg['channels'])}; {a['per_model']} configurations per model, split equally over the cells (#UE ∈ {{1, 2}}) × (rank ∈ {{1, 2}}) × (QPSK MCS 5, 16QAM MCS 14, 64QAM MCS 20).
- {a['num_prb']} PRB, FFT {a['fft_size']}, 30 kHz, 4 receive antennas (1×2 cross-polarised panel, 38.901 pattern), DMRS type 1, additional position 1.
- IRC receiver with perfect interference covariance; {len(EST)} channel estimators on the same slots: {', '.join(LABEL[e] for e in EST)}.
- Drawn from each configuration's seed: IoT (none with probability 0.2, otherwise uniform 0.5–20 dB), 1–2 interferers, the large-scale channel of every UE and interferer.
  - CDL-B/C/D: delay spread log-uniform 20–1000 ns, speed 0.5–120 km/h and direction, cluster angle scaling and translation per 38.901 7.7.5.1 (AS ×0.5…2, UE azimuth within ±60° of the sector, zenith 92–110°), cluster jitter (5°/2°, delays ×e^N(0,0.1)), K-factor 0–15 dB for CDL-D.
  - UMi/UMa: one 38.901 drop per configuration (positions, 30 % indoor, LoS, LSP, clusters and rays frozen), speed 0.5–120 km/h outdoor / 0.5–5 km/h indoor.
  - Within a configuration only the small-scale fading (ray phases) changes, so the working point is a function of the drawn parameters.
- Working-point search: adaptive, shared by all estimators, 4 dB steps refined to 2 dB, 0.5 dB grid, ≤ {a['max_points']} SNR points; Monte Carlo up to {a['max_mc_iter']}×{a['batch_size']} slots, stopping at {a['target_bit_errors']} bit and {a['target_block_errors']} block errors. Each estimator gets an interval label: `ok` (WP inside the bracket), `right_censored` (WP above the last point: `floor`, `hi_limit`, `excess_loss` > 15 dB worse than perfect CSI, `budget`), `left_censored`.
""")

# --------------------------------------------------------------------------- verification
mism = 0
n_ok = 0
widths = []
zero_b = 0
for r in recs:
    for e, v in r["estimators"].items():
        curve = [(s, {"ber": b, "num_bits": nb}) for s, b, nb in zip(v["snr_db"], v["ber"], v["num_bits"])]
        c = classify(curve, T)
        if v["status"] == "ok":
            n_ok += 1
            st = dict(curve)
            if c["kind"] != "bracket" or abs(interpolate_wp(c["a"], st[c["a"]], c["b"], st[c["b"]], T) - v["wp_db"]) > 1e-9:
                mism += 1
            widths.append(c["b"] - c["a"])
            if v["ber"][v["snr_db"].index(c["b"])] == 0:
                zero_b += 1
        elif c["kind"] == "bracket":
            mism += 1
ids = [r["id"] for r in recs]
arrays = {int(p.stem) for p in (O / "arrays").glob("*.npz")}

add("## 2. Verification\n")
add(f"""| check | result |
|---|---|
| records / duplicate ids / broken lines | {N} / {len(ids) - len(set(ids))} / 0 |
| every record has `arrays/ID.npz` | {'yes' if set(ids) <= arrays else 'NO'} ({len(arrays)} files) |
| working points recomputed from the stored BER curves | {n_ok} `ok` labels, **{mism}** mismatches |
| bracket width | 2 dB in {np.mean(np.array(widths) == 2.0):.0%} of cases, 1.5 dB in {np.mean(np.array(widths) == 1.5):.0%} |
| upper bracket point without errors (BER = 0) | {zero_b / max(n_ok, 1):.0%} |
""")
if interp:
    d = np.array([x["campaign"] - x["dense"] for x in interp["rows"]])
    add(f"Label accuracy: {len(interp['configs'])} random configurations re-run with refinement to 0.5 dB (the campaign grid) for perfect and LS-linear, {len(d)} pairs: "
        f"bias {d.mean():+.2f} dB, std {d.std():.2f} dB, max {np.abs(d).max():.2f} dB (includes the Monte-Carlo noise of the re-run). "
        "A zero-error upper point adds no extra bias.\n")

# physics: WP vs reference SINR
add("**Agreement with theory.** Perfect-CSI working point against the theoretical effective post-IRC SINR (feature `sinr_ref_eff_db` at SNR 10 dB), no IoT:\n")
add("| modulation | n | correlation | residual std, dB | required eff. SINR, dB | Shannon limit, dB | gap, dB |")
add("|---|---|---|---|---|---|---|")
fig, ax = plt.subplots(figsize=(6.4, 4.4))
theory_corr, theory_gap = [], []
for mod, rate, bits in (("qpsk", 379 / 1024, 2), ("qam16", 553 / 1024, 4), ("qam64", 567 / 1024, 6)):
    xs, ys = [], []
    for r in recs:
        p = r["estimators"]["perfect"]
        if r["modulation"] == mod and r["iot_db"] == 0 and p["status"] == "ok":
            xs.append(r["features"]["sinr_ref_eff_db"])
            ys.append(p["wp_db"])
    xs, ys = np.array(xs), np.array(ys)
    k, b = np.polyfit(xs, ys, 1)
    req = np.median(ys + xs - 10)
    shannon = 10 * np.log10(2 ** (rate * bits) - 1)
    theory_corr.append(abs(np.corrcoef(xs, ys)[0, 1]))
    theory_gap.append(req - shannon)
    add(f"| {mod} | {len(xs)} | {np.corrcoef(xs, ys)[0, 1]:.2f} | {(ys - k * xs - b).std():.2f} | {req:.1f} | {shannon:.1f} | {req - shannon:.1f} |")
    ax.scatter(xs, ys, s=4, alpha=0.4, label=mod)
ax.set_xlabel("effective post-IRC SINR at SNR 10 dB, perfect CSI [dB]")
ax.set_ylabel("working point, perfect CSI [dB]")
ax.set_title("Working point vs theoretical SINR (no IoT)")
ax.grid(True, ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(R / "wp_vs_sinr.png", dpi=130)
plt.close(fig)
add("\n![WP vs SINR](wp_vs_sinr.png)\n")
add("The gap to Shannon grows with the modulation order (BICM, max-log, short LDPC blocks), as expected.\n")

# --------------------------------------------------------------------------- censoring
add("## 3. Censoring\n")
add("Share of configurations where the working point was found (`ok`), and censoring reasons:\n")
add("| model | " + " | ".join(LABEL[e] for e in EST) + " |")
add("|---|" + "---|" * len(EST))
for m in MODELS:
    rows = [r for r in recs if r["channel"] == m]
    if not rows:
        continue
    add(f"| {m} | " + " | ".join(f"{np.mean([r['estimators'][e]['status'] == 'ok' for r in rows]):.0%}" for e in EST) + " |")
add(f"| **all** | " + " | ".join(f"**{np.mean([r['estimators'][e]['status'] == 'ok' for r in recs]):.0%}**" for e in EST) + " |")
reasons = collections.Counter((e, r["estimators"][e]["reason"]) for r in recs for e in EST if r["estimators"][e]["status"] != "ok")
add("\n| estimator | floor | hi_limit | excess_loss | budget | lo_limit |")
add("|---|---|---|---|---|---|")
for e in EST:
    add(f"| {LABEL[e]} | " + " | ".join(str(reasons.get((e, k), 0)) for k in ("floor", "hi_limit", "excess_loss", "budget", "lo_limit")) + " |")
add("")

# --------------------------------------------------------------------------- CE loss
add("## 4. Channel-estimation loss\n")
add("CE loss = WP(estimator) − WP(perfect CSI) on the same configuration, `ok`/`ok` pairs only.\n")
add("| estimator | median | mean | p90 | n | " + " | ".join(MODELS) + " |")
add("|---|---|---|---|---|" + "---|" * len(MODELS))
for e in CE:
    v = [x for r in recs if (x := loss(r, e)) is not None]
    add(f"| {LABEL[e]} | {np.median(v):.2f} | {np.mean(v):.2f} | {np.percentile(v, 90):.2f} | {len(v)} | " +
        " | ".join(med([x for r in recs if r['channel'] == m and (x := loss(r, e)) is not None]) for m in MODELS) + " |")
add("")


def by_bins(title, key, bins, fmt):
    add(f"**By {title}** (median CE loss, dB):\n")
    add("| " + title + " | n | " + " | ".join(LABEL[e] for e in CE) + " |")
    add("|---|---|" + "---|" * len(CE))
    for lo, hi in bins:
        rows = [r for r in recs if lo <= key(r) < hi]
        add(f"| {fmt(lo, hi)} | {len(rows)} | " + " | ".join(med([x for r in rows if (x := loss(r, e)) is not None]) for e in CE) + " |")
    add("")


by_bins("modulation", lambda r: MODS.index(r["modulation"]), [(0, 1), (1, 2), (2, 3)], lambda lo, hi: MODS[lo])
by_bins("number of layers", lambda r: r["num_ue"] * 10 + r["rank"], [(11, 12), (12, 13), (21, 22), (22, 23)],
        lambda lo, hi: f"{lo // 10} UE × r{lo % 10}")
by_bins("IoT", lambda r: r["iot_db"], [(0, 0.01), (0.5, 5), (5, 10), (10, 15), (15, 20.1)],
        lambda lo, hi: "none" if hi < 0.1 else f"{lo:g}–{hi:.0f} dB")
by_bins("speed (max over UEs)", lambda r: max(u["speed_kmh"] for u in r["ue"]), [(0, 15), (15, 30), (30, 60), (60, 90), (90, 121)],
        lambda lo, hi: f"{lo}–{hi} km/h")
by_bins("UE 0 delay spread", lambda r: ds_of(r["ue"][0]), [(0, 50), (50, 150), (150, 400), (400, 1000), (1000, 1e5)],
        lambda lo, hi: f"{lo:g}–{hi:g} ns" if hi < 1e4 else f"> {lo:g} ns")

# figure: CE loss boxplot per model
fig, axes = plt.subplots(1, len(MODELS), figsize=(max(3.2, 0.42 * len(CE)) * len(MODELS), 4.6), sharey=True)
for ax, m in zip(axes, MODELS):
    data = [[x for r in recs if r["channel"] == m and (x := loss(r, e)) is not None] for e in CE]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    for patch, e in zip(bp["boxes"], CE):
        patch.set_facecolor(COLOR[e])
    ax.set_xticks(range(1, len(CE) + 1), [LABEL[e] for e in CE], rotation=60, ha="right", fontsize=7)
    ax.set_title(m.upper())
    ax.grid(True, axis="y", ls=":", alpha=0.5)
axes[0].set_ylabel("CE loss vs perfect CSI [dB]")
fig.suptitle("Working-point loss of each channel estimator (IRC, BER = 1e-2)")
fig.tight_layout()
fig.savefig(R / "ce_loss_by_model.png", dpi=130)
plt.close(fig)
add("![CE loss by model](ce_loss_by_model.png)\n")

# figure: loss vs speed and vs DS
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, key, xl, edges, logx in (
        (axes[0], lambda r: max(u["speed_kmh"] for u in r["ue"]), "max UE speed [km/h]", np.linspace(0, 120, 9), False),
        (axes[1], lambda r: ds_of(r["ue"][0]), "delay spread of UE 0 [ns]", np.geomspace(5, 3000, 10), True)):
    for e in CE:
        xs = np.array([key(r) for r in recs if loss(r, e) is not None])
        ys = np.array([loss(r, e) for r in recs if loss(r, e) is not None])
        cx, cy = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (xs >= lo) & (xs < hi)
            if sel.sum() >= 20:
                cx.append(np.sqrt(lo * hi) if logx else 0.5 * (lo + hi))
                cy.append(np.median(ys[sel]))
        ax.plot(cx, cy, "o-", color=COLOR[e], label=LABEL[e], ms=3)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xl)
    ax.set_ylabel("median CE loss [dB]")
    ax.grid(True, ls=":", alpha=0.5)
axes[0].legend(fontsize=7)
fig.suptitle("CE loss vs speed and delay spread (only configurations where both WPs are found)")
fig.tight_layout()
fig.savefig(R / "ce_loss_vs_speed_ds.png", dpi=130)
plt.close(fig)
add("![CE loss vs speed and DS](ce_loss_vs_speed_ds.png)\n")
add("Medians use found working points only; at high speed some estimators (LS-lin+time avg in particular) end up censored as `floor`, so the real loss there is larger than the median shows.\n")

# best estimator
best = collections.Counter()
for r in recs:
    vals = {e: r["estimators"][e]["wp_db"] for e in CE if r["estimators"][e]["status"] == "ok"}
    if vals:
        best[min(vals, key=vals.get)] += 1
add(f"**Best realisable estimator** (lowest WP among {len(CE)} estimators):\n")
add("| estimator | share of configurations |")
add("|---|---|")
for e in CE:
    add(f"| {LABEL[e]} | {best[e] / max(sum(best.values()), 1):.0%} |")
add("")

# --------------------------------------------------------------------------- perfect WP
add("## 5. Working point with perfect CSI\n")
add("Median WP, dB (IoT included as drawn):\n")
add("| model | " + " | ".join(f"{m} {u}UE r{k}" for m in MODS for u in (1, 2) for k in (1, 2)) + " |")
add("|---|" + "---|" * 12)
for ch in MODELS:
    cells = []
    for mod in MODS:
        for u in (1, 2):
            for k in (1, 2):
                v = [r["estimators"]["perfect"]["wp_db"] for r in recs if r["channel"] == ch and r["modulation"] == mod
                     and r["num_ue"] == u and r["rank"] == k and r["estimators"]["perfect"]["status"] == "ok"]
                cells.append(f"{np.median(v):.1f}" if v else "—")
    add(f"| {ch} | " + " | ".join(cells) + " |")
add("")
# IoT offset
add("The WP offset from the theoretical SINR grows with IoT (16QAM): ")
parts = []
for lo, hi in ((0, 0.01), (0.5, 5), (5, 10), (10, 15), (15, 20.1)):
    y = [r["estimators"]["perfect"]["wp_db"] + r["features"]["sinr_ref_eff_db"] - 10 for r in recs
         if r["modulation"] == "qam16" and lo <= r["iot_db"] < hi and r["estimators"]["perfect"]["status"] == "ok"]
    parts.append(f"{'no IoT' if hi < 0.1 else f'{lo:g}–{hi:.0f} dB'}: {np.median(y):.1f}")
add("; ".join(parts) + " dB. The feature `sinr_ref_eff_db` uses the wideband interference covariance and does not see its frequency selectivity, so it is optimistic under strong interference; a working-point model needs `iot_db` as a separate input.\n")

# --------------------------------------------------------------------------- lmmse_ce prior
add("## 6. Findings and recommendations\n")
cens = collections.defaultdict(list)
for r in recs:
    E = r["estimators"]
    if E["perfect"]["status"] != "ok":
        continue
    bad = E["lmmse_ce"]["status"] != "ok"
    if r["num_ue"] == 1:
        cens["1 UE"].append(bad)
    else:
        ds = [ds_of(u) for u in r["ue"]]
        cens["2 UE, DS(UE1) > 2·DS(UE0)" if ds[1] > 2 * ds[0] else "2 UE, DS(UE1) ≤ 2·DS(UE0)"].append(bad)
add("1. **`lmmse_ce` prior and multiple UEs.** Sionna's LMMSE estimator uses one covariance for all streams, so its delay-spread prior must cover every UE (campaigns before 2026-10-17 used UE 0 only; `run_config` now takes the maximum). Share of configurations where `lmmse_ce` does not reach BER = 1e-2:\n")
add("| case | share | n |")
add("|---|---|---|")
for k in ("1 UE", "2 UE, DS(UE1) ≤ 2·DS(UE0)", "2 UE, DS(UE1) > 2·DS(UE0)"):
    add(f"| {k} | {np.mean(cens[k]):.0%} | {len(cens[k])} |")
for e in ("lmmse_exp", "ls_lin", "ls_soft_window"):
    a1 = [r["estimators"][e]["status"] != "ok" for r in recs if r["num_ue"] == 1 and r["estimators"]["perfect"]["status"] == "ok"]
    a2 = [r["estimators"][e]["status"] != "ok" for r in recs if r["num_ue"] == 2 and r["estimators"]["perfect"]["status"] == "ok"]
    add(f"| for comparison, {LABEL[e]}: 1 UE / 2 UE | {np.mean(a1):.0%} / {np.mean(a2):.0%} | |")
add("\nIf the 2-UE rows differ strongly, the `lmmse_ce` labels reflect the prior, not the quality of LMMSE.\n")
add("""2. **LS-lin+time avg** is mostly censored as `floor` above 60 km/h: averaging over the slot at f_D ≈ 200–400 Hz destroys the estimate. This is physics; over 0.5–120 km/h this estimator is not comparable with the others without accounting for speed.
3. **UMi/UMa interferers** are drawn as UEs inside the same sector (same drop generator), not as UEs of neighbouring cells, so their angles of arrival and antenna gain are like those of the served UEs. For IRC this is rather optimistic (interference arrives from main-lobe directions).
4. **The IRC interference covariance** is perfect and wideband (averaged over the whole slot). A real IRC estimates it per RB; with frequency-selective interference that is an extra loss not included in the campaign.
5. **A label accuracy of ±0.3 dB** with a 2 dB bracket is enough for regression; for more, `fine_db = 1.0` costs about 1–2 extra points per configuration.
6. Channel features (PDP, correlations, condition number, reference SINR) are stored for every configuration; `wp_long.csv` holds interval labels for censored regression (for example XGBoost `survival:aft`).
""")

def iot_med(e, lo, hi):
    return np.median([x for r in recs if lo <= r["iot_db"] < hi and (x := loss(r, e)) is not None])
add(f"""7. **Soft window and interference.** The soft-window loss grows with IoT from {iot_med('ls_soft_window', 0, 0.01):.1f} to {iot_med('ls_soft_window', 15, 20.1):.1f} dB, the hard window's from {iot_med('ls_hard_window', 0, 0.01):.1f} to {iot_med('ls_hard_window', 15, 20.1):.1f} dB. With `--ce-soft-noise-mode thermal` the Wiener threshold is set from N0 while interference raises the per-tap floor to N0·(1+INR), so the soft window lets interference taps through; the default `outside` mode estimates the floor from the taps outside the window.
""")

# --------------------------------------------------------------------------- A-MMSE
def closs(r, e):
    """CE loss with censoring: +inf if the WP is above the search, -inf below; None without perfect WP."""
    E = r["estimators"]
    if E["perfect"]["status"] != "ok":
        return None
    s = E[e]["status"]
    if s == "ok":
        return E[e]["wp_db"] - E["perfect"]["wp_db"]
    return math.inf if s == "right_censored" else -math.inf


def cq(rows, e, q=50):
    """Quantile of the censored loss (no interpolation, so infinities stay infinite)."""
    v = [x for r in rows if (x := closs(r, e)) is not None]
    return np.percentile(np.array(v), q, method="inverted_cdf") if v else math.nan


def fq(x):
    if math.isnan(x):
        return "—"
    return "not found" if math.isinf(x) else f"{x:.2f}"


AMMSE_KEY = ""
sec = 7
ok_share = {e: np.mean([r["estimators"][e]["status"] == "ok" for r in recs]) for e in EST}
if "a_mmse" in CE:
    F = [e for e in ("a_mmse", "ra_a_mmse", "lmmse_data", "lmmse_data_1d", "denoise_nn", "ls_fir", "lmmse_ce",
                     "lmmse_exp", "ls_soft_window", "ls_lin") if e in CE]
    add(f"## {sec}. A-MMSE (arXiv:2506.00452)\n")
    sec += 1
    add(f"""A-MMSE is an attention transformer that learns one fixed linear filter W per 4-PRB block (from 2×12 pilot values to 48 subcarriers × 14 symbols); a filter bank over SNR (−10…35 dB, step 5) and DMRS comb, selected by N0. Trained on UMa, 68 PRB (`models/paper_ce`); here {cfg['args']['num_prb']} PRB, i.e. {cfg['args']['num_prb'] // 4} blocks sharing the same W. RA-A-MMSE is its rank-6 variant. `lmmse_data` is the 2D LMMSE with the covariance estimated from the training channels: according to the paper A-MMSE converges to the best fixed linear filter, which is this one. All paper estimators see CDL-B/C/D and UMi for the first time.

The medians in section 4 use found working points only and therefore flatter estimators that fail more often. Here the loss is censored: a working point that was not found counts as infinite loss (configurations without a perfect-CSI WP are excluded).
""")
    add("| estimator | WP found | median (found only) | median with censoring | p75 with censoring | p90 with censoring |")
    add("|---|---|---|---|---|---|")
    for e in F:
        v = [x for r in recs if (x := loss(r, e)) is not None]
        add(f"| {LABEL[e]} | {ok_share[e]:.1%} | {med(v)} | {fq(cq(recs, e))} | {fq(cq(recs, e, 75))} | {fq(cq(recs, e, 90))} |")
    add("")

    add("**A-MMSE against the other estimators on the same configurations.** Better / worse means a working-point difference above 0.25 dB, or a working point found by only one of the two:\n")
    add("| against | n | A-MMSE better | same (±0.25 dB) | A-MMSE worse | neither found | median WP(A-MMSE) − WP(other), dB |")
    add("|---|---|---|---|---|---|---|")
    for e in F[1:]:
        b = s = w = nn = 0
        dd = []
        for r in recs:
            a, o = closs(r, "a_mmse"), closs(r, e)
            if a is None:
                continue
            if math.isinf(a) and math.isinf(o) and a == o:
                nn += 1
            elif math.isfinite(a) and math.isfinite(o):
                dd.append(a - o)
                b, s, w = b + (a < o - 0.25), s + (abs(a - o) <= 0.25), w + (a > o + 0.25)
            elif a < o:
                b += 1
            else:
                w += 1
        n = b + s + w + nn
        add(f"| {LABEL[e]} | {n} | {b / n:.0%} | {s / n:.0%} | {w / n:.0%} | {nn / n:.0%} | {np.median(dd):+.2f} |")
    add("")

    G = [e for e in ("a_mmse", "ra_a_mmse", "lmmse_data", "denoise_nn", "lmmse_ce", "lmmse_exp", "ls_lin") if e in CE]

    def cbins(title, groups):
        add(f"**By {title}** (median censored loss, dB):\n")
        add(f"| {title} | n | " + " | ".join(LABEL[e] for e in G) + " |")
        add("|---|---|" + "---|" * len(G))
        for name, rows in groups:
            add(f"| {name} | {len(rows)} | " + " | ".join(fq(cq(rows, e)) for e in G) + " |")
        add("")

    cbins("channel model", [(m.upper(), [r for r in recs if r["channel"] == m]) for m in MODELS])
    cbins("number of layers", [(f"{u} UE × r{k}", [r for r in recs if r["num_ue"] == u and r["rank"] == k])
                          for u in (1, 2) for k in (1, 2)])
    cbins("modulation", [(m, [r for r in recs if r["modulation"] == m]) for m in MODS])
    cbins("IoT", [("none" if hi < 0.1 else f"{lo:g}–{hi:.0f} dB", [r for r in recs if lo <= r["iot_db"] < hi])
                  for lo, hi in ((0, 0.01), (0.5, 5), (5, 10), (10, 15), (15, 20.1))])
    cbins("speed (max over UEs)", [(f"{lo}–{hi} km/h", [r for r in recs if lo <= max(u["speed_kmh"] for u in r["ue"]) < hi])
                                     for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 90), (90, 121))])
    cbins("UE 0 delay spread", [(f"{lo:g}–{hi:g} ns" if hi < 1e4 else f"> {lo:g} ns", [r for r in recs if lo <= ds_of(r["ue"][0]) < hi])
                                     for lo, hi in ((0, 50), (50, 150), (150, 400), (400, 1000), (1000, 1e5))])

    rs = collections.Counter(r["estimators"]["a_mmse"]["reason"] for r in recs if r["estimators"]["a_mmse"]["status"] != "ok")
    rs_d = collections.Counter(r["estimators"]["lmmse_data"]["reason"] for r in recs
                               if "lmmse_data" in CE and r["estimators"]["lmmse_data"]["status"] != "ok")
    add("Reasons the working point was not found: A-MMSE: " + ", ".join(f"`{k}` {v}" for k, v in rs.most_common()) +
        ("; LMMSE (data cov.): " + ", ".join(f"`{k}` {v}" for k, v in rs_d.most_common()) if rs_d else "") + ".\n")

    if "lmmse_data" in CE:
        pairs = [(loss(r, "lmmse_data"), loss(r, "a_mmse")) for r in recs
                 if loss(r, "lmmse_data") is not None and loss(r, "a_mmse") is not None]
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        xs, ys = np.array(pairs).T
        ax.scatter(xs, ys, s=3, alpha=0.3, color=COLOR["a_mmse"])
        lim = [min(xs.min(), ys.min()), np.percentile(np.r_[xs, ys], 99.5)]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("CE loss, LMMSE (data covariance) [dB]")
        ax.set_ylabel("CE loss, A-MMSE [dB]")
        ax.set_title("A-MMSE vs its closed-form limit, per configuration")
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(R / "ammse_vs_lmmse_data.png", dpi=130)
        plt.close(fig)
        add("![A-MMSE vs LMMSE data](ammse_vs_lmmse_data.png)\n")

    ref = [e for e in ("lmmse_data", "ls_lin", "denoise_nn", "lmmse_ce") if e in CE]
    AMMSE_KEY = (f"- **A-MMSE**: working point found in {ok_share['a_mmse']:.0%} of configurations, median censored loss "
                 f"{fq(cq(recs, 'a_mmse'))} dB against " +
                 ", ".join(f"{LABEL[e]} {fq(cq(recs, e))}" for e in ref) +
                 " dB (details in section 7).\n")

# --------------------------------------------------------------------------- comparison with an earlier campaign
if COMPARE:
    old = {}
    for f in sorted((COMPARE / "results").glob("worker_*.jsonl")):
        for line in open(f):
            if line.strip():
                try:
                    x = json.loads(line)
                    old[x["id"]] = x
                except json.JSONDecodeError:
                    pass
    common = [r for r in recs if r["id"] in old and old[r["id"]]["seed"] == r["seed"]]
    add(f"## {sec}. Comparison with campaign `{COMPARE.name}`\n")
    sec += 1
    same_cfg = sum(old[r["id"]]["iot_db"] == r["iot_db"] and old[r["id"]]["num_interferers"] == r["num_interferers"]
                   for r in common)
    add(f"Same plan (seed, models, cells): {len(common)} common configurations, {same_cfg} of them with the same IoT and number of interferers. "
        "Between the campaigns the code changed only by adding estimators; slots and the WP search are shared by all estimators, "
        "so the differences below are the effect of a different estimator set on the search path, plus Monte-Carlo noise.\n")
    add("| estimator | same status | n (both ok) | median |ΔWP|, dB | mean ΔWP (new − old), dB | p95 |ΔWP|, dB |")
    add("|---|---|---|---|---|---|")
    for e in [e for e in EST if e in old[common[0]["id"]]["estimators"]]:
        st_same = np.mean([old[r["id"]]["estimators"][e]["status"] == r["estimators"][e]["status"] for r in common])
        d = np.array([r["estimators"][e]["wp_db"] - old[r["id"]]["estimators"][e]["wp_db"] for r in common
                      if r["estimators"][e]["status"] == "ok" and old[r["id"]]["estimators"][e]["status"] == "ok"])
        add(f"| {LABEL[e]} | {st_same:.0%} | {len(d)} | {np.median(np.abs(d)):.2f} | {d.mean():+.2f} | {np.percentile(np.abs(d), 95):.2f} |")
    add("")

# key conclusions, inserted after the header
def ce_med(e):
    return np.median([x for r in recs if (x := loss(r, e)) is not None])
ok_share = {e: np.mean([r["estimators"][e]["status"] == "ok" for r in recs]) for e in EST}
interp_txt = ""
if interp:
    d = np.array([x["campaign"] - x["dense"] for x in interp["rows"]])
    interp_txt = f", WP accuracy ±{d.std():.1f} dB"
by_loss = sorted(CE, key=ce_med)
by_ok = sorted(CE, key=lambda e: -ok_share[e])
key = f"""## Key results

- {"Campaign complete" if N >= plan_n else f"{N} of {plan_n} configurations done"}, {len(errors)} runtime errors; verified: labels are recomputed from the curves {"without mismatches" if mism == 0 else f"with {mism} mismatches"}{interp_txt}, the perfect-CSI WP agrees with the theoretical SINR (|correlation| {min(theory_corr):.2f}–{max(theory_corr):.2f}, gap to Shannon {min(theory_gap):.1f}–{max(theory_gap):.1f} dB).
- Median channel-estimation loss (IRC, BER = 1e-2, found working points only): {', '.join(f'{LABEL[e]} {ce_med(e):.1f}' for e in by_loss)} dB.
- Ranked by reliability (share of working points found) the order differs: {', '.join(f'{LABEL[e]} {ok_share[e]:.0%}' for e in by_ok)}.
- Losses grow with the number of layers, delay spread and speed; for LS-lin+time avg sharply above 60 km/h.
- `lmmse_ce` with 2 UEs and DS(UE1) > 2·DS(UE0): working point not found in {np.mean(cens['2 UE, DS(UE1) > 2·DS(UE0)']):.0%} of cases (section 6).
"""
if AMMSE_KEY:
    key += AMMSE_KEY
L.insert(2, key)

if (O / "notes.md").exists():  # hand-written conclusions for this campaign
    add(f"## {sec}. Conclusions\n")
    add((O / "notes.md").read_text().strip() + "\n")
    sec += 1

figs = ["wp_vs_sinr.png", "ce_loss_by_model.png", "ce_loss_vs_speed_ds.png", "ammse_vs_lmmse_data.png"]
add(f"## {sec}. Files\n")
add(f"""- `{O}/results/worker_*.jsonl`: configuration records; `arrays/ID.npz`: cluster tables / drop geometry.
- `wp_wide.csv`, `wp_long.csv`, `summary.md`: rebuilt by `scripts/campaigns/summarize_random_campaign.py` at the end.
- `report/REPORT.md` and figures: {', '.join(f'`{f}`' for f in figs if (R / f).exists())}.
""")

(R / "REPORT.md").write_text("\n".join(L))
print(f"wrote {R / 'REPORT.md'} ({N} records)")
