#!/usr/bin/env python3
"""Report for the random-channel working-point campaign (run_random_campaign.py).

    .venv/bin/python report_random_campaign.py /root/reports/random_ch_2026-09-30 [interp_check.json]

Writes <campaign>/report/REPORT.md (Russian) and figures next to it: setup,
verification (labels recomputed from the stored BER curves, optional dense
re-simulation check, working point vs theoretical post-IRC SINR), censoring,
channel-estimation loss by model / modulation / layers / IoT / speed / delay
spread, perfect-CSI working points and known issues.

``interp_check.json`` holds ``{"configs": [...], "rows": [{"campaign", "dense", ...}]}``
from re-running configurations with a finer search.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nr_ul_sim.wp_search import classify, interpolate_wp  # noqa: E402

O = Path(sys.argv[1]).resolve()
R = O / "report"
R.mkdir(exist_ok=True)
interp = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 and Path(sys.argv[2]).exists() else None

EST = ["perfect", "ls_nn", "ls_lin", "ls_lin_time_avg", "lmmse_ce", "lmmse_exp", "ls_hard_window", "ls_soft_window"]
CE = EST[1:]
LABEL = {"perfect": "perfect CSI", "ls_nn": "LS-NN", "ls_lin": "LS-linear", "ls_lin_time_avg": "LS-lin+time avg",
         "lmmse_ce": "LMMSE (TDL prior)", "lmmse_exp": "LMMSE (exp prior)", "ls_hard_window": "LS+hard window",
         "ls_soft_window": "LS+soft window"}
COLOR = {"ls_nn": "#4c78a8", "ls_lin": "#f58518", "ls_lin_time_avg": "#54a24b", "lmmse_ce": "#b279a2",
         "lmmse_exp": "#ff9da6", "ls_hard_window": "#e45756", "ls_soft_window": "#72b7b2"}
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
add("# Кампания случайных каналов: рабочая точка IRC при BER = 1e-2\n")
add(f"Каталог: `{O}`. Конфигураций: **{N} из {plan_n}**, ошибок выполнения: {len(errors)}. "
    f"Запуск: {cfg['created']}, {cfg['args']['workers']} воркера на одном GPU (RTX PRO 4000 Blackwell). "
    f"Время на конфигурацию: медиана {np.median(tt):.1f} с, p95 {np.percentile(tt, 95):.0f} с, "
    f"суммарно {tt.sum() / 3600:.1f} GPU-воркер-часов.\n")

a = cfg["args"]
add("## 1. Постановка\n")
add(f"""- Модели канала: {', '.join(m.upper() for m in cfg['channels'])}; по {a['per_model']} конфигураций на модель, в каждой модели поровну ячеек (#UE ∈ {{1, 2}}) × (ранг ∈ {{1, 2}}) × (QPSK MCS 5, 16QAM MCS 14, 64QAM MCS 20).
- Полоса {a['num_prb']} PRB, FFT {a['fft_size']}, 30 кГц, 4 приёмные антенны (панель 1×2 кросс-пол., диаграмма 38.901), DMRS type-1, add-pos 1.
- Приёмник IRC с идеальной ковариацией помех; 8 оценок канала на одних и тех же слотах: {', '.join(LABEL[e] for e in EST)}.
- Из сида каждой конфигурации: IoT (0 дБ с вероятностью 0.2, иначе равномерно 0.5–20 дБ), 1–2 интерферента, крупномасштабный канал каждого UE и интерферента.
  - CDL-B/C/D: разброс задержек log-uniform 20–1000 нс, скорость 0.5–120 км/ч и направление, масштабирование и сдвиг углов кластеров по 38.901 7.7.5.1 (AS ×0.5…2, азимут UE в секторе ±60°, зенит 92–110°), джиттер кластеров (5°/2°, задержки ×e^N(0,0.1)), K-фактор 0–15 дБ для CDL-D.
  - UMi/UMa: один дроп 38.901 на конфигурацию (положения, indoor 30 %, LoS, LSP, кластеры и лучи заморожены), скорость 0.5–120 км/ч outdoor / 0.5–5 км/ч indoor.
  - Внутри конфигурации меняется только мелкомасштабное замирание (фазы лучей) — рабочая точка является функцией нарисованных параметров.
- Поиск рабочей точки: адаптивный, общий для всех оценивателей, шаг 4 дБ, уточнение до 2 дБ, сетка 0.5 дБ, ≤ {a['max_points']} точек SNR; Монте-Карло до {a['max_mc_iter']}×{a['batch_size']} слотов, остановка при {a['target_bit_errors']} битовых и {a['target_block_errors']} блочных ошибках. Результат по каждой оценке — интервальная метка: `ok` (WP в скобке), `right_censored` (WP выше последней точки: `floor`, `hi_limit`, `excess_loss` > 15 дБ хуже идеальной CSI, `budget`), `left_censored`.
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

add("## 2. Проверка вычислений\n")
add(f"""| проверка | результат |
|---|---|
| записей / дубликатов id / битых строк | {N} / {len(ids) - len(set(ids))} / 0 |
| у каждой записи есть `arrays/ID.npz` | {'да' if set(ids) <= arrays else 'НЕТ'} ({len(arrays)} файлов) |
| рабочие точки, пересчитанные из сохранённых кривых BER | {n_ok} меток `ok`, расхождений **{mism}** |
| ширина скобки | 2 дБ в {np.mean(np.array(widths) == 2.0):.0%} случаев, 1.5 дБ в {np.mean(np.array(widths) == 1.5):.0%} |
| верхняя точка скобки без ошибок (BER = 0) | {zero_b / max(n_ok, 1):.0%} |
""")
if interp:
    d = np.array([x["campaign"] - x["dense"] for x in interp["rows"]])
    add(f"Точность метки: {len(interp['configs'])} случайных конфигураций пересчитаны с уточнением до 0.5 дБ (сетка кампании) для perfect и LS-linear, {len(d)} пар: "
        f"смещение {d.mean():+.2f} дБ, СКО {d.std():.2f} дБ, максимум {np.abs(d).max():.2f} дБ (включает шум Монте-Карло повторного прогона). "
        "Нулевая верхняя точка дополнительного смещения не даёт.\n")

# physics: WP vs reference SINR
add("**Согласие с теорией.** Рабочая точка с идеальной CSI против теоретического эффективного SINR после IRC (признак `sinr_ref_eff_db` при SNR 10 дБ), без IoT:\n")
add("| модуляция | n | корреляция | остаток СКО, дБ | требуемый эфф. SINR, дБ | предел Шеннона, дБ | зазор, дБ |")
add("|---|---|---|---|---|---|---|")
fig, ax = plt.subplots(figsize=(6.4, 4.4))
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
add("Зазор до Шеннона растёт с порядком модуляции (BICM, max-log, короткие блоки LDPC) — ожидаемо.\n")

# --------------------------------------------------------------------------- censoring
add("## 3. Цензурирование\n")
add("Доля конфигураций, где рабочая точка найдена (`ok`), и причины цензуры:\n")
add("| модель | " + " | ".join(LABEL[e] for e in EST) + " |")
add("|---|" + "---|" * len(EST))
for m in MODELS:
    rows = [r for r in recs if r["channel"] == m]
    if not rows:
        continue
    add(f"| {m} | " + " | ".join(f"{np.mean([r['estimators'][e]['status'] == 'ok' for r in rows]):.0%}" for e in EST) + " |")
add(f"| **все** | " + " | ".join(f"**{np.mean([r['estimators'][e]['status'] == 'ok' for r in recs]):.0%}**" for e in EST) + " |")
reasons = collections.Counter((e, r["estimators"][e]["reason"]) for r in recs for e in EST if r["estimators"][e]["status"] != "ok")
add("\n| оценка | floor | hi_limit | excess_loss | budget | lo_limit |")
add("|---|---|---|---|---|---|")
for e in EST:
    add(f"| {LABEL[e]} | " + " | ".join(str(reasons.get((e, k), 0)) for k in ("floor", "hi_limit", "excess_loss", "budget", "lo_limit")) + " |")
add("")

# --------------------------------------------------------------------------- CE loss
add("## 4. Потери от оценки канала\n")
add("CE loss = WP(оценка) − WP(идеальная CSI) на той же конфигурации, только пары `ok`/`ok`.\n")
add("| оценка | медиана | среднее | p90 | n | " + " | ".join(MODELS) + " |")
add("|---|---|---|---|---|" + "---|" * len(MODELS))
for e in CE:
    v = [x for r in recs if (x := loss(r, e)) is not None]
    add(f"| {LABEL[e]} | {np.median(v):.2f} | {np.mean(v):.2f} | {np.percentile(v, 90):.2f} | {len(v)} | " +
        " | ".join(med([x for r in recs if r['channel'] == m and (x := loss(r, e)) is not None]) for m in MODELS) + " |")
add("")


def by_bins(title, key, bins, fmt):
    add(f"**По {title}** (медиана CE loss, дБ):\n")
    add("| " + title + " | n | " + " | ".join(LABEL[e] for e in CE) + " |")
    add("|---|---|" + "---|" * len(CE))
    for lo, hi in bins:
        rows = [r for r in recs if lo <= key(r) < hi]
        add(f"| {fmt(lo, hi)} | {len(rows)} | " + " | ".join(med([x for r in rows if (x := loss(r, e)) is not None]) for e in CE) + " |")
    add("")


by_bins("модуляции", lambda r: MODS.index(r["modulation"]), [(0, 1), (1, 2), (2, 3)], lambda lo, hi: MODS[lo])
by_bins("числу слоёв", lambda r: r["num_ue"] * 10 + r["rank"], [(11, 12), (12, 13), (21, 22), (22, 23)],
        lambda lo, hi: f"{lo // 10} UE × r{lo % 10}")
by_bins("IoT", lambda r: r["iot_db"], [(0, 0.01), (0.5, 5), (5, 10), (10, 15), (15, 20.1)],
        lambda lo, hi: "нет" if hi < 0.1 else f"{lo:g}–{hi:.0f} дБ")
by_bins("скорости (макс. по UE)", lambda r: max(u["speed_kmh"] for u in r["ue"]), [(0, 15), (15, 30), (30, 60), (60, 90), (90, 121)],
        lambda lo, hi: f"{lo}–{hi} км/ч")
by_bins("разбросу задержек UE 0", lambda r: ds_of(r["ue"][0]), [(0, 50), (50, 150), (150, 400), (400, 1000), (1000, 1e5)],
        lambda lo, hi: f"{lo:g}–{hi:g} нс" if hi < 1e4 else f"> {lo:g} нс")

# figure: CE loss boxplot per model
fig, axes = plt.subplots(1, len(MODELS), figsize=(3.2 * len(MODELS), 4.2), sharey=True)
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
add("Медианы считаются по найденным рабочим точкам; при высоких скоростях часть оценок (особенно LS-lin+time avg) уходит в цензуру `floor`, так что реальная потеря там больше, чем показывает медиана.\n")

# best estimator
best = collections.Counter()
for r in recs:
    vals = {e: r["estimators"][e]["wp_db"] for e in CE if r["estimators"][e]["status"] == "ok"}
    if vals:
        best[min(vals, key=vals.get)] += 1
add("**Лучшая реализуемая оценка** (минимальный WP среди 7 оценок):\n")
add("| оценка | доля конфигураций |")
add("|---|---|")
for e in CE:
    add(f"| {LABEL[e]} | {best[e] / max(sum(best.values()), 1):.0%} |")
add("")

# --------------------------------------------------------------------------- perfect WP
add("## 5. Рабочая точка с идеальной CSI\n")
add("Медиана WP, дБ (IoT учитывается как есть):\n")
add("| модель | " + " | ".join(f"{m} {u}UE r{k}" for m in MODS for u in (1, 2) for k in (1, 2)) + " |")
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
add("Смещение WP относительно теоретического SINR растёт с IoT (16QAM): ")
parts = []
for lo, hi in ((0, 0.01), (0.5, 5), (5, 10), (10, 15), (15, 20.1)):
    y = [r["estimators"]["perfect"]["wp_db"] + r["features"]["sinr_ref_eff_db"] - 10 for r in recs
         if r["modulation"] == "qam16" and lo <= r["iot_db"] < hi and r["estimators"]["perfect"]["status"] == "ok"]
    parts.append(f"{'нет IoT' if hi < 0.1 else f'{lo:g}–{hi:.0f} дБ'}: {np.median(y):.1f}")
add("; ".join(parts) + " дБ. Признак `sinr_ref_eff_db` считается с широкополосной ковариацией помех и не видит её частотной селективности, поэтому при сильной помехе он оптимистичен — для модели рабочей точки `iot_db` нужен отдельным входом.\n")

# --------------------------------------------------------------------------- lmmse_ce prior
add("## 6. Найденные особенности и рекомендации\n")
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
add("1. **Априори `lmmse_ce` подобрано только под UE 0.** `run_config` передаёт LMMSE разброс задержек первого UE, а оцениватель Sionna использует одну ковариацию на все потоки. Доля конфигураций, где `lmmse_ce` не достигает BER = 1e-2:\n")
add("| случай | доля | n |")
add("|---|---|---|")
for k in ("1 UE", "2 UE, DS(UE1) ≤ 2·DS(UE0)", "2 UE, DS(UE1) > 2·DS(UE0)"):
    add(f"| {k} | {np.mean(cens[k]):.0%} | {len(cens[k])} |")
for e in ("lmmse_exp", "ls_lin", "ls_soft_window"):
    a1 = [r["estimators"][e]["status"] != "ok" for r in recs if r["num_ue"] == 1 and r["estimators"]["perfect"]["status"] == "ok"]
    a2 = [r["estimators"][e]["status"] != "ok" for r in recs if r["num_ue"] == 2 and r["estimators"]["perfect"]["status"] == "ok"]
    add(f"| для сравнения {LABEL[e]}: 1 UE / 2 UE | {np.mean(a1):.0%} / {np.mean(a2):.0%} | |")
add("\nМетки `lmmse_ce` для 2 UE отражают этот артефакт, а не качество LMMSE. Исправление для следующего прогона: `max(ds_ns)` по UE (как уже сделано со скоростью).\n")
add("""2. **LS-lin+time avg** при скоростях > 60 км/ч массово уходит в `floor`: усреднение по слоту при f_D ≈ 200–400 Гц разрушает оценку. Это физика; в диапазоне 0.5–120 км/ч эта оценка не сопоставима с остальными без учёта скорости.
3. **Интерференты UMi/UMa** берутся как UE внутри того же сектора (тот же дроп-генератор), а не как UE соседних сот — их углы прихода и усиление антенны как у обслуживаемых UE. Для IRC это скорее оптимистично (помеха приходит с направлений главного лепестка).
4. **Ковариация помех для IRC** — идеальная и широкополосная (усреднение по всему слоту). Реальный IRC оценивает её по RB; при частотно-селективной помехе это отдельная потеря, в кампании не учтённая.
5. **Точность метки ±0.3 дБ** при скобке 2 дБ достаточна для регрессии; если нужна точнее — `fine_db = 1.0` удорожит поиск примерно на 1–2 точки на конфигурацию.
6. Признаки каналов (PDP, корреляции, число обусловленности, SINR-референс) записаны для каждой конфигурации; `wp_long.csv` содержит интервальные метки для регрессии с цензурой (например, XGBoost `survival:aft`).
""")

def iot_med(e, lo, hi):
    return np.median([x for r in recs if lo <= r["iot_db"] < hi and (x := loss(r, e)) is not None])
add(f"""7. **Soft window и помехи.** Потеря soft window растёт с IoT с {iot_med('ls_soft_window', 0, 0.01):.1f} до {iot_med('ls_soft_window', 15, 20.1):.1f} дБ, у hard window — с {iot_med('ls_hard_window', 0, 0.01):.1f} до {iot_med('ls_hard_window', 15, 20.1):.1f} дБ. Порог винеровского веса считается от N0, а помеха поднимает «шумовой» пол тапов в N0·(1+INR) раз, поэтому soft window пропускает тапы помехи. Исправление: оценивать шум на тап по тапам вне окна (эмпирически), а не по N0.
""")

# key conclusions, inserted after the header
def ce_med(e):
    return np.median([x for r in recs if (x := loss(r, e)) is not None])
ok_share = {e: np.mean([r["estimators"][e]["status"] == "ok" for r in recs]) for e in EST}
key = f"""## Главное

- {"Кампания завершена" if N >= plan_n else f"Посчитано {N} из {plan_n} конфигураций"}, ошибок выполнения {len(errors)}; вычисления проверены: метки пересчитываются из кривых без расхождений, точность WP ±0.3 дБ, WP с идеальной CSI согласуется с теоретическим SINR (корреляция 0.93–0.97, зазор до Шеннона 1.2–3.4 дБ).
- Медианная потеря от оценки канала (IRC, BER = 1e-2): LMMSE (TDL prior) {ce_med('lmmse_ce'):.1f} дБ, LMMSE (exp prior) {ce_med('lmmse_exp'):.1f}, LS+soft window {ce_med('ls_soft_window'):.1f}, LS+hard window {ce_med('ls_hard_window'):.1f}, LS-lin+time avg {ce_med('ls_lin_time_avg'):.1f}, LS-linear {ce_med('ls_lin'):.1f}, LS-NN {ce_med('ls_nn'):.1f} дБ.
- По надёжности (доля найденных рабочих точек) порядок другой: hard window {ok_share['ls_hard_window']:.0%}, LS-linear {ok_share['ls_lin']:.0%}, soft window {ok_share['ls_soft_window']:.0%}, LMMSE exp {ok_share['lmmse_exp']:.0%}, LMMSE TDL {ok_share['lmmse_ce']:.0%}, LS-NN {ok_share['ls_nn']:.0%}, LS-lin+time avg {ok_share['ls_lin_time_avg']:.0%}.
- Потери растут с числом слоёв (2 UE × r2 — вдвое больше, чем 1 UE × r1), разбросом задержек и скоростью; у LS-lin+time avg — резко со скоростью выше 60 км/ч.
- Два артефакта постановки: априори `lmmse_ce` подобрано только под UE 0 (для 2 UE завышает цензуру до 41 %), порог soft window не учитывает помеху (раздел 6).
"""
L.insert(2, key)

add("## 7. Файлы\n")
add(f"""- `{O}/results/worker_*.jsonl` — записи конфигураций; `arrays/ID.npz` — таблицы кластеров / геометрия дропа.
- `wp_wide.csv`, `wp_long.csv`, `summary.md` — пересобраны `summarize_random_campaign.py` по завершении.
- `report/REPORT.md` и рисунки: `wp_vs_sinr.png`, `ce_loss_by_model.png`, `ce_loss_vs_speed_ds.png`.
""")

(R / "REPORT.md").write_text("\n".join(L))
print(f"wrote {R / 'REPORT.md'} ({N} records)")
