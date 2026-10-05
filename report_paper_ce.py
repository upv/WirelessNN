#!/usr/bin/env python3
"""Report on the paper channel estimators: training, NMSE, link-level campaign.

    .venv/bin/python report_paper_ce.py /root/reports/paper_ce [models/paper_ce]

Reads <dir>/nmse/nmse.json, <dir>/campaigns/{ce,incm}/*/*.json and
<models>/train_meta.json; writes <dir>/REPORT.md (Russian) and figures.
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
from nr_ul_sim.plotting import RECEIVER_STYLE  # noqa: E402

D = Path(sys.argv[1]).resolve()
M = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).resolve().parent / "models" / "paper_ce"
NEW = ["ls_fir", "denoise_nn", "lmmse_data", "lmmse_data_1d", "a_mmse", "ra_a_mmse"]
OLD = ["ls_nn", "ls_lin", "ls_lin_time_avg", "lmmse_exp", "ls_hard_window", "ls_soft_window"]
CAMP_EST = ["ls_lin", "ls_soft_window", "lmmse_exp", "ls_fir", "denoise_nn", "lmmse_data", "lmmse_data_1d", "a_mmse", "ra_a_mmse"]
CH = ["uma", "umi", "cdl-b", "cdl-c"]
MODS = ["qpsk", "qam16", "qam64"]
CFGS = [("1", "1"), ("2", "1"), ("2", "2")]


def label(e):
    return RECEIVER_STYLE.get(e, {}).get("label", e)


def f(v, nd=1):
    return "n/r" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


L: list[str] = []
add = L.append
add("# Оценщики канала из EqDeepRx и A-MMSE в симуляторе WirelessNN\n")
add("""Статьи:
- **EqDeepRx** — M. Honkala, D. Korpi, E. Raninen, J. Huttunen, *EqDeepRx: Learning a Scalable MIMO Receiver*, arXiv:2602.11834 (Nokia Bell Labs).
- **A-MMSE** — T. Ha, C. Jung, H. Kim, J. Park, J. Park, *Learning MMSE Filters for OFDM Channel Estimation: Attention Transformer Gains at Linear Inference*, arXiv:2506.00452v5.

Код: `nr_ul_sim/paper_ce.py` (оценщики), `nr_ul_sim/interference.py` (INCM), `train_paper_ce.py` (обучение), `eval_paper_ce.py` (NMSE), `run_paper_ce_campaign.sh` (BER/BLER), `tests/test_paper_ce.py`.
""")

add("## 1. Что взято из статей и как перенесено на PUSCH 68 PRB\n")
add("""| имя | статья | метод | адаптация |
|---|---|---|---|
| `ls_fir` | EqDeepRx, Sec. II-B (baseline) | LS на пилотах → статический фильтр сглаживания по частоте → линейная интерполяция | КИХ 17 отводов (Ханн × sinc), полоса задержек [−1, +3] мкс, нормировка на краях полосы |
| `denoise_nn` | EqDeepRx, Sec. III-A | LS → DenoiseNN (1D-свёртки по частоте с даунсэмплингом, 1×1-миксер по DMRS-символам, на каждую пару RX-антенна/слой) → линейная интерполяция | 4 subsampled residual blocks, 32 канала; обучена отдельно на MSE канала (в статье — end-to-end через BCE приёмника) |
| INCM `--iot-cov incm_oas` | EqDeepRx, Sec. II-C | ковариация помехи+шума по остатку на DMRS `y − Ĥp`, полоса 2 PRB, линейная усадка к tr(S)/p·I | вес усадки — оракул для комплексного гауссова случая с несмещёнными оценками моментов (OAS-подобный) |
| `lmmse_data` | A-MMSE, Sec. III-A ("LMMSE mismatch") | 2D LMMSE с ковариацией, оценённой по обучающим каналам | R(t₁,t₂,Δf) по WSS-предположению из 18 432 обучающих сеток, смещённая оценка (PSD) |
| `lmmse_data_1d` | A-MMSE, eq. (9) | 1D LMMSE по частоте на каждом DMRS-символе + интерполяция во времени | та же ковариация |
| `a_mmse` | A-MMSE, Sec. IV | двухступенчатый Attention-энкодер (частота → время) + ResFC-декодер учат **один фиксированный линейный фильтр W** | фильтр на блок 4 PRB (48 поднесущих × 14 символов из 2×12 пилотных значений), один и тот же для всех 17 блоков; банк фильтров по SNR (−10…35 дБ, шаг 5) и по смещению гребёнки CDM-группы; выбор по N0 |
| `ra_a_mmse` | A-MMSE, Sec. V | W·U·Vᵀ с обучаемыми U, V ранга r | r = 6 из 24 (25 %) |

Общее для всех: вход — CDM-деспредированные LS-оценки одной RX-антенны и одного слоя (по одному значению на 4-поднесущий DMRS-блок; 68 PRB → 2×204 значения). Обучение (EqDeepRx-протокол): UMa, скорость U(0, 35 м/с), 1–2 UE × ранг 1–2, 68 PRB, 4 RX-антенны, SNR U(−10, 40) дБ для DenoiseNN и фиксированный SNR на каждый фильтр A-MMSE. UMi, CDL-B, CDL-C — **вне обучающего распределения**.
""")

# ------------------------------------------------------------------ training
meta_p = M / "train_meta.json"
if meta_p.exists():
    meta = json.loads(meta_p.read_text())
    add("## 2. Обучение (валидация на UMa)\n")
    add(f"Обучающих выборок (RX-антенна × слой): {meta['train_samples']}, валидационных: {meta['val_samples']}, время {meta['duration_s'] / 60:.0f} мин на RTX PRO 4000.\n")
    v = meta["val_nmse_db_full_grid"]
    add("| SNR, дБ | LS + лин. интерп. | DenoiseNN | LMMSE (данные) |")
    add("|---|---|---|---|")
    for snr in sorted(v, key=float):
        r = v[snr]
        add(f"| {snr} | {f(r.get('ls_lin_interp'))} | {f(r.get('denoise_nn'))} | {f(r.get('lmmse_data'))} |")
    a = meta["val_nmse_db_ammse_chunks"]
    add("\nA-MMSE на блоках 4 PRB (NMSE, дБ; comb 0):\n")
    add("| SNR | LS лин. | A-MMSE фикс. W | A-MMSE W(x) по входу | фикс. W, дообученный напрямую | RA-A-MMSE r=6 |")
    add("|---|---|---|---|---|---|")
    for k, r in a.items():
        if k.startswith("c0_"):
            add(f"| {k.split('snr')[1]} | {f(r['ls_lin_interp'])} | {f(r['a_mmse_fixed'])} | {f(r['a_mmse_adaptive'])} | {f(r['fixed_filter_finetuned'])} | {f(r['ra_a_mmse'])} |")
    add("\nФиксированный W (среднее выходов сети, как в статье: «one final A-MMSE filter») не хуже входозависимого W(x) — подтверждение, что A-MMSE сходится к лучшему фиксированному линейному фильтру; прямое дообучение W даёт ещё немного (это и есть предел, о котором говорит статья).\n")

# ------------------------------------------------------------------ NMSE in the simulator
nm_p = D / "nmse" / "nmse.json"
if nm_p.exists():
    nm = json.loads(nm_p.read_text())
    add("## 3. NMSE в симуляторе (68 PRB, все оценщики)\n")
    add("![NMSE](nmse/nmse.png)\n")
    add("`eval_paper_ce.py --slots 256 --batch 16`: 256 слотов на точку, одни и те же реализации канала "
        "на всех SNR (seed сбрасывается перед каждой точкой). NMSE = Σ|Ĥ−H|² / Σ|H|² по всем слотам.\n")
    ests = list(next(iter(nm.values()))["nmse_db"])
    for snr in (0.0, 10.0, 20.0):
        add(f"**SNR {snr:g} дБ**, NMSE в дБ (жирным — лучший реализуемый):\n")
        add("| сценарий | " + " | ".join(label(e) for e in ests) + " |")
        add("|---|" + "---|" * len(ests))
        for lab, r in nm.items():
            i = r["snr_db"].index(snr)
            vals = [r["nmse_db"][e][i] for e in ests]
            real = [v for e, v in zip(ests, vals) if e != "perfect"]
            best = min(real)
            add(f"| {lab} | " + " | ".join(("**" + f(v) + "**") if v == best else f(v) for v in vals) + " |")
        add("")

# ------------------------------------------------------------------ campaign
camps = {}
for p in sorted((D / "campaigns").glob("*/*/*.json")):
    camps[p.parent.name] = json.loads(p.read_text())
if camps:
    add("## 4. Link-level: рабочая точка IRC (68 PRB)\n")
    add(f"{len(camps)} кампаний. Рабочая точка — SNR при BER = 1e-2 (и BLER = 0.1), потеря — относительно идеальной CSI на тех же слотах.\n")
    loss = collections.defaultdict(list)
    loss_b = collections.defaultdict(list)
    rows = []
    for ch in CH:
        for mod in MODS:
            for u, r in CFGS:
                c = camps.get(f"ce_{ch}_{mod}_ue{u}_r{r}")
                if not c:
                    continue
                for iot, cur in c["iot"].items():
                    p = cur["perfect"]["working_point_db"]
                    pb = cur["perfect"]["working_point_bler_db"]
                    vals = []
                    for e in CAMP_EST:
                        w = cur.get(e, {}).get("working_point_db")
                        wb = cur.get(e, {}).get("working_point_bler_db")
                        vals.append(w)
                        for key in (e, (e, ch), (e, mod), (e, f"{u}UE r{r}"), (e, f"IoT {float(iot):g}")):
                            if w is not None and p is not None:
                                loss[key].append(w - p)
                            if wb is not None and pb is not None:
                                loss_b[key].append(wb - pb)
                    rows.append((ch, mod, f"{u}UE r{r}", float(iot), p, vals))
    add("### 4.1 Медианная потеря рабочей точки, дБ (BER = 1e-2 / BLER = 0.1)\n")
    groups = ["все"] + CH + MODS + [f"{u}UE r{r}" for u, r in CFGS] + ["IoT 0", "IoT 10"]
    add("| оценщик | " + " | ".join(groups) + " | не достигнута |")
    add("|---|" + "---|" * (len(groups) + 1))
    for e in CAMP_EST:
        cells = []
        for g in groups:
            k = e if g == "все" else (e, g)
            a1, a2 = loss.get(k, []), loss_b.get(k, [])
            cells.append(f"{np.median(a1):.2f} / {np.median(a2):.2f}" if a1 and a2 else "—")
        miss = sum(1 for row in rows if row[5][CAMP_EST.index(e)] is None and row[4] is not None)
        add(f"| {label(e)} | " + " | ".join(cells) + f" | {miss}/{len(rows)} |")
    add("")
    # figure: loss boxplot
    fig, ax = plt.subplots(figsize=(10, 4.2))
    data = [loss[e] for e in CAMP_EST if loss[e]]
    labs = [label(e) for e in CAMP_EST if loss[e]]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    for patch, e in zip(bp["boxes"], [e for e in CAMP_EST if loss[e]]):
        patch.set_facecolor(RECEIVER_STYLE.get(e, {}).get("color", "grey"))
    ax.set_xticks(range(1, len(labs) + 1), labs, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("WP loss vs perfect CSI [dB]")
    ax.set_title("IRC working-point loss (BER = 1e-2), 68 PRB, all channels / modulations / UE configs / IoT")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(D / "ce_loss.png", dpi=130)
    plt.close(fig)
    add("![CE loss](ce_loss.png)\n")
    add("### 4.2 Все рабочие точки (BER = 1e-2), дБ\n")
    add("| канал | мод. | UE | IoT | perfect | " + " | ".join(label(e) for e in CAMP_EST) + " |")
    add("|---|---|---|---|---|" + "---|" * len(CAMP_EST))
    for ch, mod, cfg, iot, p, vals in rows:
        best = min([v for v in vals if v is not None], default=None)
        add(f"| {ch} | {mod} | {cfg} | {iot:g} | {f(p)} | " + " | ".join(("**" + f(v) + "**") if v is not None and v == best else f(v) for v in vals) + " |")
    add("")

    inc = [(k, v) for k, v in camps.items() if k.startswith("incm_")]
    if inc:
        add("### 4.3 Ковариация помехи для IRC (16QAM, 1 интерферент)\n")
        add("Рабочая точка BER = 1e-2, дБ. `perfect` — истинная ковариация, `residual` — широкополосная по остатку DMRS, `incm_oas` — EqDeepRx INCM (2 PRB + усадка).\n")
        add("| канал | UE | IoT | метод | " + " | ".join(f"IRC/{label(e)}" for e in ("ls_lin", "denoise_nn", "lmmse_data")) + " | L-MMSE/LS-lin |")
        add("|---|---|---|---|---|---|---|---|")
        for ch in ("uma", "cdl-c"):
            for nue in ("1", "2"):
                for iot in ("5.0", "10.0", "20.0"):
                    for cov in ("perfect", "residual", "incm_oas"):
                        c = camps.get(f"incm_{ch}_ue{nue}_{cov}")
                        if not c or iot not in c["iot"]:
                            continue
                        cur = c["iot"][iot]
                        vals = [cur.get(f"irc_{e}", {}).get("working_point_db") for e in ("ls_lin", "denoise_nn", "lmmse_data")]
                        lm = cur.get("lmmse_ls_lin", {}).get("working_point_db")
                        add(f"| {ch} | {nue} | {float(iot):g} | {cov} | " + " | ".join(f(v) for v in vals) + f" | {f(lm)} |")
        add("")

(D / "REPORT.md").write_text("\n".join(L))
print(f"wrote {D / 'REPORT.md'}")
