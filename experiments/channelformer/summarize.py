"""Report for run_campaign.py output: working points, throughput, NMSE, configs, plots.

    python summarize.py results/mimo4x2_20261116      -> report.md, *.png, summary.json there
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ("perfect", "channelformer", "denoise_nn", "lmmse_data", "lmmse_data_1d", "a_mmse", "ra_a_mmse",
         "ls_fir", "ls_soft_window", "ls_hard_window", "lmmse_exp", "lmmse_ce", "ls_lin_time_avg",
         "ls_lin", "ls_nn")
TRAINED_ON_UMA = {"denoise_nn", "lmmse_data", "lmmse_data_1d", "a_mmse", "ra_a_mmse"}
IMG = chr(33)  # markdown image prefix
CHANNEL_ORDER = ("cdl-b", "cdl-c", "cdl-d", "umi", "uma")


def fmt(v, nd=1):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def main(root: Path):
    bler = {f.stem: json.loads(f.read_text()) for f in sorted((root / "bler").glob("*.json"))}
    nmse = {f.stem: json.loads(f.read_text()) for f in sorted((root / "nmse").glob("*.json"))}
    scen = sorted({(d["config"]["channel"], d["config"]["speed_kmh"]) for d in bler.values()},
                  key=lambda x: (CHANNEL_ORDER.index(x[0]), x[1]))
    mods = [m for m in ("qpsk", "qam16", "qam64") if any(k.endswith(m) for k in bler)]
    ests = [e for e in ORDER if any(e in d["iot"]["0.0"] for d in bler.values())]
    out = ["# 4x2 MIMO PUSCH campaign: 14 built-in estimators + Channelformer", ""]
    any_cfg = next(iter(bler.values()))
    c, p = any_cfg["config"], any_cfg["provenance"]
    out += ["## Fixed configuration", "",
            "| item | value |", "|---|---|"]
    rows = [
        ("gNB antennas", f"{c['num_rx_ant']} (38.901 panel 1x2 dual-pol ±45°, 38.901 element pattern)"),
        ("UE antenna ports / layers", f"{c['num_ue_ant']} (single dual-pol omni) / rank {c['rank']}, non-codebook"),
        ("UEs / interference", f"{c['num_ue']} / none (IoT 0 dB)"),
        ("carrier / SCS", f"{c['carrier_frequency_hz'] / 1e9:g} GHz / {c['subcarrier_spacing_khz']:g} kHz"),
        ("PRB / FFT", f"{c['num_prb']} ({c['num_used_subcarriers']} subcarriers) / {c['fft_size']}"),
        ("slot", "14 OFDM symbols, mapping type A, normal CP"),
        ("DMRS", f"type {c['dmrs_type']}, length {c['dmrs_length']}, additional position "
                 f"{c['dmrs_additional_position']} (symbols 2 and 11), 2 CDM groups w/o data"),
        ("receiver", ", ".join(c["receivers"]) + " (MMSE, S = N0 I) + LDPC decoding"),
        ("power", f"{c['tx_power_norm']} (SNR per rx antenna per RE for the whole UE)"),
        ("Monte Carlo", f"batch {any_cfg['config'].get('batch_size', 8)} slots, stop at 100 block errors "
                        f"on every curve or 200 slots"),
        ("code", f"upv/WirelessNN {p['repo_commit']} (dirty={p['repo_dirty']}), torch {p['torch']}, "
                 f"sionna {p['sionna']}, {p['gpu']}"),
        ("Channelformer", f"checkpoint epoch {p['channelformer_epoch']}, val NMSE "
                          f"{fmt(p['channelformer_val_nmse_db'], 2)} dB"),
    ]
    out += [f"| {a} | {b} |" for a, b in rows]
    out += ["", "| modulation | MCS table / index | TB size [bit] | peak throughput [Mbit/s] | SNR range [dB] |",
            "|---|---|---|---|---|"]
    for m in mods:
        d = next(v for k, v in bler.items() if k.endswith(m))
        s = d["iot"]["0.0"]["perfect"]["snr_db"]
        out.append(f"| {m} | {d['config']['mcs_table']} / {d['config']['mcs_index']} | {d['tb_size_bits']} | "
                   f"{d['tb_size_bits'] / d['slot_duration_s'] / 1e6:.1f} | {s[0]:g} … {s[-1]:g} |")
    out += ["", "Channel delay spreads: CDL-B 100 ns, CDL-C 300 ns, CDL-D 100 ns (defaults); "
            "UMi/UMa draw per drop (38.901, low O2I, no pathloss/shadowing). Speeds as listed per scenario.",
            "", f"Estimators marked * ({', '.join(sorted(TRAINED_ON_UMA))}) were trained on UMa only "
            "(models/paper_ce); everything except UMa is out of their training distribution. "
            "Channelformer is trained on a mix of all five channel types. lmmse_ce uses a TDL prior "
            "that is the true PDP on CDL only (genie there, mismatched on UMi/UMa).", ""]

    summary = {}
    for m in mods:
        out += [f"## SNR [dB] at BLER 10 %, {m}", "",
                "| estimator | " + " | ".join(f"{ch} {v:g}" for ch, v in scen) + " |",
                "|---|" + "---|" * len(scen)]
        for e in ests:
            row = []
            for ch, v in scen:
                d = bler.get(f"{ch}_v{v:g}_{m}")
                wp = d["iot"]["0.0"].get(e, {}).get("working_point_bler_db") if d else None
                summary.setdefault(m, {}).setdefault(e, {})[f"{ch}_v{v:g}"] = wp
                row.append(fmt(wp))
            out.append(f"| {e}{'*' if e in TRAINED_ON_UMA else ''} | " + " | ".join(row) + " |")
        out += ["", "– = 10 % not reached inside the SNR range.", ""]
    for m in mods:
        out += [f"## Loss vs perfect CSI at BLER 10 % [dB], {m}", "",
                "| estimator | " + " | ".join(f"{ch} {v:g}" for ch, v in scen) + " |",
                "|---|" + "---|" * len(scen)]
        for e in ests:
            if e == "perfect":
                continue
            row = []
            for ch, v in scen:
                a = summary[m][e][f"{ch}_v{v:g}"]
                b = summary[m]["perfect"][f"{ch}_v{v:g}"]
                row.append(fmt(a - b, 2) if a is not None and b is not None else "–")
            out.append(f"| {e}{'*' if e in TRAINED_ON_UMA else ''} | " + " | ".join(row) + " |")
        out.append("")
    out += ["## CE NMSE [dB] (common random numbers over SNR, 64 slots per point)", ""]
    for snr in (0.0, 10.0, 20.0):
        out += [f"### SNR {snr:g} dB", "",
                "| estimator | " + " | ".join(f"{ch} {v:g}" for ch, v in scen) + " |",
                "|---|" + "---|" * len(scen)]
        for e in ests:
            if e == "perfect":
                continue
            row = []
            for ch, v in scen:
                d = nmse.get(f"{ch}_v{v:g}")
                if d and e in d["nmse_db"] and snr in d["snr_db"]:
                    row.append(fmt(d["nmse_db"][e][d["snr_db"].index(snr)]))
                else:
                    row.append("–")
            out.append(f"| {e}{'*' if e in TRAINED_ON_UMA else ''} | " + " | ".join(row) + " |")
        out.append("")

    cmap = plt.get_cmap("tab20")
    colors = {e: cmap(i) for i, e in enumerate(ests)}
    for m in mods:
        fig, axes = plt.subplots(len(scen) // 3 or 1, 3, figsize=(17, 3.6 * max(1, len(scen) // 3)), squeeze=False)
        for ax, (ch, v) in zip(axes.ravel(), scen):
            d = bler.get(f"{ch}_v{v:g}_{m}")
            if not d:
                continue
            for e in ests:
                cv = d["iot"]["0.0"].get(e)
                if cv:
                    ax.semilogy(cv["snr_db"], np.maximum(cv["bler"], 1e-3), color=colors[e],
                                lw=2 if e in ("perfect", "channelformer") else 1, label=e)
            ax.set_title(f"{ch} {v:g} km/h {m}", fontsize=9)
            ax.set_ylim(1e-3, 1.05)
            ax.grid(True, which="both", ls=":", alpha=0.5)
        axes[0, 0].legend(fontsize=6, ncol=2)
        fig.supxlabel("SNR per rx antenna [dB]")
        fig.supylabel("BLER")
        fig.tight_layout()
        fig.savefig(root / f"bler_{m}.png", dpi=110)
        plt.close(fig)
        out.append(IMG + f"[BLER {m}](bler_{m}.png)")
    fig, axes = plt.subplots(len(scen) // 3 or 1, 3, figsize=(17, 3.6 * max(1, len(scen) // 3)), squeeze=False)
    for ax, (ch, v) in zip(axes.ravel(), scen):
        d = nmse.get(f"{ch}_v{v:g}")
        if not d:
            continue
        for e, vals in d["nmse_db"].items():
            ax.plot(d["snr_db"], vals, color=colors.get(e), lw=2 if e == "channelformer" else 1, label=e)
        ax.set_title(f"{ch} {v:g} km/h", fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.supxlabel("SNR per rx antenna [dB]")
    fig.supylabel("NMSE [dB]")
    fig.tight_layout()
    fig.savefig(root / "nmse.png", dpi=110)
    out.append(IMG + "[NMSE](nmse.png)")
    (root / "report.md").write_text("\n".join(out) + "\n")
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {root / 'report.md'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
