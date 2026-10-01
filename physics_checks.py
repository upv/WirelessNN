#!/usr/bin/env python3
"""Element-by-element physics report of the simulator (figures + JSON).

    .venv/bin/python physics_checks.py --outdir /root/reports/physics

Checks every block on its own at 68 PRB: transmitter power and DMRS layout,
channel power-delay profile / delay spread / time / frequency / spatial
correlation for every channel family, noise and interference calibration,
IRC covariance estimators, channel-estimation NMSE versus SNR for every
estimator and the LDPC decoder's error-free operation at high SNR.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import j0

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.channels import set_system_topology
from nr_ul_sim.interference import (
    interference_scale,
    irc_interference_covariance,
    spatial_covariance_from_channel,
)
from nr_ul_sim.parameters import CHANNEL_ESTIMATORS, snrdb_to_noise_var
from nr_ul_sim.plotting import RECEIVER_STYLE
from nr_ul_sim.pusch import strip_guard_subcarriers

C0 = 299792458.0
PRB = 68


def sim_for(**kw) -> NRUplinkSimulator:
    base = dict(num_prb=PRB, fft_size=1024, batch_size=8, max_mc_iter=1,
                num_target_bit_errors=1, snr_db=[10.0], iot_db=[0.0], seed=100)
    base.update(kw)
    return NRUplinkSimulator(SimConfig(**base))


def gen_h(sim: NRUplinkSimulator, batch: int) -> torch.Tensor:
    """Serving channel on the used subcarriers, [B, Rx, RxAnt, Tx, TxAnt, Sym, Sc]."""
    if sim.cfg.channel in ("umi", "uma"):
        set_system_topology(sim.serving_model, sim.cfg, batch, sim.cfg.num_ue)
    return strip_guard_subcarriers(sim.gen_serving(batch), sim.cfg.guard_carriers)


def pdp_of(h_used: torch.Tensor, scs: float):
    taps = torch.fft.ifft(h_used, dim=-1)
    n = taps.shape[-1]
    pdp = taps.abs().pow(2).mean(dim=tuple(range(taps.ndim - 1))).cpu().numpy()
    ts = 1.0 / (n * scs)
    t = np.arange(n) * ts
    t[n // 2 :] -= n * ts
    order = np.argsort(t)
    t, pdp = t[order], pdp[order] / pdp.sum()
    mean = (pdp * t).sum()
    rms = float(np.sqrt((pdp * (t - mean) ** 2).sum()))
    return t, pdp, rms


def _np(x):
    return np.asarray(x.detach().cpu() if torch.is_tensor(x) else x)


def cdl_table(sim: NRUplinkSimulator):
    """Cluster delays [s] and linear powers of the CDL model inside ``sim``."""
    link = sim.serving_model.links[0]
    delays = _np(link._delays).ravel()
    powers = _np(link._powers).ravel()
    if delays.max() > 1e-2:  # normalised (unit RMS) delays, not seconds
        delays = delays * sim.cfg.delay_spread
    powers = powers / powers.sum()
    return delays, powers


def cir_rms_ds(link, batch: int = 64) -> float:
    """RMS delay spread of the cluster impulse response, antenna pattern included."""
    a, tau = link(batch, 1, 1.0)
    p = _np(a.abs().pow(2).mean(dim=(0, 1, 2, 3, 4, 6)))
    t = _np(tau[0, 0, 0])
    p = p / p.sum()
    mean = (p * t).sum()
    return float(np.sqrt((p * (t - mean) ** 2).sum()))


def omni_cdl(cfg):
    from sionna.phy.channel.tr38901 import CDL, AntennaArray

    omni = AntennaArray(num_rows=1, num_cols=1, polarization="single", polarization_type="V",
                        antenna_pattern="omni", carrier_frequency=cfg.carrier_frequency)
    return CDL(model={"cdl-b": "B", "cdl-c": "C"}[cfg.channel], delay_spread=cfg.delay_spread,
               carrier_frequency=cfg.carrier_frequency, ut_array=omni, bs_array=omni,
               direction="uplink", min_speed=cfg.speed_mps, max_speed=cfg.speed_mps)


# --------------------------------------------------------------------------- #

def check_transmitter(out: dict) -> None:
    rows = []
    for mod, rank in (("qpsk", 1), ("qam16", 1), ("qam64", 1), ("qam16", 2)):
        sim = sim_for(modulation=mod, num_layers=rank, batch_size=4)
        tx = sim.transmitter
        x, b = sim.transmit(4)
        used = strip_guard_subcarriers(x, sim.cfg.guard_carriers)
        rows.append({
            "modulation": mod, "rank": rank,
            "mcs": list(sim.cfg.resolved_mcs),
            "target_coderate": float(tx._target_coderate),
            "effective_coderate": float(tx._tb_encoder.coderate),
            "tb_size": int(tx._tb_size),
            "num_codeblocks": int(getattr(tx._tb_encoder, "_num_cbs", 1)),
            "basegraph": str(getattr(tx._tb_encoder._encoder, "_bg", "?")),
            "power_per_layer": used.abs().pow(2).mean(dim=(0, 1, 3, 4)).tolist(),
            "guard_energy": float(x[..., : sim.cfg.guard_carriers[0]].abs().pow(2).sum()),
            "dmrs_symbols": sim.rx.dmrs_syms,
            "dmrs_pilot_power": float(sim.rx.pilot_grid.abs().pow(2).max()),
        })
    out["transmitter"] = rows


def check_channels(out: dict, outdir: Path) -> None:
    fams = [("cdl-b", 100.0), ("cdl-c", 300.0), ("umi", None), ("uma", None)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    rows = []
    for ax, (chan, ds) in zip(axes.ravel(), fams):
        sim = sim_for(channel=chan, delay_spread_ns=ds, batch_size=32, seed=5)
        h = gen_h(sim, 32)
        energy = float(h.abs().pow(2).mean())
        t, pdp, rms = pdp_of(h[:, 0, :, 0, 0, 0, :], sim.cfg.subcarrier_spacing)
        ax.semilogy(t * 1e6, np.maximum(pdp, 1e-7), lw=1.2, label=f"measured PDP, RMS DS {rms*1e9:.0f} ns")
        row = {"channel": chan, "energy_per_re": energy, "rms_ds_measured_ns": rms * 1e9,
               "energy_before_zero": float(pdp[t < -0.05e-6].sum()),
               "energy_beyond_cp_2p34us": float(pdp[t > 2.34e-6].sum())}
        if ds is not None:
            d, p = cdl_table(sim)
            ax.stem(d * 1e6, p, linefmt="C3-", markerfmt="C3.", basefmt=" ", label="38.901 clusters")
            mean = (p * d).sum()
            row["rms_ds_table_ns"] = float(np.sqrt((p * (d - mean) ** 2).sum()) * 1e9)
            row["configured_ds_ns"] = ds
            row["rms_ds_cir_with_38901_pattern_ns"] = cir_rms_ds(sim.serving_model.links[0]) * 1e9
            row["rms_ds_cir_omni_ns"] = cir_rms_ds(omni_cdl(sim.cfg)) * 1e9
        ax.axvline(2.34, color="grey", ls=":", label="normal CP 2.34 us")
        ax.set_xlim(-1.5, 5)
        ax.set_ylim(1e-6, 1)
        ax.set_title(f"{chan.upper()}  E|h|^2/RE = {energy:.3f}")
        ax.set_xlabel("delay [us]")
        ax.set_ylabel("normalised power")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=7)
        rows.append(row)
    fig.suptitle("Power-delay profile of the generated channels (68 PRB, 24.5 MHz band)")
    fig.tight_layout()
    fig.savefig(outdir / "pdp.png", dpi=140)
    plt.close(fig)
    out["channel_pdp"] = rows

    # time correlation vs Jakes
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rows = []
    for speed in (3.0, 10.0, 30.0, 120.0):
        sim = sim_for(channel="cdl-c", speed_kmh=speed, batch_size=32, seed=8)
        h = gen_h(sim, 32)[:, 0, :, 0, 0]
        rg = sim.transmitter.resource_grid
        fd = speed / 3.6 * sim.cfg.carrier_frequency / C0
        lags = np.arange(14)
        rho = []
        for lag in lags:
            num = (h[..., lag:, :] * h[..., : 14 - lag, :].conj()).mean()
            rho.append(float((num / h.abs().pow(2).mean()).abs()))
        theory = j0(2 * np.pi * fd * lags * rg.ofdm_symbol_duration)
        line, = ax.plot(lags, rho, "o", ms=4, label=f"{speed:g} km/h measured")
        ax.plot(lags, theory, "-", color=line.get_color(), lw=1, label=f"{speed:g} km/h J0(2 pi fd t)")
        rows.append({"speed_kmh": speed, "fd_hz": fd, "rho_lag13": rho[13], "j0_lag13": float(theory[13])})
    ax.set_xlabel("OFDM symbol lag")
    ax.set_ylabel("|time correlation|")
    ax.set_title("Time correlation over one slot vs Jakes (CDL-C, 3.5 GHz)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "time_correlation.png", dpi=140)
    plt.close(fig)
    out["time_correlation"] = rows

    # frequency correlation vs subcarrier lag
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rows = []
    for chan, ds in (("cdl-c", 30.0), ("cdl-c", 100.0), ("cdl-c", 300.0), ("cdl-b", 100.0), ("umi", None), ("uma", None)):
        sim = sim_for(channel=chan, delay_spread_ns=ds, batch_size=32, seed=9)
        h = gen_h(sim, 32)[:, 0, :, 0, 0, 0]
        lags = np.arange(0, 200, 4)
        rho = []
        for lag in lags:
            num = (h[..., lag:] * h[..., : h.shape[-1] - lag].conj()).mean()
            rho.append(float((num / h.abs().pow(2).mean()).abs()))
        label = f"{chan} {ds:g} ns" if ds else chan
        ax.plot(lags * 0.03, rho, label=label)
        half = lags[np.argmax(np.asarray(rho) < 0.5)] * 30e3 if min(rho) < 0.5 else None
        rows.append({"channel": chan, "ds_ns": ds, "coherence_bw_50pct_hz": half})
    ax.set_xlabel("frequency lag [MHz]")
    ax.set_ylabel("|frequency correlation|")
    ax.set_title("Frequency correlation of the channel (coherence bandwidth)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "frequency_correlation.png", dpi=140)
    plt.close(fig)
    out["frequency_correlation"] = rows

    # spatial correlation
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    rows = []
    for ax, (chan, ds) in zip(axes, fams):
        sim = sim_for(channel=chan, delay_spread_ns=ds, batch_size=32, seed=12)
        h = gen_h(sim, 32)
        r = spatial_covariance_from_channel(h[..., :1, :1, :, :]).mean(dim=0).cpu()
        d = r.diagonal().real.sqrt()
        rho = (r / torch.outer(d, d)).abs().numpy()
        im = ax.imshow(rho, vmin=0, vmax=1, cmap="viridis")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", color="w" if rho[i, j] < 0.6 else "k", fontsize=8)
        ax.set_title(f"{chan.upper()} |rho| rx antennas")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        rows.append({"channel": chan, "abs_corr": rho.tolist(),
                     "note": "1x2 dual-polarised panel: (0,1) and (2,3) are the two polarisations of one column"})
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(outdir / "spatial_correlation.png", dpi=140)
    plt.close(fig)
    out["spatial_correlation"] = rows


def check_noise_and_interference(out: dict) -> None:
    sim = sim_for(batch_size=8, seed=21)
    rows = []
    for snr in (-10.0, 0.0, 10.0, 20.0):
        slot = sim.generate_slot(8, snr, 0.0)
        clean = sim.apply_channel(sim.transmit(8)[0], slot["h"])
        no = snrdb_to_noise_var(snr)
        y = strip_guard_subcarriers(slot["y"], sim.cfg.guard_carriers)
        sig = strip_guard_subcarriers(clean, sim.cfg.guard_carriers)
        p_sig, p_y = float(sig.abs().pow(2).mean()), float(y.abs().pow(2).mean())
        rows.append({"snr_db": snr, "signal_power": p_sig, "noise_var_expected": no,
                     "noise_var_measured": p_y - p_sig,
                     "snr_realised_db": 10 * math.log10(p_sig / max(p_y - p_sig, 1e-12))})
    out["snr_calibration"] = rows

    rows = []
    for num_int, rank in ((1, 1), (2, 1), (3, 1), (2, 2)):
        sim = sim_for(num_interferers=num_int, num_layers=rank, modulation="qam16", batch_size=8, seed=31)
        cfg = sim.cfg
        no = snrdb_to_noise_var(10.0)
        h_i = sim.gen_int(8)
        y_i = strip_guard_subcarriers(sim.apply_channel(sim.interferer_grid(8, h_i), h_i), cfg.guard_carriers)
        for iot in (5.0, 10.0, 20.0):
            p = float((interference_scale(no, iot, cfg.num_interferer_streams) * y_i).abs().pow(2).mean())
            rows.append({"num_interferers": num_int, "ue_ant": cfg.resolved_ue_ant, "iot_db": iot,
                         "inr_realised_db": 10 * math.log10(p / no)})
    out["inr_calibration"] = rows

    # IRC covariance estimators vs truth
    rows = []
    for iot in (10.0, 20.0):
        for est_name in ("perfect", "ls_lin", "ls_soft_window"):
            sim = sim_for(num_interferers=2, batch_size=4, seed=41, channel_estimators=(est_name,))
            cfg = sim.cfg
            no = snrdb_to_noise_var(15.0)
            slot = sim.generate_slot(4, 15.0, iot)
            sim.estimate_csi(slot)
            h_hat = slot["h_perf"] if est_name == "perfect" else slot["h_hat"]
            r_true = irc_interference_covariance(slot["h_int"], slot["y"], no, iot, "perfect")
            r_res = irc_interference_covariance(
                slot["h_int"], slot["y"], no, iot, "residual", h_hat=h_hat,
                pilot_grid=sim.rx.pilot_grid, dmrs_syms=sim.rx.dmrs_syms, guard_carriers=cfg.guard_carriers)
            r_est = irc_interference_covariance(slot["h_int"], slot["y"], no, iot, "estimated")
            rel = lambda r: float(((r - r_true).norm(dim=(1, 2)) / r_true.norm(dim=(1, 2))).mean())
            rows.append({"iot_db": iot, "csi_for_residual": est_name,
                         "rel_error_residual": rel(r_res), "rel_error_estimated_ryy": rel(r_est),
                         "trace_true": float(r_true.diagonal(dim1=1, dim2=2).real.sum(-1).mean()),
                         "trace_residual": float(r_res.diagonal(dim1=1, dim2=2).real.sum(-1).mean())})
    out["irc_covariance"] = rows


def check_channel_estimation(out: dict, outdir: Path) -> None:
    ests = tuple(CHANNEL_ESTIMATORS)
    snrs = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    rows = []
    for ax, (chan, ds, nue) in zip(axes, (("cdl-c", 300.0, 1), ("cdl-b", 100.0, 2), ("uma", None, 1))):
        curves = {e: [] for e in ests if e != "perfect"}
        calib = {e: [] for e in curves}
        for snr in snrs:
            sim = sim_for(channel=chan, delay_spread_ns=ds, num_ue=nue, batch_size=4, seed=2,
                          receivers=("irc",), channel_estimators=ests, snr_db=[snr])
            slot = sim.generate_slot(4, snr, 0.0)
            sim.estimate_csi(slot)
            hp = slot["csi"]["perfect"][0]
            for e in curves:
                h, err = slot["csi"][e]
                mse = float((h - hp).abs().pow(2).mean())
                curves[e].append(10 * math.log10(mse / float(hp.abs().pow(2).mean())))
                calib[e].append(float(err.mean()) / mse)
        for e, vals in curves.items():
            st = RECEIVER_STYLE[e]
            ax.plot(snrs, vals, marker=st["marker"], color=st["color"], label=st["label"], ms=4)
        ax.set_title(f"{chan.upper()} {f'{ds:g} ns' if ds else ''} {nue} UE")
        ax.set_xlabel("SNR per antenna [dB]")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=7)
        rows.append({"channel": chan, "ds_ns": ds, "num_ue": nue, "snr_db": snrs,
                     "nmse_db": curves, "reported_over_measured_err_var": calib})
    axes[0].set_ylabel("NMSE [dB]")
    fig.suptitle("Channel-estimation NMSE vs SNR, 68 PRB, DMRS type-1 add-pos 1")
    fig.tight_layout()
    fig.savefig(outdir / "ce_nmse_vs_snr.png", dpi=140)
    plt.close(fig)
    out["channel_estimation"] = rows

    # tap-domain picture of the LS estimate and of both windows
    sim = sim_for(channel="cdl-c", delay_spread_ns=300.0, batch_size=1, seed=2, receivers=("irc",),
                  channel_estimators=("perfect", "ls_hard_window", "ls_soft_window"), snr_db=[5.0])
    slot = sim.generate_slot(1, 5.0, 0.0)
    sim.estimate_csi(slot)
    est = sim.rx.estimators["ls_soft_window"]
    y_p = est._extract_pilots(slot["y"])
    h_p, err_p = est.estimate_at_pilot_locations(y_p, slot["no"])
    err_p = torch.broadcast_to(err_p, h_p.shape)
    blk = est._block_estimates(h_p)
    taps = torch.fft.ifft(blk, dim=-1)
    power = taps.abs().pow(2).mean(dim=(0, 1, 2, 5))[0, 0].cpu().numpy()
    gain = est._tap_gains(taps, est._block_estimates(err_p.to(h_p.dtype)).real)[0, 0, 0, 0, 0, 0].cpu().numpy()
    hp = slot["csi"]["perfect"][0][..., 2, 1::4]
    true_p = torch.fft.ifft(hp, dim=-1).abs().pow(2).mean(dim=(0, 1, 2, 3, 4)).cpu().numpy()
    m = len(power)
    k = np.arange(m)
    kk = np.where(k < (m + 1) // 2, k, k - m)
    order = np.argsort(kk)
    noise_tap = float(err_p.mean()) / m
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.semilogy(kk[order] * est.tap_spacing_s * 1e6, true_p[order], "k-", lw=1, label="true channel taps")
    ax.semilogy(kk[order] * est.tap_spacing_s * 1e6, power[order], "C0.", ms=3, label="LS taps at 5 dB SNR")
    ax.axhline(noise_tap, color="grey", ls=":", label="noise per tap")
    win = est._window.cpu().numpy()
    ax.fill_between(kk[order] * est.tap_spacing_s * 1e6, 1e-9, 10, where=win[order] > 0, color="C3", alpha=0.08, label="hard window")
    ax2 = ax.twinx()
    ax2.plot(kk[order] * est.tap_spacing_s * 1e6, gain[order], "C2-", lw=0.9, label="soft-window gain")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("soft gain")
    ax.set_ylim(1e-7, 1)
    ax.set_xlabel("delay [us]")
    ax.set_ylabel("tap power")
    ax.set_title("Delay-domain view of the DMRS LS estimate (CDL-C 300 ns, 68 PRB)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "ce_tap_domain.png", dpi=140)
    plt.close(fig)


def check_receivers(out: dict) -> None:
    rows = []
    sim = sim_for(receivers=("mr", "lmmse", "ideal_mmse", "zf", "irc"), num_ue=2, batch_size=4, seed=53, max_mc_iter=2)
    stats = sim.measure_point(30.0, 0.0)
    rows.append({"check": "all receivers error-free at 30 dB, 2 UE",
                 "ber": {k: s.ber for k, s in stats.items()}, "blocks": {k: s.num_blocks for k, s in stats.items()}})
    sim = sim_for(receivers=("lmmse", "irc"), batch_size=2, seed=51)
    slot = sim.generate_slot(2, 5.0, 0.0)
    sim.estimate_csi(slot)
    rows.append({"check": "IRC == LMMSE without interference (bit-exact)",
                 "equal": bool(torch.equal(sim.detect(slot, "lmmse"), sim.detect(slot, "irc")))})
    sim = sim_for(receivers=("lmmse", "irc"), num_interferers=1, batch_size=4, seed=54, max_mc_iter=4)
    stats = sim.measure_point(10.0, 20.0)
    rows.append({"check": "IRC vs L-MMSE under 20 dB coloured interference at 10 dB SNR",
                 "ber": {k: s.ber for k, s in stats.items()}})
    out["receivers"] = rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("physics_out"))
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out: dict = {"prb": PRB, "device": str(sim_for().device)}
    t0 = time.time()
    for name, fn in (("transmitter", lambda: check_transmitter(out)),
                     ("channels", lambda: check_channels(out, args.outdir)),
                     ("noise_interference", lambda: check_noise_and_interference(out)),
                     ("channel_estimation", lambda: check_channel_estimation(out, args.outdir)),
                     ("receivers", lambda: check_receivers(out))):
        t = time.time()
        fn()
        print(f"{name:20s} done in {time.time() - t:.1f} s")
    out["duration_s"] = time.time() - t0
    (args.outdir / "physics_checks.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {args.outdir / 'physics_checks.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
