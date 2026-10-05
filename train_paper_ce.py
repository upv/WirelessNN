#!/usr/bin/env python3
"""Train the paper channel estimators (nr_ul_sim/paper_ce.py).

    .venv/bin/python train_paper_ce.py --out models/paper_ce

Training data follow the EqDeepRx protocol: 3GPP UMa drops, UE speed
uniform in 0-35 m/s, 68 PRB, 4 RX antennas, configurations 1-2 UE x rank 1-2
(so both DMRS CDM groups and the inter-port leakage of OCC despreading are
seen). Noise is added on the fly to the noiseless despread LS estimates.

Writes into --out:
    data_cov.pt     time-frequency covariance R[t1, t2, Δn] (lmmse_data, lmmse_data_1d)
    denoise_nn.pt   DenoiseNN weights + error-variance table
    a_mmse.pt       fixed A-MMSE / RA-A-MMSE filters per comb offset and SNR + tables
    train_meta.json settings and validation NMSE per SNR
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from nr_ul_sim import NRUplinkSimulator, SimConfig
from nr_ul_sim.paper_ce import (
    AMMSENet,
    BlockEstimator,
    DataCovariance,
    DenoiseNN,
    ch_to_complex,
    chunk_inputs,
    chunk_targets,
    complex_to_ch,
    freq_cov_from_samples,
    lmmse_matrices,
    psd_clip,
    smoothing_taps,
)

CONFIGS = [(1, 1), (2, 1), (1, 2), (2, 2)]
SNR_BANK = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------- data
def make_pool(channel: str, slots_per_cfg: int, batch: int, prb: int, seed: int, max_speed: float):
    """Noiseless despread LS blocks, true grids and comb offsets, flattened over
    (slot, rx antenna, layer)."""
    from sionna.phy.channel import gen_single_sector_topology

    blks, grids, combs = [], [], []
    fe0 = None
    for ci, (nue, rank) in enumerate(CONFIGS):
        cfg = SimConfig(channel=channel, num_ue=nue, num_layers=rank, num_prb=prb, num_interferers=0,
                        receivers=("irc",), channel_estimators=("ls_lin",), seed=seed + ci)
        sim = NRUplinkSimulator(cfg)
        tx = sim.transmitter
        fe = BlockEstimator(tx.resource_grid, tx._dmrs_length, tx._dmrs_additional_position,
                            tx._num_cdm_groups_without_data)
        fe0 = fe0 or fe
        model = sim.serving_model
        for _ in range(slots_per_cfg // batch):
            if channel in ("umi", "uma"):
                if hasattr(model, "allocate_topology_tensors") and getattr(model, "_nrul_alloc", None) != (batch, nue):
                    model.allocate_topology_tensors(batch_size=batch, num_bs=1, num_ut=nue)
                    model._nrul_alloc = (batch, nue)
                model.set_topology(*gen_single_sector_topology(
                    batch, nue, channel, min_ut_velocity=0.0, max_ut_velocity=max_speed))
            with torch.no_grad():
                x, _ = sim.transmit(batch)
                h = sim.gen_serving(batch)
                y = sim.apply_channel(x, h)
                blk, _ = fe.blocks(y, torch.tensor(1e-12, device=y.device))   # [B,1,Rx,Tx,S,Nd,M]
                grid = sim.true_channel(h)                                    # [B,1,Rx,Tx,S,T,N]
            b, _, rx, nt, ns = blk.shape[:5]
            comb = torch.as_tensor(fe.comb, device=blk.device).reshape(1, 1, nt, ns).expand(b, rx, nt, ns)
            blks.append(blk[:, 0].reshape(-1, *blk.shape[-2:]))
            grids.append(grid[:, 0].reshape(-1, *grid.shape[-2:]))
            combs.append(comb.reshape(-1))
        log(f"pool {channel} cfg {nue}UE r{rank}: {sum(len(c) for c in combs)} samples so far")
        del sim
    return torch.cat(blks), torch.cat(grids), torch.cat(combs), fe0


def add_noise(blk: torch.Tensor, snr_db: torch.Tensor):
    """Despread LS noise: CN(0, N0 / 4) per block, N0 = 10^(-SNR/10)."""
    var = (10.0 ** (-snr_db / 10.0) / 4.0).to(blk.real.dtype)
    shape = (-1,) + (1,) * (blk.dim() - 1)
    noise = torch.complex(torch.randn_like(blk.real), torch.randn_like(blk.real)) * (var / 2).sqrt().reshape(shape)
    return blk + noise, var


def block_targets(grid, comb, dmrs_syms, spacing):
    """True channel at the block centres 4k + c + 1 on the DMRS symbols: [S, Nd, M]."""
    n = grid.shape[-1]
    m = n // spacing
    pos = torch.arange(m, device=grid.device) * spacing + 1 + comb[:, None]          # [S, M]
    g = grid[:, dmrs_syms, :]                                                          # [S, Nd, N]
    return torch.gather(g, 2, pos[:, None, :].expand(-1, g.shape[1], -1))


def interp_full(fe: BlockEstimator, blk, comb):
    """Linear interpolation of [S, Nd, M] blocks to [S, T, N] grids."""
    out = torch.empty(blk.shape[0], fe.num_sym, fe.num_sc, dtype=blk.dtype, device=blk.device)
    w = fe._time_w.to(blk.device).to(blk.dtype)
    for c in (0, 1):
        sel = comb == c
        if sel.any():
            f = fe._interp[c].to(blk.device).to(blk.dtype)
            h_d = torch.einsum("nm,sdm->sdn", f, blk[sel])
            out[sel] = torch.einsum("ld,sdn->sln", w, h_d)
    return out


# --------------------------------------------------------------------------- covariance
def train_covariance(grids, out: Path):
    s, t, n = grids.shape
    acc = 0
    for i in range(0, s, 256):
        acc = acc + freq_cov_from_samples(grids[i: i + 256])
    # biased lag estimate (divide by N): the block-Toeplitz covariance built from
    # it is positive semi-definite; the unbiased 1/(N-|lag|) one is not and makes
    # (R_xx + s2 I)^-1 blow up
    r = acc / (s * n)
    torch.save({"r": r.cpu(), "num_samples": s, "num_sc": n}, out / "data_cov.pt")
    log(f"covariance from {s} grids, mean power {r[torch.arange(t), torch.arange(t), n - 1].real.mean():.3f}")
    return r


# --------------------------------------------------------------------------- DenoiseNN
def train_denoise(fe, tr, va, out: Path, steps: int, batch: int):
    blk, grid, comb = tr
    tgt = block_targets(grid, comb, fe.dmrs_syms, fe.block_spacing)
    net = DenoiseNN(num_sym=len(fe.dmrs_syms)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    for step in range(steps):
        idx = torch.randint(0, blk.shape[0], (batch,), device=DEV)
        snr = torch.empty(batch, device=DEV).uniform_(-10.0, 40.0)
        noisy, _ = add_noise(blk[idx], snr)
        pred = ch_to_complex(net(complex_to_ch(noisy)), noisy.shape)
        mse = (pred - tgt[idx]).abs().pow(2).mean(dim=(-2, -1))
        loss = torch.log10(mse + 1e-7).mean()          # balances the SNR range
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            log(f"denoise step {step} loss {loss.item():.3f}")
    net.eval()
    snrs, mse_full, nmse = [], [], {}
    vb, vg, vc = va
    with torch.no_grad():
        for snr in range(-10, 45, 5):
            g = torch.Generator(device=DEV).manual_seed(snr + 100)
            torch.manual_seed(snr + 100)
            noisy, _ = add_noise(vb, torch.full((vb.shape[0],), float(snr), device=DEV))
            den = torch.cat([ch_to_complex(net(complex_to_ch(noisy[i: i + 1024])), noisy[i: i + 1024].shape)
                             for i in range(0, noisy.shape[0], 1024)])
            full = interp_full(fe, den, vc)
            lin = interp_full(fe, noisy, vc)
            p = vg.abs().pow(2).mean()
            m = (full - vg).abs().pow(2).mean()
            snrs.append(float(snr))
            mse_full.append(float(m))
            nmse[snr] = {"denoise_nn": float(10 * torch.log10(m / p)),
                         "ls_lin_interp": float(10 * torch.log10((lin - vg).abs().pow(2).mean() / p))}
            log(f"denoise val SNR {snr:3d}: NMSE {nmse[snr]['denoise_nn']:.1f} dB (linear interp {nmse[snr]['ls_lin_interp']:.1f})")
    torch.save({"config": net.config, "state_dict": net.cpu().state_dict(),
                "errvar_snr_db": snrs, "errvar_mse": mse_full}, out / "denoise_nn.pt")
    return nmse


# --------------------------------------------------------------------------- A-MMSE
def train_ammse(fe, tr, va, out: Path, steps: int, warm_steps: int, ra_steps: int, rank: int, batch: int):
    chunk_blocks = 12  # 4 PRB per chunk: 48 subcarriers, 12 DMRS blocks per symbol
    chunk_sc = chunk_blocks * fe.block_spacing
    nd, t = len(fe.dmrs_syms), fe.num_sym
    w_bank, w_ra_bank, meta = {}, {}, {}
    err_tab = {s: [] for s in SNR_BANK}
    err_tab_ra = {s: [] for s in SNR_BANK}
    for c in (0, 1):
        sel, vsel = tr[2] == c, va[2] == c
        if not sel.any():
            continue
        x_all = chunk_inputs(tr[0][sel], chunk_blocks)                 # [S*C, L]
        h_all = chunk_targets(tr[1][sel], chunk_sc)                    # [S*C, T*Csc]
        xv = chunk_inputs(va[0][vsel], chunk_blocks)
        hv = chunk_targets(va[1][vsel], chunk_sc)
        keep = torch.randperm(xv.shape[0], device=xv.device)[:8192]     # validation chunks
        xv, hv = xv[keep], hv[keep]
        l = x_all.shape[1]
        net = AMMSENet(num_pilots=l, num_sc=chunk_sc, num_sym=t).to(DEV)
        w_bank[c], w_ra_bank[c] = [], []
        for si, snr in enumerate(SNR_BANK):
            n_steps = steps if si == 0 else warm_steps
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=2e-5)
            for step in range(n_steps):
                idx = torch.randint(0, x_all.shape[0], (batch,), device=DEV)
                xn, _ = add_noise(x_all[idx], torch.full((batch,), snr, device=DEV))
                w = net(torch.cat([xn.real, xn.imag], dim=-1))
                est = torch.einsum("bnl,bl->bn", w, xn)
                loss = (est - h_all[idx]).abs().pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            with torch.no_grad():
                torch.manual_seed(int(snr) + 7)
                xvn, _ = add_noise(xv, torch.full((xv.shape[0],), snr, device=DEV))
                w_sum, ad = 0, []
                for i in range(0, xvn.shape[0], 512):
                    w = net(torch.cat([xvn[i: i + 512].real, xvn[i: i + 512].imag], -1))
                    w_sum = w_sum + w.sum(dim=0)
                    ad.append(torch.einsum("bnl,bl->bn", w, xvn[i: i + 512]))
                ad = torch.cat(ad)
                w_fix = w_sum / xvn.shape[0]                           # one fixed filter (paper Sec. IV-B3)
                p = hv.abs().pow(2).mean()
                mse_fix = (xvn @ w_fix.T - hv).abs().pow(2).mean()
                mse_ad = (ad - hv).abs().pow(2).mean()
            # direct fine-tune of the fixed filter: the paper's large-sample target
            w_ft = w_fix.clone().requires_grad_(True)
            opt_ft = torch.optim.Adam([w_ft], lr=1e-3)
            for _ in range(warm_steps):
                idx = torch.randint(0, x_all.shape[0], (batch * 4,), device=DEV)
                xn, _ = add_noise(x_all[idx], torch.full((batch * 4,), snr, device=DEV))
                loss = (xn @ w_ft.T - h_all[idx]).abs().pow(2).mean()
                opt_ft.zero_grad()
                loss.backward()
                opt_ft.step()
            with torch.no_grad():
                mse_ft = (xvn @ w_ft.T - hv).abs().pow(2).mean()
            # RA-A-MMSE: W U V^T with real U, V [L, r], W fixed
            u = (torch.randn(l, rank, device=DEV) / math.sqrt(l)).requires_grad_(True)
            v = (torch.randn(l, rank, device=DEV) / math.sqrt(rank)).requires_grad_(True)
            with torch.no_grad():  # start from the dominant right singular subspace
                _, _, vh = torch.linalg.svd(w_fix)
                basis = vh[:rank].conj().T.real
                u.copy_(basis)
                v.copy_(basis)
            opt_ra = torch.optim.Adam([u, v], lr=1e-3)
            for _ in range(ra_steps):
                idx = torch.randint(0, x_all.shape[0], (batch * 4,), device=DEV)
                xn, _ = add_noise(x_all[idx], torch.full((batch * 4,), snr, device=DEV))
                w_r = w_fix @ (u @ v.T).to(w_fix.dtype)
                loss = (xn @ w_r.T - h_all[idx]).abs().pow(2).mean()
                opt_ra.zero_grad()
                loss.backward()
                opt_ra.step()
            with torch.no_grad():
                w_r = w_fix @ (u @ v.T).to(w_fix.dtype)
                mse_ra = (xvn @ w_r.T - hv).abs().pow(2).mean()
                lin = interp_chunk_linear(fe, xvn, c, chunk_blocks, chunk_sc)
                mse_lin = (lin - hv).abs().pow(2).mean()
            w_bank[c].append(w_fix.detach().cpu())
            w_ra_bank[c].append(w_r.detach().cpu())
            err_tab[snr].append(float(mse_fix))
            err_tab_ra[snr].append(float(mse_ra))
            db = lambda m: float(10 * torch.log10(m / p))
            meta[f"c{c}_snr{snr:g}"] = {"a_mmse_fixed": db(mse_fix), "a_mmse_adaptive": db(mse_ad),
                                        "fixed_filter_finetuned": db(mse_ft), "ra_a_mmse": db(mse_ra),
                                        "ls_lin_interp": db(mse_lin)}
            log(f"A-MMSE comb {c} SNR {snr:5.1f}: fixed {db(mse_fix):6.1f} adaptive {db(mse_ad):6.1f} "
                f"fixed-finetuned {db(mse_ft):6.1f} RA(r={rank}) {db(mse_ra):6.1f} LS-lin {db(mse_lin):6.1f} dB")
        w_bank[c] = torch.stack(w_bank[c])
        w_ra_bank[c] = torch.stack(w_ra_bank[c])
    tab = lambda d: {"snr_db": SNR_BANK, "mse": [float(np.mean(d[s])) for s in SNR_BANK]}
    torch.save({"snr_db": SNR_BANK, "w": w_bank, "w_ra": w_ra_bank, "rank": rank,
                "chunk_blocks": chunk_blocks, "chunk_sc": chunk_sc,
                "errvar": tab(err_tab), "errvar_ra": tab(err_tab_ra)}, out / "a_mmse.pt")
    return meta


def interp_chunk_linear(fe, x, comb, chunk_blocks, chunk_sc):
    """Linear interpolation inside one chunk (reference for the chunk NMSE)."""
    from nr_ul_sim.paper_ce import linear_interp_matrix
    pos = np.arange(chunk_blocks) * fe.block_spacing + comb + 1
    f = torch.as_tensor(linear_interp_matrix(chunk_sc, pos), device=x.device).to(x.dtype)
    nd = len(fe.dmrs_syms)
    xd = x.reshape(x.shape[0], nd, chunk_blocks)
    h_d = torch.einsum("nm,sdm->sdn", f, xd)
    w = fe._time_w.to(x.device).to(x.dtype)
    return torch.einsum("ld,sdn->sln", w, h_d).reshape(x.shape[0], -1)


# --------------------------------------------------------------------------- reference NMSE
def reference_nmse(fe, va, r, nmse):
    """NMSE of FIR smoothing and data LMMSE (2D / 1D) on the validation pool."""
    vb, vg, vc = va
    cov = DataCovariance(r.to(DEV), fe.num_sc)
    df = fe.block_spacing * float(fe.rg.subcarrier_spacing)
    g, lp = smoothing_taps(17, df, -1e-6, 3e-6)
    p = vg.abs().pow(2).mean()
    for snr in range(-10, 45, 5):
        torch.manual_seed(snr + 100)
        noisy, var = add_noise(vb, torch.full((vb.shape[0],), float(snr), device=DEV))
        v = float(var[0])
        out = torch.empty_like(vg)
        for c in (0, 1):
            sel = vc == c
            if not sel.any():
                continue
            rxx, rhx = lmmse_matrices(cov, fe.dmrs_syms, fe.num_blocks, fe.block_spacing, c, fe.num_sym, fe.num_sc, DEV)
            rxx = psd_clip(rxx)
            w = torch.linalg.solve((rxx + v * torch.eye(rxx.shape[0], device=DEV, dtype=rxx.dtype)).T, rhx.T).T
            out[sel] = (noisy[sel].reshape(int(sel.sum()), -1) @ w.T).reshape(-1, fe.num_sym, fe.num_sc)
        nmse.setdefault(snr, {})["lmmse_data"] = float(10 * torch.log10((out - vg).abs().pow(2).mean() / p))
        log(f"data-LMMSE val SNR {snr:3d}: NMSE {nmse[snr]['lmmse_data']:.1f} dB")
    return nmse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models/paper_ce"))
    ap.add_argument("--channel", default="uma")
    ap.add_argument("--prb", type=int, default=68)
    ap.add_argument("--train-slots", type=int, default=512, help="slots per UE/rank configuration")
    ap.add_argument("--val-slots", type=int, default=128)
    ap.add_argument("--batch-slots", type=int, default=16)
    ap.add_argument("--max-speed-mps", type=float, default=35.0)
    ap.add_argument("--denoise-steps", type=int, default=6000)
    ap.add_argument("--ammse-steps", type=int, default=4000)
    ap.add_argument("--ammse-warm-steps", type=int, default=3000)
    ap.add_argument("--ra-steps", type=int, default=400)
    ap.add_argument("--ra-rank", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    t0 = time.time()

    *tr, fe = make_pool(args.channel, args.train_slots, args.batch_slots, args.prb, args.seed, args.max_speed_mps)
    *va, _ = make_pool(args.channel, args.val_slots, args.batch_slots, args.prb, args.seed + 1000, args.max_speed_mps)
    log(f"train {tr[0].shape[0]} / val {va[0].shape[0]} (rx antenna, layer) samples")

    r = train_covariance(tr[1], args.out)
    nmse = train_denoise(fe, tr, va, args.out, args.denoise_steps, batch=256)
    nmse = reference_nmse(fe, va, r, nmse)
    ammse = train_ammse(fe, tr, va, args.out, args.ammse_steps, args.ammse_warm_steps, args.ra_steps,
                        args.ra_rank, batch=512)
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "configs": CONFIGS,
            "train_samples": int(tr[0].shape[0]), "val_samples": int(va[0].shape[0]),
            "val_nmse_db_full_grid": nmse, "val_nmse_db_ammse_chunks": ammse,
            "duration_s": time.time() - t0}
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
