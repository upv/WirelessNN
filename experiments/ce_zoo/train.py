"""Train one ce_zoo model on the Channelformer 4x2 dataset (channelformer/gen_data.py output).

    PYTHONPATH=.:experiments/ce_zoo python experiments/ce_zoo/train.py --model chea --data data/cf_4x2 --out runs/zoo/chea

Training samples are random contiguous PRB crops (4..68 PRB, one length per batch,
full band with probability --p-full) of the per-(rx antenna, layer) slices; validation
is always on the whole 68-PRB band, same val split as Channelformer, so the numbers are
directly comparable with the Channelformer checkpoint's "val_nmse_db_by_snr".
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from models import BLOCK_SC, BLOCKS_PER_PRB, LSGrid, build, estimate_slices, to_input


def load(data: Path, name: str):
    blk = torch.from_numpy(np.load(data / f"{name}_blk.npy"))
    tgt = torch.from_numpy(np.load(data / f"{name}_tgt.npy", mmap_mode="r")[:])
    snr = np.load(data / f"{name}_snr.npy")
    mp = data / f"{name}_meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else ["all"] * len(blk)
    return blk, tgt, snr, meta


def load_many(dirs, name):
    parts = [load(d, name) for d in dirs]
    if len(parts) == 1:
        return parts[0]
    return (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]), sum((list(p[3]) for p in parts), []))


def channel_of(m):
    if isinstance(m, (list, tuple)):
        return str(m[0])
    if isinstance(m, dict):
        return str(m.get("channel", "all"))
    return str(m)


def make_batch(blk, tgt, idx, n_prb, dev, rng, tau_max_s=0.0, scs_hz=30e3):
    m_all = blk.shape[-1]
    p_all = m_all // BLOCKS_PER_PRB
    p0 = int(rng.integers(0, p_all - n_prb + 1))
    b0, b1 = p0 * BLOCKS_PER_PRB, (p0 + n_prb) * BLOCKS_PER_PRB
    b = blk[idx, :, b0:b1]
    t = tgt[idx, :, b0 * BLOCK_SC:b1 * BLOCK_SC].float()                         # [n, S, F, 2]
    if tau_max_s > 0:
        # timing-offset augmentation: a delay tau is a phase ramp exp(-j 2 pi k df tau) over
        # subcarriers k; applied to both the LS input (block centres) and the target
        tau = torch.as_tensor(rng.uniform(-tau_max_s, tau_max_s, len(b)), dtype=torch.float64)
        k_sc = torch.arange(t.shape[2], dtype=torch.float64)
        k_bl = torch.arange(b.shape[-1], dtype=torch.float64) * BLOCK_SC + (BLOCK_SC - 1) / 2
        ph_sc = torch.polar(torch.ones(1, dtype=torch.float64), -2 * math.pi * scs_hz * tau[:, None] * k_sc)
        ph_bl = torch.polar(torch.ones(1, dtype=torch.float64), -2 * math.pi * scs_hz * tau[:, None] * k_bl)
        b = b * ph_bl[:, None, :].to(b.dtype)
        tc = torch.complex(t[..., 0], t[..., 1]) * ph_sc[:, None, :].to(torch.complex64)
        t = torch.stack([tc.real, tc.imag], -1)
    x, scale = to_input(b)
    t = t.permute(0, 3, 1, 2)                                                      # [n, 2, S, F]
    return x.to(dev), (t / scale[:, None, None, None]).to(dev)


def ramp(tau_s, n, scs_hz, offset=0.0):
    """Phase ramps exp(-j 2 pi (k + offset) df tau) [len(tau), n] complex64."""
    k = torch.arange(n, dtype=torch.float64) + offset
    tau = torch.as_tensor(tau_s, dtype=torch.float64)
    return torch.polar(torch.ones(1, dtype=torch.float64), -2 * math.pi * scs_hz * tau[:, None] * k).to(torch.complex64)


@torch.no_grad()
def evaluate(net, blk, tgt, dev, bs=512, tau_s=None, scs_hz=30e3):
    """Whole-band error per slice; tau_s (per slice, seconds) applies a timing offset."""
    net.eval()
    err = np.zeros(len(blk))
    pw = np.zeros(len(blk))
    for i in range(0, len(blk), bs):
        b = blk[i:i + bs]
        t = tgt[i:i + bs].float()
        t = torch.complex(t[..., 0], t[..., 1])
        if tau_s is not None:
            tt = tau_s[i:i + bs]
            # block i is centred on subcarrier 4 i + 1.5
            b = b * ramp(tt, b.shape[-1], scs_hz * BLOCK_SC, (BLOCK_SC - 1) / 2 / BLOCK_SC)[:, None, :]
            t = t * ramp(tt, t.shape[-1], scs_hz)[:, None, :]
        h = estimate_slices(net, b.to(dev))
        t = t.to(dev)
        err[i:i + bs] = (h - t).abs().pow(2).mean(dim=(1, 2)).cpu().numpy()
        pw[i:i + bs] = t.abs().pow(2).mean(dim=(1, 2)).cpu().numpy()
    net.train()
    return err, pw


def summarise(err, pw, snr, meta, bins):
    by_snr, mse = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (snr >= lo) & (snr < hi)
        by_snr.append(10 * math.log10(err[sel].sum() / pw[sel].sum()) if sel.any() else float("nan"))
        mse.append(float(err[sel].mean()) if sel.any() else float("nan"))
    ch = np.array([channel_of(m) for m in meta])
    by_ch = {c: 10 * math.log10(err[ch == c].sum() / pw[ch == c].sum()) for c in sorted(set(ch))}
    return 10 * math.log10(err.sum() / pw.sum()), by_snr, mse, by_ch


class Identity(torch.nn.Module):
    def __init__(self, dmrs, n_sym):
        super().__init__()
        self.ls = LSGrid(dmrs, n_sym)

    def forward(self, x):
        return self.ls(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, nargs="+", required=True,
                    help="one or more gen_data.py folders with the same layout (concatenated)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dmrs", default="2,11", help="DMRS symbol indices in the slot")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=("huber", "mse"), default="huber")
    ap.add_argument("--min-prb", type=int, default=4)
    ap.add_argument("--p-full", type=float, default=0.3)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--tau-max-ns", type=float, default=0.0,
                    help="timing-offset augmentation: uniform delay in [-tau, tau] ns (0 = off)")
    ap.add_argument("--scs-khz", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dmrs = tuple(int(s) for s in args.dmrs.split(","))

    tr_blk, tr_tgt, _, _ = load_many(args.data, "train")
    va_blk, va_tgt, va_snr, va_meta = load_many(args.data, "val")
    p_all = tr_blk.shape[-1] // BLOCKS_PER_PRB
    bins = np.arange(-11.0, 33.0, 2.0)
    centres = (bins[:-1] + bins[1:]) / 2
    print(f"train {len(tr_blk)} val {len(va_blk)} slices, {p_all} PRB, Nd {tr_blk.shape[1]}, dev {dev}", flush=True)

    # fixed LS interpolation baseline (also a DMRS-position sanity check)
    e, p = evaluate(Identity(dmrs, va_tgt.shape[1]).to(dev), va_blk, va_tgt, dev)
    ls_overall, ls_snr, _, ls_ch = summarise(e, p, va_snr, va_meta, bins)
    print(f"LS-interp val NMSE {ls_overall:.2f} dB  (10 dB: {ls_snr[10]:.1f}, 30 dB: {ls_snr[-1]:.1f})", flush=True)

    n_sym = tr_tgt.shape[1]
    assert len(dmrs) == tr_blk.shape[1], f"--dmrs has {len(dmrs)} symbols, data has Nd={tr_blk.shape[1]}"
    kw = dict(nd=tr_blk.shape[1], dmrs=dmrs, n_sym=n_sym)
    if args.model.endswith("_window") or args.model == "lmmse_exp":
        kw["scs_khz"] = args.scs_khz
    net = build(args.model, **kw).to(dev)
    n_par = sum(q.numel() for q in net.parameters())
    print(f"{args.model}: {type(net).__name__} {n_par} params", flush=True)
    if n_par == 0:            # classical estimator (windows, lmmse_exp): evaluate only
        args.epochs = 1
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        steps_ep = math.ceil(len(tr_blk) / args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_ep,
                                                    pct_start=0.05)
    loss_fn = torch.nn.HuberLoss(delta=1.0) if args.loss == "huber" else torch.nn.MSELoss()
    va_tau = None
    if args.tau_max_ns > 0:   # fixed random offsets for a second validation metric
        va_tau = np.random.default_rng(12345).uniform(-args.tau_max_ns, args.tau_max_ns, len(va_blk)) * 1e-9
    log, best, t0 = [], float("inf"), time.time()
    for ep in range(args.epochs):
        perm = torch.as_tensor(rng.permutation(len(tr_blk)))
        tot = 0.0
        for i in range(0, len(perm) if n_par else 0, args.batch):
            n_prb = p_all if rng.random() < args.p_full else int(rng.integers(args.min_prb, p_all + 1))
            x, y = make_batch(tr_blk, tr_tgt, perm[i:i + args.batch], n_prb, dev, rng,
                              args.tau_max_ns * 1e-9, args.scs_khz * 1e3)
            loss = loss_fn(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        e, p = evaluate(net, va_blk, va_tgt, dev)
        overall, by_snr, mse, by_ch = summarise(e, p, va_snr, va_meta, bins)
        sel_metric, tau_overall = overall, None
        if va_tau is not None:
            et, pt = evaluate(net, va_blk, va_tgt, dev, tau_s=va_tau, scs_hz=args.scs_khz * 1e3)
            tau_overall, _, mse, _ = summarise(et, pt, va_snr, va_meta, bins)
            sel_metric = tau_overall
        mins = (time.time() - t0) / 60
        log.append({"epoch": ep + 1, "train_loss": tot / len(perm), "val_nmse_db": overall,
                    "val_tau_nmse_db": tau_overall,
                    "val_nmse_db_by_snr": by_snr, "val_nmse_db_by_channel": by_ch, "minutes": mins})
        tau_txt = f" val+tau {tau_overall:.2f} dB" if tau_overall is not None else ""
        print(f"ep {ep + 1:3d} loss {tot / len(perm):.5f} val {overall:.2f} dB{tau_txt} "
              f"(-10: {by_snr[0]:.1f}, 10: {by_snr[10]:.1f}, 30: {by_snr[-1]:.1f}) [{mins:.1f} min]", flush=True)
        if sel_metric < best:
            best = sel_metric
            ok = np.isfinite(mse)
            mse_f = np.exp(np.interp(centres, centres[ok], np.log(np.asarray(mse)[ok]))).tolist()
            torch.save({"name": args.model, "class": type(net).__name__, "config": net.config,
                        "state_dict": net.state_dict(), "params": n_par,
                        "errvar_snr_db": centres.tolist(), "errvar_mse": mse_f,
                        "val_nmse_db": overall, "val_tau_nmse_db": tau_overall,
                        "tau_max_ns": args.tau_max_ns, "val_nmse_db_by_snr": by_snr,
                        "val_nmse_db_by_channel": by_ch, "epoch": ep + 1,
                        "args": {k: str(v) for k, v in vars(args).items()}}, args.out / "model.pt")
        (args.out / "log.json").write_text(json.dumps(log, indent=1))
        if time.time() - t0 > args.max_hours * 3600:
            print("time limit reached", flush=True)
            break
    (args.out / "summary.json").write_text(json.dumps({
        "model": args.model, "params": n_par, "best_val_nmse_db": best,
        "selected_on": "val_with_timing_offsets" if va_tau is not None else "val",
        "ls_interp_val_nmse_db": ls_overall, "ls_interp_by_snr": ls_snr, "ls_interp_by_channel": ls_ch,
        "snr_centres": centres.tolist(), "minutes": (time.time() - t0) / 60}, indent=1))
    print(f"best val NMSE {best:.2f} dB", flush=True)


if __name__ == "__main__":
    main()
