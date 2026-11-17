"""Train Channelformer (paper Table I: Adam 2e-3, halve every 50 epochs, 100 epochs,
batch 128, L2 1e-7, Huber delta 1) on gen_data.py output.

    PYTHONPATH=.:experiments/channelformer python experiments/channelformer/train.py --data data/cf_4x2 --out runs/cf_4x2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import BLOCK_SC, WIN_BLOCKS, Channelformer, estimate_slices, to_tokens


def load(data: Path, name: str):
    blk = torch.from_numpy(np.load(data / f"{name}_blk.npy"))
    tgt = torch.from_numpy(np.load(data / f"{name}_tgt.npy", mmap_mode="r")[:])
    snr = np.load(data / f"{name}_snr.npy")
    return blk, tgt, snr


def batch(blk, tgt, idx, dev, rng):
    m = blk.shape[-1]
    starts = torch.as_tensor(rng.integers(0, m - WIN_BLOCKS + 1, len(idx)))
    ar = torch.arange(WIN_BLOCKS)
    b = blk[idx]                                                    # [n, Nd, M]
    gi = (starts[:, None] + ar).unsqueeze(1).expand(-1, b.shape[1], -1)
    win = torch.gather(b, 2, gi)
    sc = (starts[:, None] * BLOCK_SC + torch.arange(WIN_BLOCKS * BLOCK_SC))
    t = tgt[idx].float()                                            # [n, 14, N, 2]
    t = torch.gather(t, 2, sc[:, None, :, None].expand(-1, t.shape[1], -1, 2))
    tok, scale = to_tokens(win)
    return tok.to(dev), (t / scale[:, None, None, None]).to(dev)


@torch.no_grad()
def evaluate(net, blk, tgt, snr, dev, bins):
    """Full-slot assembly on validation slices: MSE and NMSE per SNR bin."""
    net.eval()
    err = np.zeros(len(blk))
    pw = np.zeros(len(blk))
    for i in range(0, len(blk), 1024):
        h = estimate_slices(net, blk[i:i + 1024].to(dev))
        t = tgt[i:i + 1024].to(dev).float()
        t = torch.complex(t[..., 0], t[..., 1])
        err[i:i + 1024] = (h - t).abs().pow(2).mean(dim=(1, 2)).cpu().numpy()
        pw[i:i + 1024] = t.abs().pow(2).mean(dim=(1, 2)).cpu().numpy()
    net.train()
    mse, nmse = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (snr >= lo) & (snr < hi)
        mse.append(float(err[sel].mean()) if sel.any() else float("nan"))
        nmse.append(10 * math.log10(err[sel].sum() / pw[sel].sum()) if sel.any() else float("nan"))
    return mse, nmse, 10 * math.log10(err.sum() / pw.sum())


def fill_nan(x, y):
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    return np.exp(np.interp(x, x[ok], np.log(y[ok]))).tolist() if ok.any() else y.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--drop-every", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--l2", type=float, default=1e-7)
    ap.add_argument("--windows-per-slice", type=int, default=1, help="random windows per slice per epoch")
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tr_blk, tr_tgt, _ = load(args.data, "train")
    va_blk, va_tgt, va_snr = load(args.data, "val")
    print(f"train slices {len(tr_blk)}, val slices {len(va_blk)}, device {dev}", flush=True)
    net = Channelformer().to(dev)
    print("parameters", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2)
    sched = torch.optim.lr_scheduler.StepLR(opt, args.drop_every, 0.5)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    bins = np.arange(-11.0, 33.0, 2.0)                    # centres -10 .. 30
    centres = (bins[:-1] + bins[1:]) / 2
    log, best, t0 = [], float("inf"), time.time()
    for ep in range(args.epochs):
        perm = torch.as_tensor(np.tile(rng.permutation(len(tr_blk)), args.windows_per_slice))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            x, y = batch(tr_blk, tr_tgt, perm[i:i + args.batch], dev, rng)
            loss = loss_fn(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sched.step()
        mse, nmse, overall = evaluate(net, va_blk, va_tgt, va_snr, dev, bins)
        log.append({"epoch": ep + 1, "train_huber": tot / len(perm), "val_nmse_db": overall,
                    "val_nmse_db_by_snr": nmse, "minutes": (time.time() - t0) / 60})
        print(f"ep {ep + 1:3d} train {tot / len(perm):.5f} val NMSE {overall:.2f} dB "
              f"(-10: {nmse[0]:.1f}, 10: {nmse[10]:.1f}, 30: {nmse[-1]:.1f}) "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        if overall < best:
            mse = fill_nan(centres, mse)
            best = overall
            torch.save({"config": net.config, "state_dict": net.state_dict(),
                        "errvar_snr_db": centres.tolist(), "errvar_mse": mse,
                        "val_nmse_db_by_snr": nmse, "val_nmse_db": overall, "epoch": ep + 1,
                        "args": {k: str(v) for k, v in vars(args).items()}},
                       args.out / "channelformer.pt")
        (args.out / "log.json").write_text(json.dumps(log, indent=1))
        if time.time() - t0 > args.max_hours * 3600:
            print("time limit reached", flush=True)
            break
    print(f"best val NMSE {best:.2f} dB", flush=True)


if __name__ == "__main__":
    main()
