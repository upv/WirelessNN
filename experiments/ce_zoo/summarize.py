"""Markdown table of every trained ce_zoo model vs Channelformer and LS interpolation
(same validation split).

    python experiments/ce_zoo/summarize.py runs/zoo [--channelformer experiments/channelformer/weights/channelformer_4x2.pt]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CF_DEFAULT = REPO / "experiments/channelformer/weights/channelformer_4x2.pt"
PICK = [0, 5, 10, 15, 20]                       # SNR bins -10, 0, 10, 20, 30 dB
CHANNELS = ("cdl-b", "cdl-c", "cdl-d", "umi", "uma")


def row(name, params, overall, by_snr, by_ch=None):
    snr = " | ".join(f"{by_snr[i]:.1f}" for i in PICK)
    ch = " | ".join(f"{by_ch[c]:.1f}" if by_ch and c in by_ch else "" for c in CHANNELS)
    return f"| {name} | {params} | {overall:.2f} | {snr} | {ch} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--channelformer", type=Path, default=CF_DEFAULT)
    args = ap.parse_args()
    print("| model | params | val NMSE [dB] | -10 | 0 | 10 | 20 | 30 dB | CDL-B | CDL-C | CDL-D | UMi | UMa |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    states, ls = [], None
    for ck in sorted(args.runs.glob("*/model.pt")):
        states.append(torch.load(ck, map_location="cpu", weights_only=False))
        sj = ck.parent / "summary.json"
        if sj.exists():
            ls = json.loads(sj.read_text())
    for s in sorted(states, key=lambda s: s["val_nmse_db"]):
        print(row(s["name"], s["params"], s["val_nmse_db"], s["val_nmse_db_by_snr"], s["val_nmse_db_by_channel"]))
    if args.channelformer.exists():
        s = torch.load(args.channelformer, map_location="cpu", weights_only=False)
        n = sum(v.numel() for v in s["state_dict"].values())
        print(row("channelformer", n, s["val_nmse_db"], s["val_nmse_db_by_snr"]))
    if ls:
        print(row("LS interpolation", 0, ls["ls_interp_val_nmse_db"], ls["ls_interp_by_snr"], ls["ls_interp_by_channel"]))


if __name__ == "__main__":
    main()
