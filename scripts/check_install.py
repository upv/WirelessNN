#!/usr/bin/env python3
"""Check the installation:  .venv/bin/python scripts/check_install.py

Prints the library versions, the device the simulator will use, whether the trained
channel-estimator models are present, and runs a three-point simulation. Exits with
a non-zero status if anything required is missing or the simulation fails.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = ("torch", "sionna", "numpy", "matplotlib")
OPTIONAL = {
    "scipy": "tests, scripts/physics_checks.py",
    "pytest": "tests",
    "sklearn": "scripts/training/train_wp_models*.py",
    "lightgbm": "scripts/training/train_wp_models*.py",
}
MODEL_FILES = ("data_cov.pt", "denoise_nn.pt", "a_mmse.pt")


def main() -> int:
    ok = True
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"{name:11s} {getattr(mod, '__version__', '?')}")
        except Exception as exc:
            ok = False
            print(f"{name:11s} MISSING ({exc}) -> pip install -r requirements.txt")
    for name, used_by in OPTIONAL.items():
        try:
            mod = importlib.import_module(name)
            print(f"{name:11s} {getattr(mod, '__version__', '?')}")
        except Exception:
            print(f"{name:11s} not installed (optional: {used_by})")
    if not ok:
        return 1

    import torch

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}       {props.name}, {props.total_memory / 2**30:.0f} GB")
    else:
        print("GPU         none visible to PyTorch -> simulations run on the CPU")

    model_dir = ROOT / "models" / "paper_ce"
    missing = [f for f in MODEL_FILES if not (model_dir / f).exists()]
    if missing:
        print(f"models      {model_dir} lacks {', '.join(missing)}: the trained estimators "
              "(denoise_nn, lmmse_data*, a_mmse, ra_a_mmse) are unavailable until "
              "scripts/training/train_paper_ce.py is run")
    else:
        print(f"models      trained channel estimators found in {model_dir}")

    from nr_ul_sim import NRUplinkSimulator, SimConfig

    print("\nsmoke simulation: CDL-C, QPSK, 8 PRB, MR / L-MMSE / IRC, SNR -10 / 0 / 10 dB")
    cfg = SimConfig(num_prb=8, snr_db=[-10.0, 0.0, 10.0], iot_db=[0.0],
                    receivers=("mr", "lmmse", "irc"), batch_size=2, max_mc_iter=2)
    t0 = time.time()
    sim = NRUplinkSimulator(cfg)
    campaign = sim.run(verbose=False)
    print(f"device      {sim.device}   ({time.time() - t0:.1f} s)")
    for name, curve in campaign["iot"]["0.0"].items():
        print(f"  {name:6s} BER " + "  ".join(f"{b:.2e}" for b in curve["ber"]))
        if not curve["ber"][0] > curve["ber"][-1]:
            ok = False
            print("         ^ BER does not fall with SNR: something is wrong")
    print("\nOK - next: python examples/01_quickstart.py, then docs/TUTORIAL.md" if ok
          else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
