#!/usr/bin/env python3
"""One PUSCH slot, block by block:  python examples/02_one_slot_step_by_step.py

Opens the simulator up: instead of ``sim.run()`` it calls the stages that
``run()`` chains for every Monte-Carlo iteration and prints what each one
produces — tensor shapes, powers, channel-estimation error and the BER of every
receiver on the very same slot. Runs on a CPU in a few seconds (8 PRB).

    transmit -> channel -> + interference + noise -> channel estimation
             -> interference covariance (IRC) -> equalise -> LDPC decode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import torch  # noqa: E402

from nr_ul_sim import NRUplinkSimulator, SimConfig  # noqa: E402
from nr_ul_sim.interference import irc_interference_covariance  # noqa: E402
from nr_ul_sim.pusch import strip_guard_subcarriers  # noqa: E402

SNR_DB = 12.0   # per receive antenna, per resource element, for one UE
IOT_DB = 10.0   # total other-cell interference over thermal noise (0 = none)
BATCH = 8       # slots simulated in parallel

cfg = SimConfig(
    channel="cdl-c",
    num_ue=2,                # two co-scheduled UEs (MU-MIMO)
    num_layers=1,            # rank 1 each
    num_rx_ant=4,
    modulation="qam16",
    num_prb=8,               # 68 is the full allocation; 8 keeps this instant
    num_interferers=1,
    receivers=("mr", "zf", "lmmse", "irc"),
    channel_estimators=("perfect", "ls_lin", "lmmse_exp"),
    seed=1,
)
sim = NRUplinkSimulator(cfg)


def power(x: torch.Tensor) -> float:
    return float(x.abs().pow(2).mean())


def shape(x: torch.Tensor) -> str:
    return str(list(x.shape))


print(f"device: {sim.device}")
print(f"grid: {cfg.num_prb} PRB = {cfg.num_used_subcarriers} used subcarriers, FFT {cfg.fft_size}, "
      f"guards {cfg.guard_carriers}, {cfg.num_ofdm_symbols} OFDM symbols")
print(f"DMRS on OFDM symbols {sim.rx.dmrs_syms}; MCS (table, index) = {cfg.resolved_mcs}")

# 1) Transmitter: random bits -> LDPC -> QAM -> layers -> grid with DMRS ------
x, bits = sim.transmit(BATCH)
print("\n1) transmit")
print(f"   bits  {shape(bits)}   [batch, UE, transport-block bits]")
print(f"   x     {shape(x)}   [batch, UE, UE antenna, OFDM symbol, FFT bin]")

# 2-3) Channel, interference and noise -------------------------------------------
# generate_slot() draws fresh bits, serving and interferer channels and returns
# everything the receiver (and a genie) could want.
slot = sim.generate_slot(BATCH, SNR_DB, IOT_DB)
h, h_int, y, no = slot["h"], slot["h_int"], slot["y"], slot["no"]
print("\n2) channel (frequency domain, one tap per subcarrier)")
print(f"   h     {shape(h)}   [batch, 1, RX ant, UE, UE ant, symbol, FFT bin]")
print(f"   h_int {shape(h_int)}   same layout for the interferers")
print(f"   mean |h|^2 on the used band = {power(strip_guard_subcarriers(h, cfg.guard_carriers)):.3f}"
      "   (normalised to 1)")
print("\n3) received grid  y = H x + sqrt(INR N0 / streams) H_int x_int + noise")
print(f"   y     {shape(y)}   [batch, 1, RX ant, symbol, FFT bin]")
print(f"   N0 = 10^(-SNR/10) = {no:.4f};   interference power = INR x N0 = "
      f"{10 ** (IOT_DB / 10) * no:.4f} per antenna")

# 4) Channel estimation ------------------------------------------------------------
sim.estimate_csi(slot)                       # fills slot["csi"][name] = (h_hat, err_var)
h_true = slot["h_perf"]                      # true channel on the used subcarriers
print("\n4) channel estimation  (NMSE = |h_hat - h|^2 / |h|^2)")
print(f"   h_hat {shape(h_true)}   [batch, 1, RX ant, UE, stream, symbol, used subcarrier]")
for name, (h_hat, err_var) in slot["csi"].items():
    nmse = power(h_hat - h_true) / power(h_true)
    nmse_db = 10 * torch.log10(torch.tensor(max(nmse, 1e-12))).item()
    print(f"   {name:10s} NMSE = {nmse_db:7.2f} dB   reported error variance = {float(err_var.mean()):.4f}")

# 5) Interference covariance used by IRC -------------------------------------------
r_iot = irc_interference_covariance(h_int, y, no, IOT_DB, method=cfg.iot_cov)
eig = torch.linalg.eigvalsh(r_iot[0]).flip(0)
print(f"\n5) IRC covariance R_iot {shape(r_iot)} (method '{cfg.iot_cov}')")
print("   eigenvalues / N0 of slot 0: " + ", ".join(f"{float(e) / float(no):.2f}" for e in eig)
      + "   <- few strong directions = interference IRC can null")

# 6) Equalise + demap + LDPC decode, every receiver on this same slot ---------
print(f"\n6) detection: BER / BLER on these {BATCH} slots (SNR {SNR_DB} dB, IoT {IOT_DB} dB)")
print(f"   {'receiver':8s} " + " ".join(f"{e:>22s}" for e in cfg.channel_estimators))
for receiver in cfg.receivers:
    cells = []
    for estimator in cfg.channel_estimators:
        b_hat = sim.detect(slot, receiver, estimator)
        err = b_hat != slot["b"]
        bler = err.reshape(-1, err.shape[-1]).any(dim=1).float().mean()
        cells.append(f"{float(err.float().mean()):.2e} / {float(bler):.2f}")
    print(f"   {receiver:8s} " + " ".join(f"{c:>22s}" for c in cells))

print("\nsim.run() repeats steps 1-6 until enough errors are counted, for every SNR and IoT value,")
print("then interpolates the working point: the SNR where BER = 0.01 (and where BLER = 0.1).")
