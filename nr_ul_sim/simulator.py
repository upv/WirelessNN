from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .channels import build_cdl_links, build_system_level_channel, set_system_topology
from .interference import (
    irc_interference_covariance,
    random_ofdm_grid,
    serving_plus_interference,
)
from .metrics import ErrorStats, SweepResult, working_point_snr
from .parameters import (
    SimConfig,
    detection_keys,
    receiver_uses_perfect_csi,
    snrdb_to_noise_var,
)
from .pusch import build_transmitter
from .receivers import build_receivers, effective_channel


def _select_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


class NRUplinkSimulator:
    def __init__(self, cfg: SimConfig):
        import sionna.phy
        from sionna.phy.channel import ApplyOFDMChannel, GenerateOFDMChannel

        self.device = _select_device()
        sionna.phy.config.device = "cuda:0" if self.device.type == "cuda" else "cpu"
        sionna.phy.config.seed = cfg.seed
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.cfg = cfg
        self.transmitter = build_transmitter(cfg)
        self.rx = build_receivers(self.transmitter, cfg)
        self.apply_channel = ApplyOFDMChannel()

        rg = self.transmitter.resource_grid
        self.serving_model = self._make_model(cfg.num_ue)
        self.gen_serving = GenerateOFDMChannel(self.serving_model, rg, normalize_channel=True)
        self.int_model = None
        self.gen_int = None
        if cfg.num_interferers > 0:
            self.int_model = self._make_model(cfg.num_interferers)
            self.gen_int = GenerateOFDMChannel(self.int_model, rg, normalize_channel=True)

    def _make_model(self, num_tx: int):
        if self.cfg.channel in ("umi", "uma"):
            return build_system_level_channel(self.cfg)
        return build_cdl_links(self.cfg, num_tx)

    def transmit(self, batch_size: int):
        """PUSCH grid of every UE with the configured per-UE power normalisation."""
        x, b = self.transmitter(batch_size)
        scale = self.cfg.layer_power_scale
        if scale != 1.0:
            x = x * scale
        return x, b

    def interferer_grid(self, batch_size: int, like: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        return random_ofdm_grid(
            batch_size, cfg.num_interferers, cfg.resolved_ue_ant,
            cfg.num_ofdm_symbols, cfg.fft_size,
            dtype=like.real.dtype, device=like.device,
            guard_carriers=cfg.guard_carriers,
        )

    def generate_slot(self, batch_size: int, snr_db: float, iot_db: float):
        cfg = self.cfg
        if cfg.channel in ("umi", "uma"):
            set_system_topology(self.serving_model, cfg, batch_size, cfg.num_ue)
            if iot_db > 0.0 and self.int_model is not None:
                set_system_topology(self.int_model, cfg, batch_size, cfg.num_interferers)

        x, b = self.transmit(batch_size)
        no = snrdb_to_noise_var(snr_db)

        h_s = self.gen_serving(batch_size)
        y_s = self.apply_channel(x, h_s)

        h_i = y_i = None
        if iot_db > 0.0 and self.gen_int is not None:
            h_i = self.gen_int(batch_size)
            y_i = self.apply_channel(self.interferer_grid(batch_size, x), h_i)

        return {
            "b": b,
            "y": serving_plus_interference(y_s, y_i, no, iot_db, cfg.num_interferer_streams),
            "h": h_s,
            "h_int": h_i,
            "no": no,
            "iot_db": iot_db,
        }

    def _keys(self) -> list[tuple[str, str, str]]:
        return detection_keys(self.cfg.receivers, self.cfg.channel_estimators)

    def true_channel(self, h):
        return effective_channel(h, self.cfg.guard_carriers, self.rx.w, self.cfg.layer_power_scale)

    def estimate_csi(self, slot: dict[str, Any]) -> None:
        y, no = slot["y"], slot["no"]
        if not torch.is_tensor(no):
            no = torch.tensor(no, device=y.device, dtype=y.real.dtype)
            slot["no"] = no
        need_perf = "perfect" in self.cfg.channel_estimators or any(
            receiver_uses_perfect_csi(n, self.cfg.perfect_csi) for n in self.cfg.receivers
        )
        if need_perf:
            slot["h_perf"] = self.true_channel(slot["h"])
        slot["csi"] = {}
        default = None
        for name in self.cfg.channel_estimators:
            if name == "perfect":
                h_hat = slot["h_perf"]
                err_var = torch.zeros((), dtype=h_hat.real.dtype, device=h_hat.device)
            else:
                h_hat, err_var = self.rx.estimate(y, no, name)
            slot["csi"][name] = (h_hat, err_var)
            if default is None and name != "perfect":
                default = name
        if default is None:
            default = self.cfg.channel_estimators[0]
        slot["h_hat"], slot["err_var"] = slot["csi"][default]

    def detect(self, slot: dict[str, Any], receiver_name: str, estimator: str | None = None):
        if "csi" not in slot and "h_hat" not in slot:
            self.estimate_csi(slot)
        y, no = slot["y"], slot["no"]

        if estimator is None:
            estimator = self.cfg.channel_estimators[0]
        use_true = estimator == "perfect" or receiver_uses_perfect_csi(
            receiver_name, self.cfg.perfect_csi
        )
        if use_true:
            if "h_perf" not in slot:
                slot["h_perf"] = self.true_channel(slot["h"])
            h_hat = slot["h_perf"]
            err_var = torch.zeros((), dtype=h_hat.real.dtype, device=h_hat.device)
        elif "csi" in slot:
            h_hat, err_var = slot["csi"][estimator]
        else:
            h_hat, err_var = slot["h_hat"], slot["err_var"]

        if receiver_name == "irc":
            self.rx.irc_eq.set_covariance(
                irc_interference_covariance(
                    slot["h_int"], y, no, slot["iot_db"], method=self.cfg.iot_cov,
                    h_hat=h_hat, pilot_grid=self.rx.pilot_grid, dmrs_syms=self.rx.dmrs_syms,
                    guard_carriers=self.cfg.guard_carriers,
                )
            )
        else:
            self.rx.irc_eq.set_covariance(0.0)
        return self.rx.decode(y, h_hat, err_var, no, receiver_name)

    def _enough_errors(self, stats: dict[str, ErrorStats]) -> bool:
        cfg = self.cfg
        return all(
            s.bit_errors >= cfg.num_target_bit_errors
            and s.block_errors >= cfg.num_target_block_errors
            for s in stats.values()
        )

    def measure_point(self, snr_db: float, iot_db: float) -> dict[str, ErrorStats]:
        """Monte-Carlo BER/BLER of every (receiver, estimator) at one (SNR, IoT) point."""
        cfg = self.cfg
        keys = self._keys()
        stats = {key: ErrorStats() for key, _, _ in keys}
        for _ in range(cfg.max_mc_iter):
            slot = self.generate_slot(cfg.batch_size, snr_db, iot_db)
            self.estimate_csi(slot)
            b_np = slot["b"].detach().cpu().numpy()
            for key, receiver, estimator in keys:
                stats[key].update(
                    b_np, self.detect(slot, receiver, estimator).detach().cpu().numpy()
                )
            if self._enough_errors(stats):
                break
        return stats

    def run(self, snr_db=None, iot_db=None, verbose: bool = True) -> dict[str, Any]:
        cfg = self.cfg
        snr_db = list(cfg.snr_db if snr_db is None else snr_db)
        iot_db = list(cfg.iot_db if iot_db is None else iot_db)
        campaign = {"config": cfg.summary(), "device": str(self.device), "iot": {}}
        t0 = time.time()
        for iot in iot_db:
            campaign["iot"][str(iot)] = self._sweep_snr(snr_db, iot, verbose)
        campaign["duration_s"] = time.time() - t0
        return campaign

    def _sweep_snr(self, snr_db: list[float], iot_db: float, verbose: bool):
        cfg = self.cfg
        names = [key for key, _, _ in self._keys()]
        results = {name: SweepResult(snr_db=list(snr_db)) for name in names}
        if verbose:
            csi = "perfect" if cfg.perfect_csi else ",".join(cfg.channel_estimators)
            if not cfg.perfect_csi and "ideal_mmse" in cfg.receivers:
                csi += "; Ideal MMSE uses true H"
            print(
                f"\n=== {cfg.channel.upper()} | {cfg.modulation.upper()} | "
                f"{cfg.num_ue} UE x rank {cfg.num_layers} | "
                f"IoT={iot_db} dB | CSI={csi} ==="
            )

        for snr in snr_db:
            t_snr = time.time()
            stats = self.measure_point(snr, iot_db)
            if verbose:
                parts = [f"SNR {snr:6.1f} dB"]
                parts += [f"{n} BER={s.ber:.3e} BLER={s.bler:.2e}" for n, s in stats.items()]
                parts.append(f"[{time.time() - t_snr:.1f} s]")
                print("  " + " | ".join(parts))
            for name, st in stats.items():
                results[name].ber.append(st.ber)
                results[name].bler.append(st.bler)
                results[name].stats.append(st.as_dict())

        out = {}
        for name, res in results.items():
            res.working_point_db = working_point_snr(res.snr_db, res.ber, cfg.target_ber)
            res.working_point_bler_db = working_point_snr(res.snr_db, res.bler, cfg.target_bler)
            out[name] = res.as_dict()
            if verbose:
                wp, wpb = res.working_point_db, res.working_point_bler_db
                txt = f"{wp:.2f} dB" if wp is not None else "not reached"
                txtb = f"{wpb:.2f} dB" if wpb is not None else "not reached"
                print(
                    f"  {name:16s} working point BER={cfg.target_ber:g}: {txt:12s} "
                    f"BLER={cfg.target_bler:g}: {txtb}"
                )
        return out
