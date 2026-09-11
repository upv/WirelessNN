"""End-to-end NR PUSCH uplink simulator."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .channels import (
    build_cdl_links,
    build_system_level_channel,
    is_system_level,
    set_system_topology,
)
from .interference import (
    irc_interference_covariance,
    random_ofdm_grid,
    serving_plus_interference,
)
from .metrics import ErrorStats, SweepResult, working_point_snr
from .parameters import SimConfig, snrdb_to_noise_var
from .pusch import build_transmitter
from .receivers import build_receivers


def _select_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


class NRUplinkSimulator:
    """5G NR PUSCH uplink Monte-Carlo simulator.

    One object holds the transmitter, channel models and all linear receivers.
    A transmitted slot is reused across MR / L-MMSE / ZF / IRC so receiver
    comparisons are fair and cheaper.
    """

    def __init__(self, cfg: SimConfig):
        import sionna.phy

        sionna.phy.config.device = "cpu"
        self.device = _select_device()
        sionna.phy.config.device = "cuda:0" if self.device.type == "cuda" else "cpu"
        sionna.phy.config.seed = cfg.seed
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.cfg = cfg

        self.transmitter, self.pusch_configs = build_transmitter(cfg)
        self.receivers, self.irc_eq = build_receivers(self.transmitter, cfg)
        self._init_channels()

        from sionna.phy.channel import ApplyOFDMChannel

        self.apply_channel = ApplyOFDMChannel()
        self._need_iot = any(v > 0.0 for v in cfg.iot_db)

    def _init_channels(self) -> None:
        from sionna.phy.channel import GenerateOFDMChannel

        cfg = self.cfg
        rg = self.transmitter.resource_grid
        self.serving_model = self._make_model(cfg.num_ue)
        self.gen_serving = GenerateOFDMChannel(
            self.serving_model, rg, normalize_channel=True
        )
        self.int_model = None
        self.gen_int = None
        if self.cfg.num_interferers > 0:
            self.int_model = self._make_model(cfg.num_interferers)
            self.gen_int = GenerateOFDMChannel(
                self.int_model, rg, normalize_channel=True
            )

    def _make_model(self, num_tx: int):
        cfg = self.cfg
        if is_system_level(cfg.channel):
            return build_system_level_channel(cfg)
        return build_cdl_links(cfg, num_tx)

    def _prepare_topology(self, batch_size: int, iot_db: float) -> None:
        cfg = self.cfg
        if not is_system_level(cfg.channel):
            return
        set_system_topology(self.serving_model, cfg, batch_size, cfg.num_ue)
        if iot_db > 0.0 and self.int_model is not None:
            set_system_topology(
                self.int_model, cfg, batch_size, cfg.num_interferers
            )

    def generate_slot(self, batch_size: int, snr_db: float, iot_db: float):
        """Draw one batch of PUSCH slots, channel, IoT and noise."""
        self._prepare_topology(batch_size, iot_db)
        x, b = self.transmitter(batch_size)
        no = snrdb_to_noise_var(snr_db)
        h_s = self.gen_serving(batch_size)
        y_s = self.apply_channel(x, h_s)

        h_i = None
        y_i = None
        if iot_db > 0.0 and self.gen_int is not None:
            h_i = self.gen_int(batch_size)
            x_i = random_ofdm_grid(
                batch_size,
                self.cfg.num_interferers,
                self.cfg.resolved_ue_ant,
                self.cfg.num_ofdm_symbols,
                self.cfg.fft_size,
                dtype=x.real.dtype,
                device=x.device,
                guard_carriers=self.cfg.guard_carriers,
            )
            y_i = self.apply_channel(x_i, h_i)

        y = serving_plus_interference(y_s, y_i, no, iot_db)
        return {
            "x": x,
            "b": b,
            "y": y,
            "h": h_s,
            "h_int": h_i,
            "no": no,
            "iot_db": iot_db,
        }

    def detect(self, slot: dict[str, Any], receiver_name: str):
        rx = self.receivers[receiver_name]
        y, no, h = slot["y"], slot["no"], slot["h"]
        if receiver_name == "irc":
            r_int = irc_interference_covariance(
                slot["h_int"],
                y,
                no,
                iot_db=slot["iot_db"],
                method=self.cfg.iot_cov,
            )
            self.irc_eq.set_covariance(r_int)
        else:
            self.irc_eq.set_covariance(0.0)

        if self.cfg.perfect_csi:
            return rx(y, no, h)
        return rx(y, no)

    def run(
        self,
        snr_db: list[float] | None = None,
        iot_db: list[float] | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Sweep SNR for every IoT level and receiver; return BER and working points."""
        cfg = self.cfg
        snr_db = list(cfg.snr_db if snr_db is None else snr_db)
        iot_db = list(cfg.iot_db if iot_db is None else iot_db)
        campaign: dict[str, Any] = {
            "config": cfg.summary(),
            "device": str(self.device),
            "iot": {},
        }
        t0 = time.time()
        for iot in iot_db:
            campaign["iot"][str(iot)] = self._sweep_snr(snr_db, iot, verbose)
        campaign["duration_s"] = time.time() - t0
        return campaign

    def _sweep_snr(self, snr_db: list[float], iot_db: float, verbose: bool):
        cfg = self.cfg
        results = {name: SweepResult(snr_db=list(snr_db)) for name in cfg.receivers}

        if verbose:
            print(
                f"\n=== {cfg.channel.upper()} | {cfg.modulation.upper()} | "
                f"{cfg.num_ue} UE x rank {cfg.num_layers} | "
                f"IoT={iot_db} dB | CSI="
                f"{'perfect' if cfg.perfect_csi else 'DMRS-LS'} ==="
            )

        for snr in snr_db:
            stats = {name: ErrorStats() for name in cfg.receivers}
            t_snr = time.time()
            for _ in range(cfg.max_mc_iter):
                slot = self.generate_slot(cfg.batch_size, snr, iot_db)
                b_np = slot["b"].detach().cpu().numpy()
                for name in cfg.receivers:
                    b_hat = self.detect(slot, name)
                    stats[name].update(b_np, b_hat.detach().cpu().numpy())
                if all(
                    s.bit_errors >= cfg.num_target_bit_errors for s in stats.values()
                ):
                    break

            if verbose:
                self._print_snr_row(snr, stats, time.time() - t_snr)

            for name, st in stats.items():
                results[name].ber.append(st.ber)
                results[name].bler.append(st.bler)
                results[name].stats.append(st.as_dict())

        out = {}
        for name, res in results.items():
            res.working_point_db = working_point_snr(
                res.snr_db, res.ber, cfg.target_ber
            )
            out[name] = res.as_dict()
            if verbose:
                wp = res.working_point_db
                wp_txt = f"{wp:.2f} dB" if wp is not None else "not reached"
                print(
                    f"  {name.upper():6s} working point "
                    f"(BER={cfg.target_ber:g}): {wp_txt}"
                )
        return out

    @staticmethod
    def _print_snr_row(snr: float, stats: dict[str, ErrorStats], runtime: float) -> None:
        parts = [f"SNR {snr:6.1f} dB"]
        for name, st in stats.items():
            parts.append(f"{name} BER={st.ber:.3e}")
        parts.append(f"[{runtime:.1f} s]")
        print("  " + " | ".join(parts))
