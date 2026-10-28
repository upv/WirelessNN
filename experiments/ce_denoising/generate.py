"""Generate independent, replayable CE examples; run from the repository root."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
import sionna.phy

from nr_ul_sim.parameters import CHANNELS, MODULATIONS, SimConfig, snrdb_to_noise_var
from nr_ul_sim.pusch import strip_guard_subcarriers
from nr_ul_sim.random_campaign import RandomChannelSimulator
from nr_ul_sim.random_channels import build_frozen_system_level, build_random_cdl, sample_cdl_link
from nr_ul_sim.losses import constellation
from nr_ul_sim.simulator import _select_device

SCHEMA = 1
SPEED_RANGE = (3.0, 30.0)


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def plan(seed=20261014, counts=(540, 180, 180)):
    """Randomized stratification over model x IoT x 10-dB SNR band, per split."""
    if any(n <= 0 or n % 45 for n in counts):
        raise ValueError("Each split size must be a positive multiple of 45 (5 x 3 x 3)")
    rows = []
    for split_id, (split, count) in enumerate(zip(("train", "validation", "test"), counts)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, split_id, 718]))
        cells = [(ch, iot, low) for _ in range(count // 45)
                 for ch in CHANNELS for iot in (0, 10, 20) for low in (-20, -10, 0)]
        rng.shuffle(cells)
        for i, (ch, iot, low) in enumerate(cells):
            record_seed = int(np.random.SeedSequence([seed, split_id, i, 193]).generate_state(1)[0] % (2**31 - 1))
            r = np.random.default_rng(record_seed)
            rows.append(dict(id=f"{split}_{i:06d}", split=split, seed=record_seed,
                             channel=ch, iot_db=iot, snr_db=float(r.uniform(low, low + 10)),
                             snr_bin_low_db=low, modulation=str(r.choice(MODULATIONS)),
                             num_interferers_configured=int(r.integers(2, 7))))
    return rows


class CaptureCIR:
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        self.a, self.tau = self.model(*args, **kwargs)
        return self.a, self.tau


def snapshot(model, prefix, result):
    """Resolved numeric model, topology, LSP and ray state; never pickle objects."""
    if hasattr(model, "links"):
        for i, link in enumerate(model.links):
            snapshot(link, f"{prefix}_link{i}", result)
    objects = [(prefix, model)]
    if hasattr(model, "model"):
        objects += [(prefix + "_scenario", model.model._scenario),
                    (prefix + "_lsp", model.lsp), (prefix + "_rays", model.rays)]
    for tag, obj in objects:
        values = {**vars(obj), **getattr(obj, "_buffers", {})}
        for name, value in values.items():
            if torch.is_tensor(value) or isinstance(value, (np.ndarray, int, float, bool)):
                a = array(value)
                if a.dtype.kind in "biufc":
                    result[f"{tag}_{name.lstrip('_')}"] = a


def make_links(cfg, n, rng):
    if cfg.channel.startswith("cdl"):
        links = [sample_cdl_link(rng, cfg.channel) for _ in range(n)]
        for link in links:
            link["speed_kmh"] = float(rng.uniform(*SPEED_RANGE))
        model, info, data = build_random_cdl(cfg, links)
        arrays = {f"link{i}_{key}": np.asarray(val)
                  for i, item in enumerate(data) for key, val in item.items()}
    else:
        model, info, arrays = build_frozen_system_level(
            cfg, n, rng, speed_kmh_range=SPEED_RANGE, indoor_speed_kmh_range=SPEED_RANGE)
    snapshot(model, "resolved", arrays)
    return model, info, arrays


def pilot_blocks(grid, pilot_grid, dmrs_symbols):
    """CDM-aware LS: average the two nonzero pilots of each four-SC block.

    grid: [...,14,F], pilots: [14,F], returns [...,2,F/4]. Rank-1 port 0 only.
    """
    obs = grid[..., dmrs_symbols, 0::2] / pilot_grid[dmrs_symbols, 0::2]
    return obs.reshape(*obs.shape[:-1], -1, 2).mean(-1)


def validate_record(data, meta):
    """Physical identities are checked before an example becomes visible."""
    for name, value in data.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"non-finite {name}")
    np.testing.assert_allclose(data["y"], data["y_target"] + data["noise"] + data["interference"], rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(data["y_target"], data["h_true"] * data["x_tx"][None], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(data["interference"], data["interference_per_link"].sum(0), rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(data["interference_per_link"], data["h_interferers"] * data["x_interferers_scaled"][:, None], rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(data["h_ls"], data["h_ls_clean"] + data["h_ls_noise"] + data["h_ls_interference"], rtol=2e-5, atol=3e-5)
    np.testing.assert_allclose(data["ls_pilots"], data["ls_pilots_clean"] + data["ls_pilots_noise"] + data["ls_pilots_interference"], rtol=2e-5, atol=3e-5)
    for link in meta["target_links"] + meta["interferer_links"]:
        assert 3 <= link["speed_kmh"] <= 30
    assert -20 <= meta["snr_db"] <= 10
    assert meta["iot_db"] in (0, 10, 20)
    if meta["iot_db"] == 0:
        assert not np.any(data["interference"])
    assert data["h_true"].shape == data["h_ls"].shape


@torch.no_grad()
def generate_record(row, *, num_prb=16, fft_size=256):
    started = time.perf_counter()
    seed = row["seed"]
    sionna.phy.config.device = "cuda:0" if _select_device().type == "cuda" else "cpu"
    sionna.phy.config.seed = seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    n_int = row["num_interferers_configured"]
    cfg = SimConfig(channel=row["channel"], modulation=row["modulation"],
                    num_ue=1, num_layers=1, num_rx_ant=4, num_prb=num_prb, fft_size=fft_size,
                    receivers=("lmmse",), channel_estimators=("ls_lin",),
                    num_interferers=n_int, seed=seed, batch_size=1, max_mc_iter=1,
                    snr_db=[row["snr_db"]], iot_db=[row["iot_db"]])
    target, target_info, target_params = make_links(cfg, 1, rng)
    interferers, int_info, int_params = make_links(cfg, n_int, rng)
    cfg.speed_kmh = target_info[0]["speed_kmh"]
    cfg.delay_spread_ns = target_info[0].get("ds_ns", target_info[0].get("lsp_ds_ns"))
    ctarget, cint = CaptureCIR(target), CaptureCIR(interferers)
    sim = RandomChannelSimulator(cfg, ctarget, cint)
    h, hi = sim.gen_serving(1), sim.gen_int(1)
    x, bits = sim.transmit(1)
    xi = sim.interferer_grid(1, x)
    n0 = snrdb_to_noise_var(row["snr_db"])
    inr = 0.0 if row["iot_db"] == 0 else 10.0 ** (row["iot_db"] / 10)
    scale = (n0 * inr / n_int) ** 0.5
    ys = sim.apply_channel(x, h)
    # Per-link received contributions, identical sum to ApplyOFDMChannel.
    yi_links = hi[0, 0, :, :, 0].permute(1, 0, 2, 3) * xi[0, :, 0, None] * scale
    yi = yi_links.sum(0)[None, None]
    noise = torch.complex(torch.randn_like(ys.real), torch.randn_like(ys.real)) * (n0 / 2) ** 0.5
    y = ys + yi + noise
    est = sim.rx.estimators["ls_lin"]
    no_t = torch.as_tensor(n0, device=y.device, dtype=y.real.dtype)
    def ce(z):
        return est(z, no_t)[0][0, 0, :, 0, 0]
    def used(z):
        return strip_guard_subcarriers(z, cfg.guard_carriers)
    p = sim.rx.pilot_grid[0, 0].to(sim.device)
    mask = sim.transmitter.pilot_pattern.mask[0, 0].bool()
    x_used = used(x)[0, 0, 0]
    y_used = used(y)[0, 0]
    data = {
        "h_true": sim.true_channel(h)[0, 0, :, 0, 0],
        "h_ls": ce(y), "h_ls_clean": ce(ys), "h_ls_noise": ce(noise), "h_ls_interference": ce(yi),
        "y": y_used, "y_target": used(ys)[0, 0], "noise": used(noise)[0, 0],
        "interference": used(yi)[0, 0], "interference_per_link": used(yi_links),
        "h_interferers": used(hi)[0, 0, :, :, 0].permute(1, 0, 2, 3),
        "x_interferers_scaled": used(xi)[0, :, 0] * scale,
        "x_interferers_unit": used(xi)[0, :, 0],
        "x_tx": x_used, "tx_tb_bits": bits[0, 0].to(torch.uint8),
        "pilot_grid": p, "pilot_reserved_mask": mask, "data_mask": ~mask,
        "dmrs_symbols": np.asarray(sim.rx.dmrs_syms, dtype=np.int64),
        "pilot_block_centres_sc": np.arange(num_prb * 3) * 4 + 1,
        "cir_target_a": ctarget.a, "cir_target_tau_s": ctarget.tau,
        "cir_interferer_a": cint.a, "cir_interferer_tau_s": cint.tau,
    }
    for suffix, grid in (("", y), ("_clean", ys), ("_noise", noise), ("_interference", yi)):
        data["ls_pilots" + suffix] = pilot_blocks(used(grid)[0, 0], p, sim.rx.dmrs_syms)
    # Label actual transmitted QAM symbols (coded and scrambled), not TB bits.
    points, labels = constellation(cfg.modulation)
    points, labels = points.to(sim.device), labels.to(sim.device)
    tx_data = x_used[~mask]
    closest = (tx_data[:, None] - points[None]).abs().argmin(-1)
    if (tx_data - points[closest]).abs().max() > 1e-5:
        raise ValueError("QAM constellation mapping differs from BCE labels")
    data["tx_data_symbols"] = tx_data
    data["tx_coded_bits"] = labels[closest].to(torch.uint8)
    data = {key: array(value) for key, value in data.items()}
    params = {**{f"target_{k}": array(v) for k, v in target_params.items()},
              **{f"interferer_{k}": array(v) for k, v in int_params.items()}}
    powers = {key: float(np.mean(np.abs(data[key]) ** 2)) for key in ("h_true", "y_target", "noise", "interference")}
    meta = {**row, "schema_version": SCHEMA, "num_interferers_active": n_int if inr else 0,
            "target_links": target_info, "interferer_links": int_info,
            "config": asdict(cfg), "resolved_config": cfg.summary(), "device": str(sim.device),
            "noise_variance": n0, "interference_variance_nominal": n0 * inr,
            "interferer_amplitude_scale": scale, "measured_power": powers,
            "measured_snr_db": float(10 * np.log10(powers["y_target"] / powers["noise"])),
            "measured_inr_db": float(10 * np.log10(powers["interference"] / powers["noise"])) if inr else None,
            "ofdm_symbol_duration_s": float(sim.transmitter.resource_grid.ofdm_symbol_duration),
            "array_shapes": {k: list(v.shape) for k, v in data.items()},
            "array_dtypes": {k: str(v.dtype) for k, v in data.items()},
            "generation_seconds": time.perf_counter() - started}
    validate_record(data, meta)
    return data, params, meta


def provenance():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "nr_ul_sim").glob("*.py")) + [Path(__file__).resolve()]
    return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "source_sha256": {str(p.relative_to(root)): digest(p) for p in paths},
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "sionna", "numpy", "scipy")},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def run(outdir, *, seed=20261014, counts=(540, 180, 180), num_prb=16, fft_size=256, limit=None, resume=False):
    outdir = Path(outdir)
    recipe = dict(schema_version=SCHEMA, seed=seed, counts=list(counts), num_prb=num_prb,
                  fft_size=fft_size, speed_kmh_range=list(SPEED_RANGE), iot_zero="no_interference")
    rows = plan(seed, counts)
    if outdir.exists() and not resume:
        raise FileExistsError(f"{outdir} exists; use --resume with the same recipe")
    outdir.mkdir(parents=True, exist_ok=True)
    # O_EXCL is portable. A stale lock after a crash requires explicit inspection/removal.
    lock = outdir / ".writer.lock"
    with lock.open("x") as f:
        f.write(str(__import__("os").getpid()))
    try:
        config_path = outdir / "config.json"
        current = provenance()
        if config_path.exists():
            old = json.loads(config_path.read_text())
            if old["recipe"] != recipe or old["provenance"]["source_sha256"] != current["source_sha256"] or old["provenance"]["versions"] != current["versions"]:
                raise ValueError("Resume requires identical recipe, sources and library versions")
        else:
            write_json(config_path, dict(recipe=recipe, provenance=current))
        completed = []
        for row in rows:
            dest = outdir / "records" / row["id"]
            if dest.exists():
                m = json.loads((dest / "meta.json").read_text())
                if any(m[k] != row[k] for k in row):
                    raise ValueError(f"Record plan mismatch: {dest}")
                for name, expected in m["sha256"].items():
                    if digest(dest / name) != expected:
                        raise ValueError(f"Checksum mismatch: {dest / name}")
                completed.append(m)
                continue
            if limit is not None and len(completed) >= limit:
                break
            if shutil.disk_usage(outdir).free < 2 * 1024**3:
                raise RuntimeError("Less than 2 GiB free; stopping before next record")
            data, params, meta = generate_record(row, num_prb=num_prb, fft_size=fft_size)
            pending = dest.with_name(dest.name + ".pending")
            pending.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(pending / "sample.npz", **data)
            np.savez_compressed(pending / "parameters.npz", **params)
            meta["sha256"] = {name: digest(pending / name) for name in ("sample.npz", "parameters.npz")}
            write_json(pending / "meta.json", meta)
            pending.rename(dest)
            completed.append(meta)
            if len(completed) % 15 == 0 or len(completed) == 1:
                print(f"{len(completed)}/{len(rows)} {row['id']} {row['channel']} SNR={row['snr_db']:.1f} IoT={row['iot_db']}", flush=True)
        # Rebuild index from committed records: never put one channel in two splits.
        with (outdir / "index.jsonl").open("w") as f:
            for m in completed:
                f.write(json.dumps({k: m[k] for k in rows[0]}) + "\n")
        summary = {"records": len(completed), "splits": dict(Counter(m["split"] for m in completed)),
                   "channels": dict(Counter(m["channel"] for m in completed)),
                   "generation_seconds": sum(m["generation_seconds"] for m in completed),
                   "bytes": sum(p.stat().st_size for p in outdir.rglob("*") if p.is_file())}
        write_json(outdir / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    finally:
        lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--counts", type=int, nargs=3, default=(540, 180, 180), metavar=("TRAIN", "VALIDATION", "TEST"))
    p.add_argument("--num-prb", type=int, default=16)
    p.add_argument("--fft-size", type=int, default=256)
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    args = vars(p.parse_args())
    run(**args)


if __name__ == "__main__":
    main()
