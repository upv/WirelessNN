#!/usr/bin/env python3
"""Merge dataset shards into one run and pack it.

    .venv/bin/python merge_shards.py dataset/irc_ce20k dataset/irc_ce20k/shards/*

Channel folders are named by the global scenario index, so shards never collide;
they are symlinked into ``<dst>/channels``. Duplicate (scenario, modulation) records
are dropped.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from nr_ul_sim.dataset import pack_dataset


def merge(dst: Path, shards: list[Path]) -> Path:
    (dst / "channels").mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, str]] = set()
    lines: list[str] = []
    for shard in sorted(shards):
        meta = shard / "meta.jsonl"
        if not meta.exists():
            continue
        for line in meta.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (int(rec["scenario_id"]), rec["modulation"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
        for folder in (shard / "channels").iterdir():
            link = dst / "channels" / folder.name
            if not link.exists():
                link.symlink_to(folder.resolve(), target_is_directory=True)
        config = shard / "dataset_config.json"
        if config.exists() and not (dst / "dataset_config.json").exists():
            shutil.copy(config, dst / "dataset_config.json")
    lines.sort(key=lambda l: (json.loads(l)["scenario_id"], json.loads(l)["modulation_code"]))
    (dst / "meta.jsonl").write_text("\n".join(lines) + "\n")
    return pack_dataset(dst)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    merge(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
