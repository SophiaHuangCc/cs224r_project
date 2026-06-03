"""Write eval JSON metrics into tfevents so they appear in TensorBoard.

Run from project root:
    python scripts/export_eval_to_tb.py
Then:
    tensorboard --logdir logs
Tag prefix `eval/` keeps them separate from `train/` and `rollout/` curves.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

EVAL_ROOT = Path("results/eval")
LOG_ROOT = Path("logs")
SKIP_KEYS = {"checkpoint", "task", "reward_type", "label"}
PATTERN = re.compile(r"^(?P<task>.+)_(?P<reward>[^_]+)_(?P<ckpt>\d+)k\.json$")


def main() -> None:
    runs: dict[Path, list[tuple[int, dict]]] = {}
    for json_path in sorted(EVAL_ROOT.rglob("*.json")):
        m = PATTERN.match(json_path.name)
        if not m:
            continue
        with json_path.open() as f:
            data = json.load(f)
        step = int(m["ckpt"]) * 1000
        # Mirror training logdir layout: logs/<reward>/<task>_<reward>/eval
        reward = json_path.parent.name
        label = f"{m['task']}_{reward}"
        run_dir = LOG_ROOT / reward / label / "eval"
        runs.setdefault(run_dir, []).append((step, data))

    for run_dir, entries in runs.items():
        run_dir.mkdir(parents=True, exist_ok=True)
        for old in run_dir.glob("events.out.tfevents.*"):
            old.unlink()
        writer = SummaryWriter(log_dir=str(run_dir))
        for step, data in sorted(entries, key=lambda x: x[0]):
            for key, val in data.items():
                if key in SKIP_KEYS or not isinstance(val, (int, float)):
                    continue
                writer.add_scalar(f"eval/{key}", float(val), step)
        writer.close()
        print(f"wrote {len(entries)} steps -> {run_dir}")


if __name__ == "__main__":
    main()
