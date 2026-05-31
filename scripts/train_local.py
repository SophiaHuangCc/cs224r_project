"""
Train SAC+HER on all 3 Fetch tasks with checkpoints at 100k, 250k, 500k steps.
Reward functions are loaded from generated_rewards/.

Usage:
    # Run all 9 scenarios (3 tasks × 3 reward types)
    python scripts/train_local.py

    # Filter by reward type or task
    python scripts/train_local.py --reward-type eureka
    python scripts/train_local.py --env FetchReach-v4
    python scripts/train_local.py --env FetchSlide-v4 --reward-type ensemble

    # Skip scenarios where results already exist
    python scripts/train_local.py --skip-existing

Results are written to results/<reward_type>/ and models/<reward_type>/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
os.chdir(_PROJECT_DIR)

from sac import load_reward_fn_from_file, train_with_checkpoints
from ensemble_reward import make_ensemble_reward_fn

# ── Experiment config ────────────────────────────────────────────────────────

TASKS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]
CHECKPOINTS = [100_000, 250_000, 500_000]
TIMESTEPS = 500_000
SEED = 42
EVAL_EPISODES = 100
REWARD_TYPES = ["vanilla", "eureka", "ensemble"]


def reward_config(task: str, reward_type: str) -> dict:
    """Return reward file paths (and optional ensemble aggregation) for a scenario."""
    if reward_type == "vanilla":
        return {"paths": [f"generated_rewards/{task}_vanilla.py"]}
    if reward_type == "eureka":
        return {"paths": [f"generated_rewards/eureka_sac/{task}_best.py"]}
    if reward_type == "ensemble":
        return {
            "paths": [f"generated_rewards/ensemble/{task}/reward_{i}.py" for i in range(1, 4)],
            "aggregation": "min",
        }
    raise ValueError(f"Unknown reward type: {reward_type!r}")


# ── Scenario runner ──────────────────────────────────────────────────────────

def run_scenario(task: str, reward_type: str, skip_existing: bool = False) -> None:
    label = f"{task}_{reward_type}"
    results_dir = f"results/{reward_type}"
    models_dir = f"models/{reward_type}"
    summary_path = os.path.join(results_dir, f"{label}_summary.json")

    if skip_existing and os.path.exists(summary_path):
        print(f"  Skipping {label} (already done)")
        return

    cfg = reward_config(task, reward_type)
    missing = [p for p in cfg["paths"] if not os.path.exists(p)]
    if missing:
        print(f"  Skipping {label}: reward file(s) not found: {missing}")
        return

    if len(cfg["paths"]) == 1:
        reward_fn = load_reward_fn_from_file(cfg["paths"][0])
    else:
        fns = [load_reward_fn_from_file(p) for p in cfg["paths"]]
        reward_fn = make_ensemble_reward_fn(fns, aggregation=cfg.get("aggregation", "min"))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Checkpoints: {[f'{c//1000}k' for c in CHECKPOINTS]}")
    print(f"{'='*60}")

    train_with_checkpoints(
        env_id=task,
        reward_fn=reward_fn,
        timesteps=TIMESTEPS,
        checkpoints=CHECKPOINTS,
        seed=SEED,
        eval_episodes=EVAL_EPISODES,
        save_dir=models_dir,
        results_dir=results_dir,
        label=label,
        tensorboard_log=f"logs/{reward_type}/{label}",
    )


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SAC+HER on Fetch tasks.")
    parser.add_argument("--env", choices=TASKS + ["all"], default="all",
                        help="Which task to run (default: all)")
    parser.add_argument("--reward-type", choices=REWARD_TYPES + ["all"], default="all",
                        help="Which reward type to run (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip scenarios where a summary JSON already exists")
    args = parser.parse_args()

    tasks = TASKS if args.env == "all" else [args.env]
    reward_types = REWARD_TYPES if args.reward_type == "all" else [args.reward_type]
    total = len(tasks) * len(reward_types)

    print(f"\nRunning {total} scenario(s)")
    print(f"  Tasks:        {tasks}")
    print(f"  Reward types: {reward_types}")
    print(f"  Checkpoints:  {[f'{c//1000}k' for c in CHECKPOINTS]}\n")

    start = time.time()
    for task in tasks:
        for reward_type in reward_types:
            run_scenario(task, reward_type, skip_existing=args.skip_existing)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Done. Total time: {elapsed / 3600:.1f}h")
    print(f"Results: results/  |  Models: models/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
