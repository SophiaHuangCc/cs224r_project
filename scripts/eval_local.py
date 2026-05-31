"""
Evaluate saved SAC+HER models: success rate, hacking rate, and safety metrics.
Mirrors train_local.py — models are loaded from models/<reward_type>/<label>_<ckpt>k.zip.

Hacking rate: fraction of episodes where the LLM proxy reward is high but the
task actually failed (reward_fn "hacks" the proxy without solving the task).

Safety metrics: action violation rate, jerk, object speed/acceleration violations,
drop/slam rates — all computed by SafetyMetricWrapper during the eval rollout.

Usage:
    # Evaluate all scenarios at all checkpoints
    python scripts/eval_local.py

    # Filter
    python scripts/eval_local.py --reward-type eureka
    python scripts/eval_local.py --env FetchReach-v4 --checkpoint 500000

    # Skip already-evaluated scenarios
    python scripts/eval_local.py --skip-existing

Results are written to results/eval/<reward_type>/<label>_<ckpt>k.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
os.chdir(_PROJECT_DIR)

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
from stable_baselines3 import SAC

from sac import load_reward_fn_from_file
from ensemble_reward import make_ensemble_reward_fn
from safety import SafetyMetricWrapper, evaluate_with_safety

# ── Config (must match train_local.py) ──────────────────────────────────────

TASKS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]
CHECKPOINTS = [100_000, 250_000, 500_000]
REWARD_TYPES = ["vanilla", "eureka", "ensemble"]
EVAL_EPISODES = 100


def reward_config(task: str, reward_type: str) -> dict:
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


def model_path(task: str, reward_type: str, checkpoint: int) -> str:
    label = f"{task}_{reward_type}"
    return f"models/{reward_type}/{label}_{checkpoint // 1000}k.zip"


def load_reward(cfg: dict):
    paths = cfg["paths"]
    if len(paths) == 1:
        return load_reward_fn_from_file(paths[0])
    fns = [load_reward_fn_from_file(p) for p in paths]
    return make_ensemble_reward_fn(fns, aggregation=cfg.get("aggregation", "min"))


# ── Evaluation ───────────────────────────────────────────────────────────────

def eval_scenario(
    task: str,
    reward_type: str,
    checkpoint: int,
    n_episodes: int,
    skip_existing: bool,
) -> dict | None:
    label = f"{task}_{reward_type}"
    results_dir = f"results/eval/{reward_type}"
    result_path = os.path.join(results_dir, f"{label}_{checkpoint // 1000}k.json")

    if skip_existing and os.path.exists(result_path):
        print(f"  Skipping {label} @ {checkpoint // 1000}k (already done)")
        return None

    mpath = model_path(task, reward_type, checkpoint)
    if not os.path.exists(mpath):
        print(f"  Skipping {label} @ {checkpoint // 1000}k: model not found at {mpath}")
        return None

    cfg = reward_config(task, reward_type)
    missing = [p for p in cfg["paths"] if not os.path.exists(p)]
    if missing:
        print(f"  Skipping {label} @ {checkpoint // 1000}k: reward file(s) missing: {missing}")
        return None

    print(f"  Evaluating {label} @ {checkpoint // 1000}k ({n_episodes} episodes)...")
    reward_fn = load_reward(cfg)
    eval_env = SafetyMetricWrapper(gym.make(task))
    model = SAC.load(mpath, env=eval_env)

    metrics = evaluate_with_safety(model, eval_env, n_episodes=n_episodes, reward_fn=reward_fn)
    eval_env.close()

    metrics.update({"checkpoint": checkpoint, "task": task, "reward_type": reward_type})

    os.makedirs(results_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ── Reporting ────────────────────────────────────────────────────────────────

_TABLE_HEADER = (
    f"  {'Ckpt':<6} {'Success':>8} {'Hacking':>8} "
    f"{'ActViol':>8} {'JerkViol':>9} {'SpdViol':>8} {'AccViol':>8}"
)
_TABLE_SEP = f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*8}"


def print_scenario_table(task: str, reward_type: str, results: dict[int, dict]) -> None:
    print(f"\n  {task}  [{reward_type}]")
    print(_TABLE_HEADER)
    print(_TABLE_SEP)
    for ckpt in sorted(results):
        m = results[ckpt]
        print(
            f"  {ckpt // 1000}k{'':<3} "
            f"{m['success_rate']:>7.1%} "
            f"{m.get('hacking_rate', 0):>7.1%} "
            f"{m.get('action_violation_rate', 0):>7.1%} "
            f"{m.get('jerk_violation_rate', 0):>8.1%} "
            f"{m.get('object_speed_violation_rate', 0):>7.1%} "
            f"{m.get('object_accel_violation_rate', 0):>7.1%}"
        )


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC+HER models.")
    parser.add_argument("--env", choices=TASKS + ["all"], default="all")
    parser.add_argument("--reward-type", choices=REWARD_TYPES + ["all"], default="all")
    parser.add_argument(
        "--checkpoint", type=int, choices=CHECKPOINTS, default=None,
        help="Single checkpoint to evaluate (default: all)",
    )
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    tasks = TASKS if args.env == "all" else [args.env]
    reward_types = REWARD_TYPES if args.reward_type == "all" else [args.reward_type]
    checkpoints = CHECKPOINTS if args.checkpoint is None else [args.checkpoint]
    total = len(tasks) * len(reward_types) * len(checkpoints)

    print(f"\nEvaluating {total} scenario-checkpoint(s)")
    print(f"  Tasks:        {tasks}")
    print(f"  Reward types: {reward_types}")
    print(f"  Checkpoints:  {[f'{c // 1000}k' for c in checkpoints]}")
    print(f"  Episodes:     {args.episodes}\n")

    start = time.time()

    for task in tasks:
        for reward_type in reward_types:
            ckpt_results: dict[int, dict] = {}
            for ckpt in checkpoints:
                m = eval_scenario(task, reward_type, ckpt, args.episodes, args.skip_existing)
                if m is not None:
                    ckpt_results[ckpt] = m
            if ckpt_results:
                print_scenario_table(task, reward_type, ckpt_results)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Done. Total time: {elapsed / 60:.1f}min")
    print(f"Results: results/eval/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
