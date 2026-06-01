"""
Train SAC+HER on all 3 Fetch tasks with checkpoints at 100k, 250k, 500k steps.
Reward functions are loaded from generated_rewards/.

Reward types:
  vanilla  — single LLM reward
  eureka   — best reward from the iterative Eureka search
  ensemble — N=3 independent Eureka rewards, aggregated with min
  kl       — Eureka reward + KL penalty against a sparse-reward reference policy.
             Sweeps β over [0.01, 0.1, 1.0] → 3 models per task.
             Requires: python scripts/train_reference.py (one-time setup).

Usage:
    # Run all scenarios
    python scripts/train_local.py

    # Filter by reward type or task
    python scripts/train_local.py --reward-type eureka
    python scripts/train_local.py --env FetchReach-v4
    python scripts/train_local.py --env FetchSlide-v4 --reward-type kl

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

import gymnasium as gym  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from sac import (  # noqa: E402
    load_reward_fn_from_file,
    train_with_checkpoints,
    train_with_kl_checkpoints,
)
from ensemble_reward import make_ensemble_reward_fn  # noqa: E402

# ── Experiment config ────────────────────────────────────────────────────────

TASKS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]
CHECKPOINTS = [100_000, 250_000, 500_000]
TIMESTEPS = 500_000
SEED = 42
EVAL_EPISODES = 100
REWARD_TYPES = ["vanilla", "eureka", "ensemble", "kl"]
KL_BETAS = [0.01, 0.1, 1.0]


def reward_config(task: str, reward_type: str) -> dict:
    """Return reward file paths (and optional ensemble aggregation) for a scenario."""
    if reward_type == "vanilla":
        return {"paths": [f"generated_rewards/{task}_vanilla.py"]}
    if reward_type in ("eureka", "kl"):
        # KL uses the Eureka reward as the proxy; the KL constraint is the mitigation.
        return {"paths": [f"generated_rewards/eureka_sac/{task}_best.py"]}
    if reward_type == "ensemble":
        return {
            "paths": [f"generated_rewards/ensemble/{task}/reward_{i}.py" for i in range(1, 4)],
            "aggregation": "min",
        }
    raise ValueError(f"Unknown reward type: {reward_type!r}")


def _load_reward_fn(cfg: dict):
    paths = cfg["paths"]
    if len(paths) == 1:
        return load_reward_fn_from_file(paths[0])
    fns = [load_reward_fn_from_file(p) for p in paths]
    return make_ensemble_reward_fn(fns, aggregation=cfg.get("aggregation", "min"))


# ── Scenario runners ─────────────────────────────────────────────────────────

def _run_standard(task: str, reward_type: str, skip_existing: bool) -> None:
    label = f"{task}_{reward_type}"
    results_dir = f"results/train/{reward_type}"
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

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Checkpoints: {[f'{c//1000}k' for c in CHECKPOINTS]}")
    print(f"{'='*60}")

    train_with_checkpoints(
        env_id=task,
        reward_fn=_load_reward_fn(cfg),
        timesteps=TIMESTEPS,
        checkpoints=CHECKPOINTS,
        seed=SEED,
        eval_episodes=EVAL_EPISODES,
        save_dir=models_dir,
        results_dir=results_dir,
        label=label,
        tensorboard_log=f"logs/{reward_type}/{label}",
    )


def _run_kl(task: str, skip_existing: bool) -> None:
    """Sweep KL_BETAS — one training run per β. Requires a reference policy."""
    ref_path = f"models/reference/{task}.zip"
    if not os.path.exists(ref_path):
        print(f"  Skipping {task}_kl: reference policy missing at {ref_path}")
        print(f"  Run `python scripts/train_reference.py --env {task}` first.")
        return

    cfg = reward_config(task, "kl")
    missing = [p for p in cfg["paths"] if not os.path.exists(p)]
    if missing:
        print(f"  Skipping {task}_kl: reward file(s) not found: {missing}")
        return

    for beta in KL_BETAS:
        label = f"{task}_kl_b{beta}"
        results_dir = "results/kl"
        summary_path = os.path.join(results_dir, f"{label}_summary.json")

        if skip_existing and os.path.exists(summary_path):
            print(f"  Skipping {label} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  KL β: {beta}")
        print(f"  Reference: {ref_path}")
        print(f"  Checkpoints: {[f'{c//1000}k' for c in CHECKPOINTS]}")
        print(f"{'='*60}")

        # Load a fresh reference per β to avoid any cross-run device/state issues.
        # Must pass env= because the reference was trained with HerReplayBuffer,
        # and SB3 asserts env is not None when loading HER-based models.
        ref_env = gym.make(task)
        reference_model = SAC.load(ref_path, env=ref_env)
        ref_env.close()

        train_with_kl_checkpoints(
            env_id=task,
            reward_fn=_load_reward_fn(cfg),
            reference_model=reference_model,
            kl_beta=beta,
            timesteps=TIMESTEPS,
            checkpoints=CHECKPOINTS,
            seed=SEED,
            eval_episodes=EVAL_EPISODES,
            save_dir="models/kl",
            results_dir=results_dir,
            label=label,
            tensorboard_log=f"logs/kl/{label}",
        )


def run_scenario(task: str, reward_type: str, skip_existing: bool) -> None:
    if reward_type == "kl":
        _run_kl(task, skip_existing)
    else:
        _run_standard(task, reward_type, skip_existing)


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

    # KL runs N_TASKS × len(KL_BETAS) jobs; other reward types run N_TASKS jobs.
    total = sum(
        len(KL_BETAS) if rt == "kl" else 1
        for rt in reward_types
    ) * len(tasks)

    print(f"\nRunning {total} training run(s)")
    print(f"  Tasks:        {tasks}")
    print(f"  Reward types: {reward_types}")
    print(f"  Checkpoints:  {[f'{c//1000}k' for c in CHECKPOINTS]}")
    if "kl" in reward_types:
        print(f"  KL β sweep:   {KL_BETAS}")
    print()

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
