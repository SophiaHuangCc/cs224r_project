"""
Generate N=3 independent Eureka-quality reward functions for the ensemble mitigation.
Rewards are saved to generated_rewards/ensemble/<task>/reward_<i>.py.

Run this before training:
    python scripts/run_ensemble_mitigation.py
    python scripts/train_local.py --reward-type ensemble

Usage:
    python scripts/run_ensemble_mitigation.py
    python scripts/run_ensemble_mitigation.py --envs FetchSlide-v4
    python scripts/run_ensemble_mitigation.py --skip-existing
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from ensemble_reward import generate_eureka_ensemble_rewards

DEFAULT_ENVS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]
DEFAULT_N = 3
SEED = 42


def generate_ensemble_rewards(
    envs: list[str] = DEFAULT_ENVS,
    n: int = DEFAULT_N,
    seed: int = SEED,
    skip_existing: bool = False,
):
    """Generate N independent Eureka reward functions per task, saved to generated_rewards/ensemble/."""
    print(f"\nGenerating ensemble rewards")
    print(f"  Tasks: {envs}")
    print(f"  N per task: {n}\n")

    for env_id in envs:
        reward_dir = f"generated_rewards/ensemble/{env_id}"

        if skip_existing and os.path.exists(reward_dir):
            print(f"  {env_id}: skipping (already exists at {reward_dir}/)")
            continue

        print(f"\n{'='*60}")
        print(f"  {env_id}: running {n} independent Eureka loops...")
        print(f"{'='*60}")

        generate_eureka_ensemble_rewards(
            env_id=env_id,
            n=n,
            eureka_iters=3,
            timesteps_per_iter=50_000,
            eval_episodes=50,
            save_dir=reward_dir,
            seed=seed,
        )
        print(f"  Saved {n} rewards to {reward_dir}/")

    print(f"\nDone. Run `python scripts/train_local.py --reward-type ensemble` to train.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ensemble reward functions.")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tasks where generated_rewards/ensemble/<task>/ already exists")
    args = parser.parse_args()

    generate_ensemble_rewards(
        envs=args.envs,
        n=args.n,
        seed=args.seed,
        skip_existing=args.skip_existing,
    )
