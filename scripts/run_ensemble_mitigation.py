"""
Run Mitigation 1: Reward Ensemble on Fetch envs.

Steps:
1. Generate N=3 independent Eureka-quality reward functions (SAC+HER search)
2. Train SAC+HER with ensemble (min aggregation), checkpoints at 100k/250k/500k
3. Evaluate: success_rate, hacking_rate, safety metrics

Usage:
    cd scripts/
    python run_ensemble_mitigation.py [--envs FetchPickAndPlace-v4] [--n 3] [--aggregation min]

Or train only (if rewards already generated):
    python sac/train_from_reward.py \
        --env FetchPickAndPlace-v4 \
        --reward generated_rewards/ensemble/FetchPickAndPlace-v4/reward_1.py \
                 generated_rewards/ensemble/FetchPickAndPlace-v4/reward_2.py \
                 generated_rewards/ensemble/FetchPickAndPlace-v4/reward_3.py \
        --aggregation min \
        --checkpoints 100000 250000 500000
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from ensemble_reward import generate_eureka_ensemble_rewards, make_ensemble_reward_fn
from llm_reward_vanilla import compile_reward_fn
from sac.train_from_reward import train_with_checkpoints

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
DEFAULT_ENVS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]
DEFAULT_N = 3
DEFAULT_AGGREGATION = "min"
DEFAULT_TIMESTEPS = 500_000
DEFAULT_CHECKPOINTS = [100_000, 250_000, 500_000]
EVAL_EPISODES = 100
SEED = 42


def run_ensemble_mitigation(
    envs: list[str] = DEFAULT_ENVS,
    n: int = DEFAULT_N,
    aggregation: str = DEFAULT_AGGREGATION,
    timesteps: int = DEFAULT_TIMESTEPS,
    checkpoints: list[int] = DEFAULT_CHECKPOINTS,
    eval_episodes: int = EVAL_EPISODES,
    seed: int = SEED,
    skip_generation: bool = False,
):
    """Full pipeline: generate ensemble rewards → train → evaluate at checkpoints."""

    print(f"\n{'='*70}")
    print(f"MITIGATION 1: REWARD ENSEMBLE")
    print(f"  Envs: {envs}")
    print(f"  N: {n} reward functions (independent Eureka loops)")
    print(f"  Aggregation: {aggregation}")
    print(f"  Checkpoints: {[f'{c//1000}k' for c in checkpoints]}")
    print(f"{'='*70}\n")

    for env_id in envs:
        print(f"\n{'='*70}")
        print(f"  ENV: {env_id}")
        print(f"{'='*70}")

        reward_dir = f"generated_rewards/ensemble/{env_id}"

        # Step 1: Generate N Eureka-quality rewards
        if skip_generation and os.path.exists(reward_dir):
            print("  Step 1: Loading pre-generated rewards...")
            reward_codes = []
            for i in range(1, n + 1):
                with open(os.path.join(reward_dir, f"reward_{i}.py")) as f:
                    reward_codes.append(f.read())
            print(f"  Loaded {len(reward_codes)} rewards from {reward_dir}/")
        else:
            print("  Step 1: Running N independent Eureka loops (SAC+HER)...")
            reward_codes = generate_eureka_ensemble_rewards(
                env_id=env_id,
                n=n,
                eureka_iters=3,
                timesteps_per_iter=50_000,
                eval_episodes=50,
                save_dir=reward_dir,
                seed=seed,
            )

        # Compile + create ensemble reward_fn
        reward_fns = [compile_reward_fn(code) for code in reward_codes]
        ensemble_fn = make_ensemble_reward_fn(reward_fns, aggregation=aggregation)

        # Step 2: Train with checkpoints
        print(f"\n  Step 2: Training SAC+HER with ensemble ({aggregation}, N={n})...")
        label = f"{env_id}_ensemble_n{n}_{aggregation}"

        train_with_checkpoints(
            env_id=env_id,
            reward_fn=ensemble_fn,
            timesteps=timesteps,
            checkpoints=checkpoints,
            seed=seed,
            eval_episodes=eval_episodes,
            save_dir=f"models/ensemble",
            results_dir=f"results/ensemble",
            label=label,
            tensorboard_log=f"logs/ensemble_{env_id}_{aggregation}_n{n}",
        )

    print(f"\n\n✅ All ensemble mitigation runs complete!")
    print(f"Results: results/ensemble/")
    print(f"Models: models/ensemble/")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Mitigation 1: Reward Ensemble")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--aggregation", choices=["min", "mean", "trimmed_mean"], default=DEFAULT_AGGREGATION)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Use pre-generated rewards from generated_rewards/ensemble/")
    args = parser.parse_args()

    run_ensemble_mitigation(
        envs=args.envs,
        n=args.n,
        aggregation=args.aggregation,
        timesteps=args.timesteps,
        checkpoints=args.checkpoints,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        skip_generation=args.skip_generation,
    )
