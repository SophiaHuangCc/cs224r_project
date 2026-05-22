"""
Standalone CLI: train PPO on a Fetch env using a reward function loaded from
`generated_rewards/`.

Examples:
    # vanilla LLM reward
    python scripts/ppo/train_from_reward.py \
        --env FetchReach-v4 \
        --reward generated_rewards/FetchReach-v4_vanilla.py \
        --timesteps 200000

    # Eureka best reward
    python scripts/ppo/train_from_reward.py \
        --env FetchPickAndPlace-v4 \
        --reward generated_rewards/eureka/FetchPickAndPlace-v4_best.py \
        --timesteps 1000000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make sibling packages importable when run as a script.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS_DIR)

from ppo import load_reward_fn_from_file, train_ppo_with_reward  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train PPO from a saved reward file.")
    parser.add_argument("--env", required=True, help="Gym env id, e.g. FetchReach-v4")
    parser.add_argument("--reward", required=True, help="Path to a saved reward .py file")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--save-path", default=None,
                        help="Where to save the trained PPO model (default: derived from reward path)")
    parser.add_argument("--metrics-path", default=None,
                        help="Where to save eval metrics JSON (default: alongside save-path)")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    reward_fn = load_reward_fn_from_file(args.reward)

    if args.save_path is None:
        reward_stem = os.path.splitext(os.path.basename(args.reward))[0]
        args.save_path = f"models/from_reward/{reward_stem}"
    if args.metrics_path is None:
        args.metrics_path = f"{args.save_path}_metrics.json"

    print(f"Training PPO on {args.env} with reward from {args.reward}")
    print(f"  timesteps={args.timesteps}  seed={args.seed}  eval_episodes={args.eval_episodes}")

    _, metrics = train_ppo_with_reward(
        env_id=args.env,
        reward_fn=reward_fn,
        timesteps=args.timesteps,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        save_path=args.save_path,
        verbose=args.verbose,
    )

    os.makedirs(os.path.dirname(args.metrics_path) or ".", exist_ok=True)
    with open(args.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {args.save_path}")
    print(f"Saved metrics to {args.metrics_path}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
