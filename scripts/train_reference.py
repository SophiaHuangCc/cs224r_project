"""
Train sparse-reward SAC+HER reference policies for all 3 Fetch tasks.

Required as a prerequisite for KL-constrained training (Mitigation 2).
The reference is trained on each env's native sparse reward (0 / -1) so the
KL penalty in train_local.py --reward-type kl pulls the learned policy
toward "normal" task-solving behavior rather than reward-hacking exploits.

Usage:
    python scripts/train_reference.py
    python scripts/train_reference.py --env FetchReach-v4
    python scripts/train_reference.py --skip-existing
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
from stable_baselines3 import HerReplayBuffer, SAC

from sac.training import HER_KWARGS, SAC_KWARGS
from safety import SafetyMetricWrapper, evaluate_with_safety

TASKS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]
TIMESTEPS = 500_000
SEED = 42
EVAL_EPISODES = 100


def train_reference(env_id: str, timesteps: int, seed: int, skip_existing: bool) -> None:
    save_path = f"models/reference/{env_id}"
    if skip_existing and os.path.exists(save_path + ".zip"):
        print(f"  {env_id}: reference already exists, skipping")
        return

    os.makedirs("models/reference", exist_ok=True)
    os.makedirs("results/reference", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training sparse-reward reference: {env_id}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}")

    # No HERLLMRewardWrapper — use the env's native sparse reward.
    train_env = SafetyMetricWrapper(gym.make(env_id))
    model = SAC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=HER_KWARGS,
        verbose=1,
        seed=seed,
        tensorboard_log=f"logs/reference/{env_id}",
        **SAC_KWARGS,
    )

    start = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start

    model.save(save_path)
    print(f"  Saved: {save_path}")

    eval_env = SafetyMetricWrapper(gym.make(env_id))
    metrics = evaluate_with_safety(model, eval_env, n_episodes=EVAL_EPISODES)
    eval_env.close()
    train_env.close()

    metrics["train_time_seconds"] = train_time
    with open(f"results/reference/{env_id}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  Train time:   {train_time/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description="Train sparse-reward reference policies for KL training.")
    parser.add_argument("--env", choices=TASKS + ["all"], default="all")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    tasks = TASKS if args.env == "all" else [args.env]

    print(f"\nTraining reference policies for {tasks}")
    print(f"  Timesteps: {args.timesteps:,}\n")

    start = time.time()
    for task in tasks:
        train_reference(task, args.timesteps, args.seed, args.skip_existing)

    print(f"\n{'='*60}")
    print(f"Done. Total time: {(time.time() - start)/3600:.1f}h")
    print(f"References: models/reference/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
