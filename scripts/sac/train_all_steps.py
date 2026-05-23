"""
Train all 6 SAC+HER scenarios (3 envs × 2 reward types) at 100k, 250k, 500k steps.
Saves models and metrics with step counts in the name.

Usage (local, Apple M4 CPU):
    cd ~/Desktop/cs224r/cs224r_project
    conda run -n cs224r_project python -u scripts/sac/train_all_steps.py --steps 100000
    conda run -n cs224r_project python -u scripts/sac/train_all_steps.py --steps 250000
    conda run -n cs224r_project python -u scripts/sac/train_all_steps.py --steps 500000

    # Run in background (survives terminal close):
    cd ~/Desktop/cs224r/cs224r_project
    nohup conda run -n cs224r_project python -u scripts/sac/train_all_steps.py --steps 500000 > logs/train_500k.log 2>&1 &

    # Skip already-completed scenarios:
    conda run -n cs224r_project python -u scripts/sac/train_all_steps.py --steps 250000 --skip-existing

Usage (Modal, parallel):
    conda run -n cs224r_project modal run scripts/sac/train_modal.py

Estimated wall time per scenario on Apple M4 (single-threaded MuJoCo):
    100k steps ~ 15 min
    250k steps ~ 35 min
    500k steps ~ 70 min
    Total (6 scenarios): 100k=1.5h, 250k=3.5h, 500k=7h
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
os.chdir(_PROJECT_DIR)  # So relative reward paths resolve correctly

from sac import load_reward_fn_from_file, train_sac_her_with_reward  # noqa: E402

# All 6 scenarios: (env_id, reward_file, label)
SCENARIOS = [
    ("FetchReach-v4", "generated_rewards/FetchReach-v4_vanilla.py", "FetchReach_vanilla"),
    ("FetchReach-v4", "generated_rewards/eureka/FetchReach-v4_iter1.py", "FetchReach_eureka"),
    ("FetchPickAndPlace-v4", "generated_rewards/FetchPickAndPlace-v4_vanilla.py", "FetchPickAndPlace_vanilla"),
    ("FetchPickAndPlace-v4", "generated_rewards/eureka/FetchPickAndPlace-v4_best.py", "FetchPickAndPlace_eureka"),
    ("FetchSlide-v4", "generated_rewards/FetchSlide-v4_vanilla.py", "FetchSlide_vanilla"),
    ("FetchSlide-v4", "generated_rewards/eureka/FetchSlide-v4_best.py", "FetchSlide_eureka"),
]


def steps_label(steps: int) -> str:
    if steps >= 1_000_000:
        return f"{steps // 1_000_000}M"
    return f"{steps // 1000}k"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True, help="Total timesteps (e.g. 100000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true", help="Skip if metrics file exists")
    args = parser.parse_args()

    slabel = steps_label(args.steps)
    model_dir = f"models/sac_{slabel}"
    log_dir = f"logs/sac_{slabel}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    for env_id, reward_path, label in SCENARIOS:
        metrics_path = f"{log_dir}/{label}_metrics.json"
        save_path = f"{model_dir}/{label}"

        if args.skip_existing and os.path.exists(metrics_path):
            print(f"\n⏭️  Skipping {label} @ {slabel} (metrics exist)")
            continue

        print(f"\n{'='*60}")
        print(f"🚀 Training {label} @ {slabel} ({args.steps} steps)")
        print(f"   env={env_id}  reward={reward_path}")
        print(f"{'='*60}")

        # Check reward file exists
        if not os.path.exists(reward_path):
            print(f"❌ Reward file not found: {reward_path} — skipping")
            continue

        reward_fn = load_reward_fn_from_file(reward_path)

        _, metrics = train_sac_her_with_reward(
            env_id=env_id,
            reward_fn=reward_fn,
            timesteps=args.steps,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            save_path=save_path,
            verbose=1,
        )

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ {label} @ {slabel} done — success_rate={metrics.get('success_rate', 'N/A')}")
        print(f"   Saved: {save_path}, {metrics_path}")

    print(f"\n{'='*60}")
    print(f"🏁 All scenarios complete for {slabel}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
