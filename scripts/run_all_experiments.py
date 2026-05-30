#!/usr/bin/env python3
"""
Run everything: Eureka baseline (SAC+HER) + Ensemble mitigation.
All 3 Fetch envs, checkpoints at 100k/250k/500k.

This is the top-level script that runs the full experiment pipeline.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

ENVS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]
CHECKPOINTS = [100_000, 250_000, 500_000]

total_start = time.time()

# ==========================================================================
# PART 1: Eureka Baseline (SAC+HER)
# ==========================================================================
print("\n" + "=" * 70)
print("PART 1: EUREKA BASELINE (SAC+HER)")
print("=" * 70)

# Step 1a: Generate Eureka rewards
print("\n--- Step 1a: Generating Eureka rewards (SAC+HER) ---")
from llm_reward_eureka_sac import eureka_loop_sac
from llm_reward_vanilla import compile_reward_fn
from sac.train_from_reward import train_with_checkpoints
import json

for env_id in ENVS:
    best_path = f"generated_rewards/eureka_sac/{env_id}_best.py"
    if os.path.exists(best_path):
        print(f"  {env_id}: reward already exists, skipping")
        continue

    print(f"\n  {env_id}: running Eureka loop...")
    history = eureka_loop_sac(
        env_id=env_id,
        n_iterations=3,
        timesteps_per_iter=50_000,
        eval_episodes=50,
    )

    best = max(history, key=lambda h: h["metrics"]["success_rate"])
    print(f"  {env_id}: best iter {best['iteration']} (success={best['metrics']['success_rate']:.1%})")

    os.makedirs("generated_rewards/eureka_sac", exist_ok=True)
    with open(best_path, "w") as f:
        f.write(best["reward_code"])
    with open(f"generated_rewards/eureka_sac/{env_id}_best_metrics.json", "w") as f:
        json.dump(best["metrics"], f, indent=2)

print("\n✅ All Eureka rewards generated.")

# Step 1b: Train baselines with checkpoints
print("\n--- Step 1b: Training Eureka baselines (100k/250k/500k) ---")

for env_id in ENVS:
    reward_path = f"generated_rewards/eureka_sac/{env_id}_best.py"
    summary_path = f"results/eureka_sac/{env_id}_eureka_summary.json"

    if os.path.exists(summary_path):
        print(f"  {env_id}: results already exist, skipping")
        continue

    with open(reward_path) as f:
        reward_code = f.read()
    reward_fn = compile_reward_fn(reward_code)

    print(f"\n  {env_id}: training with checkpoints...")
    train_with_checkpoints(
        env_id=env_id,
        reward_fn=reward_fn,
        timesteps=500_000,
        checkpoints=CHECKPOINTS,
        seed=42,
        eval_episodes=100,
        save_dir="models/eureka_sac",
        results_dir="results/eureka_sac",
        label=f"{env_id}_eureka",
    )

print("\n✅ All Eureka baselines trained.")

# ==========================================================================
# PART 2: Ensemble Mitigation
# ==========================================================================
print("\n" + "=" * 70)
print("PART 2: ENSEMBLE MITIGATION")
print("=" * 70)

from run_ensemble_mitigation import run_ensemble_mitigation

run_ensemble_mitigation(
    envs=ENVS,
    n=3,
    aggregation="min",
    timesteps=500_000,
    checkpoints=CHECKPOINTS,
    eval_episodes=100,
    seed=42,
    skip_generation=False,
)

# ==========================================================================
# DONE
# ==========================================================================
total_time = time.time() - total_start
print(f"\n\n{'=' * 70}")
print(f"ALL EXPERIMENTS COMPLETE")
print(f"Total time: {total_time/3600:.1f} hours")
print(f"{'=' * 70}")
print(f"\nResults:")
print(f"  Eureka baseline: results/eureka_sac/")
print(f"  Ensemble:        results/ensemble/")
