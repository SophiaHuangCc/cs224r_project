"""
Run Eureka (SAC+HER) for all three Fetch scenarios.
3 iterations each, 50k steps per iter.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from llm_reward_eureka_sac import eureka_loop_sac
import json

TASKS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]

for env_id in TASKS:
    print(f"\n{'='*60}")
    print(f"Running Eureka (SAC+HER) for {env_id}...")
    print(f"{'='*60}")

    history = eureka_loop_sac(
        env_id=env_id,
        n_iterations=3,
        timesteps_per_iter=50_000,
        eval_episodes=50,
    )

    # Summary
    print(f"\n--- EUREKA (SAC+HER) SUMMARY for {env_id} ---")
    best_iter = max(history, key=lambda h: h["metrics"]["success_rate"])
    for h in history:
        m = h["metrics"]
        marker = " ← BEST" if h == best_iter else ""
        print(
            f"Iter {h['iteration']}: success={m['success_rate']:.1%} | "
            f"action_mag={m['mean_action_mag']:.3f} | "
            f"jerk={m['mean_jerk']:.3f}{marker}"
        )
    print(f"Best iteration: {best_iter['iteration']} (success={best_iter['metrics']['success_rate']:.1%})")

    # Best reward already saved by eureka_loop_sac, but also save here for clarity
    os.makedirs("generated_rewards/eureka_sac", exist_ok=True)
    with open(f"generated_rewards/eureka_sac/{env_id}_best.py", "w") as f:
        f.write(best_iter["reward_code"])
    with open(f"generated_rewards/eureka_sac/{env_id}_best_metrics.json", "w") as f:
        json.dump(best_iter["metrics"], f, indent=2)

print("\n\n✅ All Eureka (SAC+HER) runs complete!")
print("Best rewards saved to generated_rewards/eureka_sac/<task>_best.py")
