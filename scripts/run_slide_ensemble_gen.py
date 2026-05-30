"""Generate ensemble rewards for FetchSlide only (no full training)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from ensemble_reward import generate_eureka_ensemble_rewards

print("Generating N=3 Eureka-quality ensemble rewards for FetchSlide-v4...")
reward_codes = generate_eureka_ensemble_rewards(
    env_id="FetchSlide-v4",
    n=3,
    eureka_iters=3,
    timesteps_per_iter=50_000,
    eval_episodes=50,
    save_dir="generated_rewards/ensemble/FetchSlide-v4",
    seed=42,
)
print(f"\n✅ Done! Generated {len(reward_codes)} rewards.")
