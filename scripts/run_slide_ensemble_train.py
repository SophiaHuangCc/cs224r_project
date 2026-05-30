"""Train ensemble for FetchSlide-v4 (rewards already generated)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from llm_reward_vanilla import compile_reward_fn
from ensemble_reward import make_ensemble_reward_fn
from sac.train_from_reward import train_with_checkpoints

env_id = "FetchSlide-v4"
reward_dir = "generated_rewards/ensemble/FetchSlide-v4"

# Load 3 reward functions
reward_codes = []
for i in range(1, 4):
    with open(os.path.join(reward_dir, f"reward_{i}.py")) as f:
        reward_codes.append(f.read())

reward_fns = [compile_reward_fn(code) for code in reward_codes]
ensemble_fn = make_ensemble_reward_fn(reward_fns, aggregation="min")

print(f"Training SAC+HER with ensemble (min, N=3) for {env_id}...")
print(f"Checkpoints: 100k, 250k, 500k")

train_with_checkpoints(
    env_id=env_id,
    reward_fn=ensemble_fn,
    timesteps=500_000,
    checkpoints=[100_000, 250_000, 500_000],
    seed=42,
    eval_episodes=100,
    save_dir="models/ensemble",
    results_dir="results/ensemble",
    label=f"{env_id}_ensemble_n3_min",
    tensorboard_log=f"logs/ensemble_{env_id}_min_n3",
)

print("\n✅ FetchSlide ensemble training complete!")
