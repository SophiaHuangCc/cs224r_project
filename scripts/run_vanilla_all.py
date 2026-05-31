"""
Run vanilla LLM reward generation for all three Fetch tasks.
Generates rewards and does a quick sanity check.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from llm_reward_vanilla import generate_reward_function, compile_reward_fn, VanillaLLMRewardWrapper
import gymnasium as gym
import gymnasium_robotics  # noqa: F401

TASKS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]

for env_id in TASKS:
    print(f"\n{'='*60}")
    print(f"Generating vanilla reward for {env_id}...")
    print(f"{'='*60}")

    reward_code = generate_reward_function(env_id)
    print(f"Generated reward:\n{reward_code}\n")

    # Save
    os.makedirs("generated_rewards", exist_ok=True)
    with open(f"generated_rewards/{env_id}_vanilla.py", "w") as f:
        f.write(reward_code)
    print(f"Saved to generated_rewards/{env_id}_vanilla.py")

    # Sanity check
    reward_fn = compile_reward_fn(reward_code)
    env = gym.make(env_id)
    env = VanillaLLMRewardWrapper(env, reward_fn)

    obs, _ = env.reset()
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"  Step {i+1}: LLM reward={reward:.4f} | Original={info['original_reward']:.4f}")
        if term or trunc:
            obs, _ = env.reset()

    env.close()
    print(f"✅ {env_id} vanilla reward working")

print("\n\nDone! All vanilla rewards saved to generated_rewards/")
