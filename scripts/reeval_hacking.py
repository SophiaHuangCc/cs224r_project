"""Re-evaluate hacking rate for Eureka SAC+HER with the fixed obs passing."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

import gymnasium as gym
import gymnasium_robotics  # noqa
from stable_baselines3 import SAC
from llm_reward_vanilla import compile_reward_fn
from safety import SafetyMetricWrapper, evaluate_with_safety

ENVS = ["FetchReach-v4", "FetchPickAndPlace-v4", "FetchSlide-v4"]
CHECKPOINTS = [100_000, 250_000, 500_000]

for env_id in ENVS:
    # Load the eureka_sac best reward
    reward_path = f"generated_rewards/eureka_sac/{env_id}_best.py"
    with open(reward_path) as f:
        reward_code = f.read()
    reward_fn = compile_reward_fn(reward_code)
    
    for ckpt in CHECKPOINTS:
        model_path = f"models/eureka_sac/{env_id}_eureka_{ckpt//1000}k"
        if not os.path.exists(model_path + ".zip"):
            print(f"  SKIP {env_id} @ {ckpt//1000}k (no model)")
            continue
        
        print(f"\n  Evaluating {env_id} @ {ckpt//1000}k...")
        eval_env = SafetyMetricWrapper(gym.make(env_id))
        model = SAC.load(model_path, env=eval_env)
        metrics = evaluate_with_safety(model, eval_env, n_episodes=100, reward_fn=reward_fn)
        eval_env.close()
        
        print(f"    success={metrics['success_rate']:.1%} hacking={metrics.get('hacking_rate', 'N/A'):.1%} "
              f"mean_proxy={metrics.get('mean_proxy_reward', 0):.2f} threshold={metrics.get('hacking_threshold', 0):.2f}")
        
        # Save
        out_path = f"results/eureka_sac/{env_id}_eureka_{ckpt//1000}k.json"
        os.makedirs("results/eureka_sac", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

print("\n✅ All re-evaluations done!")
