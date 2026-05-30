import os
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

gym.register_envs(gymnasium_robotics)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = gym.make("FetchReach-v4")
env = Monitor(env)

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/fetchreach_baseline/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.98,
)

model.learn(total_timesteps=100_000)
model.save("models/ppo_fetchreach_baseline")

env.close()
print("Saved model to models/ppo_fetchreach_baseline")