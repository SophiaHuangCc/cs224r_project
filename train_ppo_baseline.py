import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v4")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_fetchreach")