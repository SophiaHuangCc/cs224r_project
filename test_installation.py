import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v4")
print("Observation shape:", env.observation_space)
print("Action shape:", env.action_space)