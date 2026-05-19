import os
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.monitor import Monitor

gym.register_envs(gymnasium_robotics)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = gym.make("FetchPickAndPlace-v4")
env = Monitor(env)

# FetchPickAndPlace is sparse-goal based
# so SAC (soft actor-critic) + HER (hindsight experience replay) is usually much easier
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    verbose=1,
    tensorboard_log="logs/fetchpick_baseline/",
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.95,
    tau=0.05,
    learning_starts=1_000,
)

model.learn(total_timesteps=200_000)
model.save("models/sac_her_fetchpick_baseline")

env.close()
print("Saved model to models/sac_her_fetchpick_baseline")