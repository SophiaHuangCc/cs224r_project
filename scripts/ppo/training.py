"""
PPO training for LLM-generated reward functions.

This module is the single home for PPO logic. The `llm_reward_*` scripts
generate reward code and save it to `generated_rewards/`. This module reads
those reward files (or accepts an in-memory reward function) and runs PPO.

Safety-metric tracking + evaluation lives in `scripts/safety/` and is
imported from here.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch envs
from stable_baselines3 import PPO

# Reuse the wrapper/compiler from the vanilla module so reward semantics stay consistent.
from llm_reward_vanilla import VanillaLLMRewardWrapper, compile_reward_fn
from safety import SafetyMetricWrapper, evaluate_with_safety


# --------------------------------------------------------------------------- #
# Reward file loading                                                          #
# --------------------------------------------------------------------------- #
def load_reward_fn_from_file(path: str) -> Callable:
    """Read a saved reward .py file from generated_rewards/ and compile it."""
    with open(path, "r") as f:
        code = f.read()
    return compile_reward_fn(code)


# --------------------------------------------------------------------------- #
# PPO training                                                                 #
# --------------------------------------------------------------------------- #
def train_ppo_with_reward(
    env_id: str,
    reward_fn: Callable,
    timesteps: int = 50_000,
    seed: int = 42,
    eval_episodes: int = 50,
    save_path: Optional[str] = None,
    verbose: int = 0,
    n_steps: int = 1024,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
) -> tuple[PPO, dict]:
    """Train PPO on `env_id` using `reward_fn`, then evaluate on the original env."""
    train_env = SafetyMetricWrapper(VanillaLLMRewardWrapper(gym.make(env_id), reward_fn))

    model = PPO(
        "MultiInputPolicy",
        train_env,
        verbose=verbose,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    model.learn(total_timesteps=timesteps)

    # Evaluate on the original env reward (true task success) with safety logging
    eval_env = SafetyMetricWrapper(gym.make(env_id))
    metrics = evaluate_with_safety(model, eval_env, n_episodes=eval_episodes)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        model.save(save_path)

    train_env.close()
    eval_env.close()
    return model, metrics
