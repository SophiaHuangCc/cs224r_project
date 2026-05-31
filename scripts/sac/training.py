"""
SAC + HER training for LLM-generated reward functions.

The `llm_reward_*` scripts generate reward code and save it to
`generated_rewards/`. This module reads those reward files (or accepts an
in-memory reward function) and runs SAC+HER, which is a much better fit than
PPO for sparse goal-based Fetch tasks (FetchPickAndPlace, FetchSlide).

HER note: the HerReplayBuffer relabels transitions with future achieved goals
and recomputes reward by calling `env.compute_reward(achieved_goal,
desired_goal, info)`. The LLM reward signature is wider
(`compute_reward(achieved_goal, desired_goal, obs, action, info)`), so the
HER-aware wrapper here also exposes a 3-arg `compute_reward` method that
forwards to the LLM reward with `obs=None, action=None`. LLM rewards that
depend purely on `achieved_goal`/`desired_goal` (the common case) work
correctly. Rewards that depend on `obs`/`action` will receive `None` during
relabeling; the wrapper falls back to the original env reward in that case.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch envs
import numpy as np
from stable_baselines3 import HerReplayBuffer, SAC

from llm_reward_vanilla import VanillaLLMRewardWrapper, compile_reward_fn
from safety import SafetyMetricWrapper, evaluate_with_safety


# --------------------------------------------------------------------------- #
# HER-aware LLM reward wrapper                                                 #
# --------------------------------------------------------------------------- #
class HERLLMRewardWrapper(VanillaLLMRewardWrapper):
    """`VanillaLLMRewardWrapper` that also exposes `compute_reward` for HER relabeling.

    HER's replay buffer calls `compute_reward` on **batched** goals (shape
    `(B, goal_dim)`), but LLM-generated rewards typically use scalar reductions
    (e.g. `np.linalg.norm(achieved_goal - desired_goal)`) which collapse a 2D
    input to a scalar. To keep arbitrary LLM reward code working, this wrapper
    detects the batched case and loops row-by-row so the LLM reward only ever
    sees a single (achieved_goal, desired_goal) pair.
    """

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)

        if achieved_goal.ndim >= 2:
            # Batched call from HER — loop over the batch.
            batch_size = achieved_goal.shape[0]
            # `info` is typically a list/array of dicts of length `batch_size`.
            if isinstance(info, (list, tuple, np.ndarray)) and len(info) == batch_size:
                infos = list(info)
            else:
                infos = [info] * batch_size

            rewards = np.empty(batch_size, dtype=np.float32)
            for i in range(batch_size):
                try:
                    rewards[i] = float(self.reward_fn(
                        achieved_goal[i], desired_goal[i], None, None, infos[i]
                    ))
                except Exception as e:
                    print(f"LLM compute_reward error during HER relabel (row {i}): {e}; using env reward")
                    rewards[i] = float(self.env.unwrapped.compute_reward(
                        achieved_goal[i], desired_goal[i], infos[i]
                    ))
            return rewards

        # Unbatched (single transition) — call LLM reward directly.
        try:
            return float(self.reward_fn(achieved_goal, desired_goal, None, None, info))
        except Exception as e:
            print(f"LLM compute_reward error during HER relabel: {e}; using env reward")
            return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)


# --------------------------------------------------------------------------- #
# Reward file loading                                                          #
# --------------------------------------------------------------------------- #
def load_reward_fn_from_file(path: str) -> Callable:
    with open(path, "r") as f:
        code = f.read()
    return compile_reward_fn(code)


# --------------------------------------------------------------------------- #
# SAC + HER training                                                           #
# --------------------------------------------------------------------------- #
def train_sac_her_with_reward(
    env_id: str,
    reward_fn: Callable,
    timesteps: int = 200_000,
    seed: int = 42,
    eval_episodes: int = 50,
    save_path: Optional[str] = None,
    tensorboard_log: Optional[str] = None,
    verbose: int = 0,
    learning_rate: float = 1e-3,
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    gamma: float = 0.95,
    tau: float = 0.05,
    learning_starts: int = 1_000,
    n_sampled_goal: int = 4,
    goal_selection_strategy: str = "future",
) -> tuple[SAC, dict]:
    """Train SAC+HER on `env_id` using `reward_fn`, then evaluate on the original env."""
    # NOTE: SafetyMetricWrapper must be the OUTER wrapper. The HER replay buffer
    # walks `env.unwrapped` to find a `compute_reward` method, so HERLLMReward
    # remains reachable via the wrapper chain.
    train_env = SafetyMetricWrapper(
        HERLLMRewardWrapper(gym.make(env_id), reward_fn)
    )

    model = SAC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
        ),
        verbose=verbose,
        seed=seed,
        tensorboard_log=tensorboard_log,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        learning_starts=learning_starts,
    )
    model.learn(total_timesteps=timesteps)

    # Evaluate on the original env (true task reward) with safety logging
    # Pass reward_fn to compute hacking_rate (proxy reward vs ground-truth success)
    eval_env = SafetyMetricWrapper(gym.make(env_id))
    metrics = evaluate_with_safety(model, eval_env, n_episodes=eval_episodes, reward_fn=reward_fn)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        model.save(save_path)

    train_env.close()
    eval_env.close()
    return model, metrics
