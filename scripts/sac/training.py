"""
SAC + HER training for LLM-generated reward functions.

The `llm_reward_*` scripts generate reward code and save it to
`generated_rewards/`. This module reads those reward files (or accepts an
in-memory reward function) and runs SAC+HER, which is a much better fit than
PPO for sparse goal-based Fetch tasks (FetchPickAndPlace, FetchSlide).

Two entry points:
- `train_sac_her_with_reward`: train once, evaluate once. Used by the Eureka
  reward-search loop where the model is throwaway.
- `train_with_checkpoints`: train once, save + evaluate at multiple step
  counts. Used by `train_local.py` / `train_modal.py` for final training.

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

import json
import os
import time
from typing import Callable, Optional

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch envs
import numpy as np
from stable_baselines3 import HerReplayBuffer, SAC

from llm_reward_vanilla import VanillaLLMRewardWrapper, compile_reward_fn
from safety import SafetyMetricWrapper, evaluate_with_safety


# SAC+HER defaults — same everywhere for fair comparison across reward types.
SAC_KWARGS = dict(
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.95,
    tau=0.05,
    learning_starts=1_000,
)
HER_KWARGS = dict(
    n_sampled_goal=4,
    goal_selection_strategy="future",
)


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
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def load_reward_fn_from_file(path: str) -> Callable:
    """Compile a reward .py file into a callable."""
    with open(path, "r") as f:
        code = f.read()
    return compile_reward_fn(code)


def _make_train_env(env_id: str, reward_fn: Callable) -> SafetyMetricWrapper:
    """Build the training env: SafetyMetricWrapper(HERLLMRewardWrapper(gym.make(...))).

    SafetyMetricWrapper must be the OUTER wrapper. The HER replay buffer walks
    `env.unwrapped` to find `compute_reward`, so HERLLMRewardWrapper stays
    reachable via the wrapper chain.
    """
    return SafetyMetricWrapper(
        HERLLMRewardWrapper(gym.make(env_id), reward_fn)
    )


def _make_model(
    env: gym.Env,
    seed: int = 42,
    tensorboard_log: Optional[str] = None,
    verbose: int = 0,
) -> SAC:
    """Construct an SAC+HER model with the project-standard hyperparameters."""
    return SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=HER_KWARGS,
        verbose=verbose,
        seed=seed,
        tensorboard_log=tensorboard_log,
        **SAC_KWARGS,
    )


# --------------------------------------------------------------------------- #
# Training entry points                                                        #
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
) -> tuple[SAC, dict]:
    """Train SAC+HER once, evaluate at the end. Used by the Eureka reward search."""
    train_env = _make_train_env(env_id, reward_fn)
    model = _make_model(train_env, seed=seed, tensorboard_log=tensorboard_log, verbose=verbose)
    model.learn(total_timesteps=timesteps)

    # Evaluate on the original env (true task reward) with safety logging.
    # Pass reward_fn so evaluate_with_safety can also compute hacking_rate.
    eval_env = SafetyMetricWrapper(gym.make(env_id))
    metrics = evaluate_with_safety(model, eval_env, n_episodes=eval_episodes, reward_fn=reward_fn)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        model.save(save_path)

    train_env.close()
    eval_env.close()
    return model, metrics


def train_with_checkpoints(
    env_id: str,
    reward_fn: Callable,
    timesteps: int,
    checkpoints: list[int],
    seed: int = 42,
    eval_episodes: int = 100,
    save_dir: str = "models",
    results_dir: str = "results",
    label: str = "",
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
) -> dict[int, dict]:
    """Train SAC+HER once, saving + evaluating at each step count in `checkpoints`.

    Writes one JSON per checkpoint and a summary JSON. Returns a
    `{checkpoint_steps: metrics}` dict.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    checkpoints = sorted(set(checkpoints))
    if timesteps not in checkpoints:
        checkpoints.append(timesteps)

    train_env = _make_train_env(env_id, reward_fn)
    model = _make_model(train_env, seed=seed, tensorboard_log=tensorboard_log, verbose=verbose)

    all_metrics: dict[int, dict] = {}
    steps_trained = 0
    start_time = time.time()

    for checkpoint in checkpoints:
        steps_to_train = checkpoint - steps_trained
        if steps_to_train <= 0:
            continue

        print(f"\n  Training {steps_trained//1000}k → {checkpoint//1000}k ({steps_to_train:,} steps)...")
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        steps_trained = checkpoint

        model_path = os.path.join(save_dir, f"{label}_{checkpoint//1000}k")
        model.save(model_path)
        print(f"  Saved: {model_path}")

        print(f"  Evaluating ({eval_episodes} episodes)...")
        eval_env = SafetyMetricWrapper(gym.make(env_id))
        metrics = evaluate_with_safety(
            model, eval_env, n_episodes=eval_episodes, reward_fn=reward_fn
        )
        eval_env.close()

        metrics["checkpoint_steps"] = checkpoint
        results_path = os.path.join(results_dir, f"{label}_{checkpoint//1000}k.json")
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(
            f"  {checkpoint//1000}k: success={metrics['success_rate']:.1%} | "
            f"hacking={metrics.get('hacking_rate', 'N/A')} | "
            f"action_mag={metrics['mean_action_mag']:.4f}"
        )
        all_metrics[checkpoint] = metrics

    train_time = time.time() - start_time
    train_env.close()

    summary = {
        "config": {
            "env_id": env_id,
            "label": label,
            "checkpoints": checkpoints,
            "eval_episodes": eval_episodes,
            "seed": seed,
            "train_time_seconds": train_time,
        },
        "results_by_checkpoint": {
            f"{c//1000}k": all_metrics[c] for c in checkpoints if c in all_metrics
        },
    }
    summary_path = os.path.join(results_dir, f"{label}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total train time: {train_time/60:.1f} min")
    print(f"  Summary: {summary_path}")

    return all_metrics
