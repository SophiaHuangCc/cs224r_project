"""
Safety metric tracking + evaluation utilities.

NOTE: All safety-metric tracking is currently COMMENTED OUT. The wrapper acts
as a transparent passthrough and `evaluate_with_safety` returns zeros for
safety fields. Downstream callers (PPO/SAC training, Eureka) keep working
unchanged. To re-enable, uncomment the marked blocks below.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class SafetyMetricWrapper(gym.Wrapper):
    """Passthrough wrapper. Safety-metric tracking is disabled (commented out)."""

    def __init__(self, env):
        super().__init__(env)
        # --- safety tracking (disabled) ---
        # self._prev_action = None
        # self._episode_metrics = self._reset_metrics()

    # def _reset_metrics(self):
    #     return {
    #         "action_magnitudes": [],
    #         "jerks": [],
    #         "max_action_mag": 0.0,
    #         "steps": 0,
    #     }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # --- safety tracking (disabled) ---
        # self._prev_action = None
        # self._episode_metrics = self._reset_metrics()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- safety tracking (disabled) ---
        # mag = np.linalg.norm(action)
        # self._episode_metrics["action_magnitudes"].append(mag)
        # self._episode_metrics["max_action_mag"] = max(
        #     self._episode_metrics["max_action_mag"], mag
        # )
        #
        # if self._prev_action is not None:
        #     jerk = np.linalg.norm(action - self._prev_action)
        #     self._episode_metrics["jerks"].append(jerk)
        #
        # self._prev_action = action.copy()
        # self._episode_metrics["steps"] += 1
        #
        # if terminated or truncated:
        #     info["safety_metrics"] = {
        #         "mean_action_mag": float(np.mean(self._episode_metrics["action_magnitudes"])),
        #         "mean_jerk": float(np.mean(self._episode_metrics["jerks"])) if self._episode_metrics["jerks"] else 0.0,
        #         "max_action_mag": float(self._episode_metrics["max_action_mag"]),
        #         "ep_length": self._episode_metrics["steps"],
        #     }

        return obs, reward, terminated, truncated, info


def evaluate_with_safety(model, env, n_episodes: int = 50) -> dict:
    """Evaluate a policy. Safety-metric collection is currently disabled.

    Returns task metrics (success_rate, mean_reward, mean_ep_len) plus zeroed
    safety fields so callers don't need to change.
    """
    successes, rewards, ep_lens = [], [], []
    # --- safety tracking (disabled) ---
    # action_mags, jerks, max_mags = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(episode_reward)
        successes.append(info.get("is_success", False))
        ep_lens.append(steps)

        # --- safety tracking (disabled) ---
        # if "safety_metrics" in info:
        #     sm = info["safety_metrics"]
        #     action_mags.append(sm["mean_action_mag"])
        #     jerks.append(sm["mean_jerk"])
        #     max_mags.append(sm["max_action_mag"])

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_ep_len": float(np.mean(ep_lens)) if ep_lens else 0.0,
        # --- safety fields (disabled — zeroed) ---
        "mean_action_mag": 0.0,
        "mean_jerk": 0.0,
        "max_action_mag": 0.0,
        # "mean_action_mag": float(np.mean(action_mags)) if action_mags else 0.0,
        # "mean_jerk": float(np.mean(jerks)) if jerks else 0.0,
        # "max_action_mag": float(np.max(max_mags)) if max_mags else 0.0,
    }
