"""
Physics-constraint reward wrapper (Mitigation 6).

Subtracts a soft quadratic penalty from the step reward whenever a physical
proxy (action norm, jerk, object speed, object acceleration, drop height,
table-slam) exceeds a threshold. The thresholds intentionally mirror those
used by `SafetyMetricWrapper`, so a step that counts as a "violation" in the
safety report is the same step that gets penalized during training.

Penalty form (per dimension i with value `v_i` and threshold `thr_i`):
    p_i = max(0, v_i - thr_i) ** 2
    total_penalty = coef * sum(p_i)

Only `step()` is overridden; `compute_reward` (used by HER relabeling) is
inherited unchanged so HER sees the pure LLM proxy reward. This is the
right call because HER relabels with counterfactual goals — the physics
penalty is a property of the actual action taken, not of the relabeled goal.

Pass `None` for any threshold to disable that dimension (e.g. drop /
table-slam are meaningless for FetchReach).
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from .metrics import (
    DEFAULT_ACTION_NORM_THRESHOLD,
    DEFAULT_DELTA_ACTION_THRESHOLD,
    DEFAULT_DROP_HEIGHT_THRESHOLD,
    DEFAULT_OBJECT_ACCEL_THRESHOLD,
    DEFAULT_OBJECT_SPEED_THRESHOLD,
    DEFAULT_TABLE_SLAM_ACCEL_THRESHOLD,
    _achieved_goal,
)


# Per-env default constraint sets. Picks the dimensions where each env's
# hacking signature shows up in the eval JSONs.
PHYSICS_PRESETS: dict[str, dict] = {
    "FetchReach-v4": dict(
        action_norm_threshold=DEFAULT_ACTION_NORM_THRESHOLD,
        delta_action_threshold=DEFAULT_DELTA_ACTION_THRESHOLD,
        object_speed_threshold=None,
        object_accel_threshold=None,
        drop_height_threshold=None,
        table_slam_accel_threshold=None,
    ),
    "FetchPickAndPlace-v4": dict(
        action_norm_threshold=DEFAULT_ACTION_NORM_THRESHOLD,
        delta_action_threshold=DEFAULT_DELTA_ACTION_THRESHOLD,
        object_speed_threshold=None,
        object_accel_threshold=None,
        drop_height_threshold=DEFAULT_DROP_HEIGHT_THRESHOLD,
        table_slam_accel_threshold=DEFAULT_TABLE_SLAM_ACCEL_THRESHOLD,
    ),
    "FetchSlide-v4": dict(
        action_norm_threshold=DEFAULT_ACTION_NORM_THRESHOLD,
        delta_action_threshold=DEFAULT_DELTA_ACTION_THRESHOLD,
        object_speed_threshold=DEFAULT_OBJECT_SPEED_THRESHOLD,
        object_accel_threshold=DEFAULT_OBJECT_ACCEL_THRESHOLD,
        drop_height_threshold=None,
        table_slam_accel_threshold=None,
    ),
}


class PhysicsConstraintRewardWrapper(gym.Wrapper):
    """Soft physics-constraint penalty on the training reward.

    Parameters
    ----------
    env :
        Underlying env (typically the LLM-reward wrapper). Must produce
        Fetch-style dict obs with `achieved_goal`.
    coef :
        Penalty coefficient λ. Larger λ → stricter enforcement, but if too
        large the reward signal is dominated by the penalty and learning
        collapses. Sweep over {0.1, 1.0, 10.0} to calibrate.
    *_threshold :
        Per-dimension threshold; `None` disables that dimension. Defaults
        mirror `SafetyMetricWrapper`'s defaults.
    """

    def __init__(
        self,
        env,
        coef: float = 1.0,
        action_norm_threshold: Optional[float] = DEFAULT_ACTION_NORM_THRESHOLD,
        delta_action_threshold: Optional[float] = DEFAULT_DELTA_ACTION_THRESHOLD,
        object_speed_threshold: Optional[float] = None,
        object_accel_threshold: Optional[float] = None,
        drop_height_threshold: Optional[float] = None,
        table_slam_accel_threshold: Optional[float] = None,
    ):
        super().__init__(env)
        self._coef = float(coef)
        self._action_norm_thr = action_norm_threshold
        self._delta_action_thr = delta_action_threshold
        self._object_speed_thr = object_speed_threshold
        self._object_accel_thr = object_accel_threshold
        self._drop_height_thr = drop_height_threshold
        self._table_slam_accel_thr = table_slam_accel_threshold

        self._prev_action: Optional[np.ndarray] = None
        self._prev_pos: Optional[np.ndarray] = None
        self._prev_vel: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = None
        self._prev_pos = _achieved_goal(obs)
        self._prev_vel = np.zeros(3, dtype=np.float64) if self._prev_pos is not None else None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        act = np.asarray(action, dtype=np.float64)

        penalty = 0.0

        if self._action_norm_thr is not None:
            v = float(np.linalg.norm(act))
            penalty += max(0.0, v - self._action_norm_thr) ** 2

        if self._delta_action_thr is not None and self._prev_action is not None:
            v = float(np.linalg.norm(act - self._prev_action))
            penalty += max(0.0, v - self._delta_action_thr) ** 2

        curr_pos = _achieved_goal(obs)
        obj_vel = None
        if curr_pos is not None and self._prev_pos is not None:
            obj_vel = curr_pos - self._prev_pos
            obj_accel = obj_vel - (self._prev_vel if self._prev_vel is not None else np.zeros_like(obj_vel))
            obj_speed = float(np.linalg.norm(obj_vel))
            obj_accel_n = float(np.linalg.norm(obj_accel))

            if self._object_speed_thr is not None:
                penalty += max(0.0, obj_speed - self._object_speed_thr) ** 2
            if self._object_accel_thr is not None:
                penalty += max(0.0, obj_accel_n - self._object_accel_thr) ** 2
            if (
                self._drop_height_thr is not None
                and self._table_slam_accel_thr is not None
                and curr_pos[2] < self._drop_height_thr
            ):
                # Table-slam: dropped low AND high acceleration.
                penalty += max(0.0, obj_accel_n - self._table_slam_accel_thr) ** 2

        if self._drop_height_thr is not None and curr_pos is not None:
            # Standalone drop penalty: how far below the height threshold.
            penalty += max(0.0, self._drop_height_thr - float(curr_pos[2])) ** 2

        info["physics_penalty"] = penalty * self._coef
        modified_reward = float(reward) - self._coef * penalty

        self._prev_action = act
        if curr_pos is not None:
            self._prev_pos = curr_pos
            if obj_vel is not None:
                self._prev_vel = obj_vel

        return obs, modified_reward, terminated, truncated, info
