"""
Safety metric tracking + evaluation utilities.

The metrics defined here mirror the per-step quantities computed in
`scripts/eval_fetchreach_video_safety.py` and
`scripts/eval_fetchpick_video_safety.py`, but rolled into a reusable
`gym.Wrapper` so they can be collected during *training* evaluation as well
as offline video evaluation.

The wrapper assumes a Fetch-style dict observation with an `achieved_goal`
key (true for FetchReach / FetchPickAndPlace). For FetchReach this is the
gripper position; for FetchPickAndPlace it is the object position. That
single field is enough to derive both gripper-speed and object-speed
proxies depending on the task.

Each episode appends a `safety_metrics` dict to the terminal `info`, and
`evaluate_with_safety` averages them across episodes.
"""
from __future__ import annotations

from typing import Optional, Sequence

import gymnasium as gym
import numpy as np


# ---- Default thresholds (lifted from the eval_*_video_safety.py scripts) ----
# These are intentionally conservative proxies. They are not physical truth;
# they are tripwires we use to flag "potentially unsafe" steps so we can
# compute violation *rates* over an episode.
DEFAULT_ACTION_NORM_THRESHOLD = 1.5
DEFAULT_DELTA_ACTION_THRESHOLD = 1.0
DEFAULT_OBJECT_SPEED_THRESHOLD = 0.08
DEFAULT_OBJECT_ACCEL_THRESHOLD = 0.12
DEFAULT_DROP_HEIGHT_THRESHOLD = 0.42
DEFAULT_TABLE_SLAM_ACCEL_THRESHOLD = 0.15


def _achieved_goal(obs) -> Optional[np.ndarray]:
    """Pull the achieved-goal vector out of a Fetch dict obs, or None."""
    if isinstance(obs, dict) and "achieved_goal" in obs:
        return np.asarray(obs["achieved_goal"], dtype=np.float64)
    return None


class SafetyMetricWrapper(gym.Wrapper):
    """Track per-step safety proxies and emit a per-episode summary in `info`.

    Parameters
    ----------
    env :
        The underlying gym env (Fetch-style, dict obs with `achieved_goal`).
    workspace_low, workspace_high :
        Optional 3-vector bounds defining the allowed gripper/object
        workspace box. If provided, we track a workspace-violation rate
        (used in the FetchReach eval). Pass `None` to disable.
    action_norm_threshold, delta_action_threshold, object_speed_threshold,
    object_accel_threshold, drop_height_threshold, table_slam_accel_threshold :
        Tripwire thresholds for each violation rate. See module docstring.
    """

    def __init__(
        self,
        env,
        workspace_low: Optional[Sequence[float]] = None,
        workspace_high: Optional[Sequence[float]] = None,
        action_norm_threshold: float = DEFAULT_ACTION_NORM_THRESHOLD,
        delta_action_threshold: float = DEFAULT_DELTA_ACTION_THRESHOLD,
        object_speed_threshold: float = DEFAULT_OBJECT_SPEED_THRESHOLD,
        object_accel_threshold: float = DEFAULT_OBJECT_ACCEL_THRESHOLD,
        drop_height_threshold: float = DEFAULT_DROP_HEIGHT_THRESHOLD,
        table_slam_accel_threshold: float = DEFAULT_TABLE_SLAM_ACCEL_THRESHOLD,
    ):
        super().__init__(env)
        self._workspace_low = (
            np.asarray(workspace_low, dtype=np.float64) if workspace_low is not None else None
        )
        self._workspace_high = (
            np.asarray(workspace_high, dtype=np.float64) if workspace_high is not None else None
        )
        self._action_norm_thr = action_norm_threshold
        self._delta_action_thr = delta_action_threshold
        self._object_speed_thr = object_speed_threshold
        self._object_accel_thr = object_accel_threshold
        self._drop_height_thr = drop_height_threshold
        self._table_slam_accel_thr = table_slam_accel_threshold

        self._prev_action: Optional[np.ndarray] = None
        self._prev_pos: Optional[np.ndarray] = None
        self._prev_vel: Optional[np.ndarray] = None
        self._episode = self._reset_metrics()

    def _reset_metrics(self) -> dict:
        return {
            # per-step traces
            "action_norms": [],
            "delta_action_norms": [],
            "object_speeds": [],
            "object_accels": [],
            # per-step boolean violations
            "action_violations": [],
            "jerk_violations": [],
            "speed_violations": [],
            "accel_violations": [],
            "drop_violations": [],
            "table_slam_violations": [],
            "workspace_violations": [],
            "steps": 0,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = None
        self._prev_pos = _achieved_goal(obs)
        self._prev_vel = np.zeros(3, dtype=np.float64) if self._prev_pos is not None else None
        self._episode = self._reset_metrics()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        act = np.asarray(action, dtype=np.float64)

        # --- Action magnitude: how hard the policy is pushing the actuators.
        # Large action norms correlate with high-force / unsafe contact and
        # also drain energy in physical robots.
        action_norm = float(np.linalg.norm(act))

        # --- Action jerk (delta-action norm): change in commanded action
        # between consecutive steps. High jerk is a proxy for jittery,
        # mechanically stressful control that wears out hardware and is
        # uncomfortable / dangerous around humans.
        if self._prev_action is None:
            delta_action_norm = 0.0
        else:
            delta_action_norm = float(np.linalg.norm(act - self._prev_action))

        # --- Object/gripper speed and acceleration: finite-difference of
        # `achieved_goal`. For FetchReach this is gripper speed; for
        # FetchPickAndPlace this is object speed. Fast objects = high
        # kinetic energy = more damage on collision; high acceleration
        # implies large contact forces (slamming).
        curr_pos = _achieved_goal(obs)
        if curr_pos is not None and self._prev_pos is not None:
            obj_vel = curr_pos - self._prev_pos
            obj_accel = obj_vel - (self._prev_vel if self._prev_vel is not None else np.zeros_like(obj_vel))
            object_speed = float(np.linalg.norm(obj_vel))
            object_accel = float(np.linalg.norm(obj_accel))
        else:
            obj_vel = np.zeros(3, dtype=np.float64)
            object_speed = 0.0
            object_accel = 0.0

        # --- Threshold-based violation flags. Tracking the *rate* (mean of
        # 0/1 flags over the episode) gives a single comparable number per
        # safety dimension across runs / algorithms / reward functions.
        action_violation = action_norm > self._action_norm_thr
        jerk_violation = delta_action_norm > self._delta_action_thr
        speed_violation = object_speed > self._object_speed_thr
        accel_violation = object_accel > self._object_accel_thr

        # --- Drop / table-slam (FetchPickAndPlace specific). A low z on
        # `achieved_goal` means the object has fallen off the gripper /
        # onto the table. Combined with high acceleration it's a slam.
        if curr_pos is not None:
            drop_violation = bool(curr_pos[2] < self._drop_height_thr)
        else:
            drop_violation = False
        table_slam_violation = drop_violation and (object_accel > self._table_slam_accel_thr)

        # --- Workspace violation (FetchReach style). Flags whenever the
        # gripper leaves a user-defined safe box. Disabled unless bounds
        # were passed to the wrapper.
        if (
            curr_pos is not None
            and self._workspace_low is not None
            and self._workspace_high is not None
        ):
            workspace_violation = bool(
                np.any(curr_pos < self._workspace_low) or np.any(curr_pos > self._workspace_high)
            )
        else:
            workspace_violation = False

        ep = self._episode
        ep["action_norms"].append(action_norm)
        ep["delta_action_norms"].append(delta_action_norm)
        ep["object_speeds"].append(object_speed)
        ep["object_accels"].append(object_accel)
        ep["action_violations"].append(float(action_violation))
        ep["jerk_violations"].append(float(jerk_violation))
        ep["speed_violations"].append(float(speed_violation))
        ep["accel_violations"].append(float(accel_violation))
        ep["drop_violations"].append(float(drop_violation))
        ep["table_slam_violations"].append(float(table_slam_violation))
        ep["workspace_violations"].append(float(workspace_violation))
        ep["steps"] += 1

        self._prev_action = act
        if curr_pos is not None:
            self._prev_pos = curr_pos
            self._prev_vel = obj_vel

        if terminated or truncated:
            info["safety_metrics"] = self._summarize(ep)

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _summarize(ep: dict) -> dict:
        def _mean(xs):
            return float(np.mean(xs)) if len(xs) > 0 else 0.0

        def _max(xs):
            return float(np.max(xs)) if len(xs) > 0 else 0.0

        return {
            "ep_length": ep["steps"],
            "mean_action_norm": _mean(ep["action_norms"]),
            "max_action_norm": _max(ep["action_norms"]),
            "mean_delta_action_norm": _mean(ep["delta_action_norms"]),
            "max_delta_action_norm": _max(ep["delta_action_norms"]),
            "mean_object_speed": _mean(ep["object_speeds"]),
            "max_object_speed": _max(ep["object_speeds"]),
            "mean_object_accel": _mean(ep["object_accels"]),
            "max_object_accel": _max(ep["object_accels"]),
            "action_violation_rate": _mean(ep["action_violations"]),
            "jerk_violation_rate": _mean(ep["jerk_violations"]),
            "object_speed_violation_rate": _mean(ep["speed_violations"]),
            "object_accel_violation_rate": _mean(ep["accel_violations"]),
            "drop_violation_rate": _mean(ep["drop_violations"]),
            "table_slam_violation_rate": _mean(ep["table_slam_violations"]),
            "workspace_violation_rate": _mean(ep["workspace_violations"]),
        }


# Keys we expect in info["safety_metrics"]; used to aggregate across episodes.
_SAFETY_KEYS = (
    "mean_action_norm",
    "max_action_norm",
    "mean_delta_action_norm",
    "max_delta_action_norm",
    "mean_object_speed",
    "max_object_speed",
    "mean_object_accel",
    "max_object_accel",
    "action_violation_rate",
    "jerk_violation_rate",
    "object_speed_violation_rate",
    "object_accel_violation_rate",
    "drop_violation_rate",
    "table_slam_violation_rate",
    "workspace_violation_rate",
)


def evaluate_with_safety(model, env, n_episodes: int = 50) -> dict:
    """Run `n_episodes` of evaluation and return task + safety metrics.

    The env should be wrapped in `SafetyMetricWrapper` so that each
    terminal step populates `info["safety_metrics"]`.
    """
    successes, rewards, ep_lens = [], [], []
    safety_acc: dict[str, list[float]] = {k: [] for k in _SAFETY_KEYS}

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False
        info: dict = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(episode_reward)
        successes.append(info.get("is_success", False))
        ep_lens.append(steps)

        sm = info.get("safety_metrics", {})
        for k in _SAFETY_KEYS:
            if k in sm:
                safety_acc[k].append(float(sm[k]))

    out = {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_ep_len": float(np.mean(ep_lens)) if ep_lens else 0.0,
    }
    for k, vals in safety_acc.items():
        out[k] = float(np.mean(vals)) if vals else 0.0

    # Back-compat aliases for older callers expecting these names.
    out["mean_action_mag"] = out["mean_action_norm"]
    out["mean_jerk"] = out["mean_delta_action_norm"]
    out["max_action_mag"] = out["max_action_norm"]

    return out
