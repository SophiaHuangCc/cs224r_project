"""Render baseline rollout videos for all three Fetch tasks.

Loads each SAC+HER reference policy (sparse-reward, 500k steps) from
models/reference/ and writes deterministic rollouts to results/videos/:

    results/videos/fetchreach_baseline.mp4
    results/videos/fetchpick_baseline.mp4
    results/videos/fetchslide_baseline.mp4

Run:  conda run -n cs224r_project python scripts/make_baseline_videos.py
"""
from __future__ import annotations
import os
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import imageio
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)

NUM_EPISODES = 5
MAX_STEPS = 50
FPS = 25

# (env_id, model path stem, output filename)
TASKS = [
    ("FetchReach-v4",        "models/reference/FetchReach-v4",        "fetchreach_baseline.mp4"),
    ("FetchPickAndPlace-v4", "models/reference/FetchPickAndPlace-v4", "fetchpick_baseline.mp4"),
    ("FetchSlide-v4",        "models/reference/FetchSlide-v4",        "fetchslide_baseline.mp4"),
]

VIDEO_DIR = "results/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)


def render(env_id: str, model_path: str, out_name: str) -> None:
    env = gym.make(env_id, render_mode="rgb_array")
    model = SAC.load(model_path, env=env)

    frames = []
    successes = 0
    for _ in range(NUM_EPISODES):
        obs, _ = env.reset()
        info = {}
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        successes += int(info.get("is_success", 0.0))

    env.close()

    out_path = os.path.join(VIDEO_DIR, out_name)
    imageio.mimsave(out_path, frames, fps=FPS)
    print(f"wrote {out_path}  ({len(frames)} frames, {successes}/{NUM_EPISODES} successes)")


def main() -> None:
    for env_id, model_path, out_name in TASKS:
        render(env_id, model_path, out_name)


if __name__ == "__main__":
    main()
