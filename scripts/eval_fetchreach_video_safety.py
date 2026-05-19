import os
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import imageio
from stable_baselines3 import PPO

gym.register_envs(gymnasium_robotics)

os.makedirs("results/videos", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

MODEL_PATH = "models/ppo_fetchreach_baseline"
VIDEO_PATH = "results/videos/fetchreach_baseline.mp4"
METRICS_PATH = "results/metrics/fetchreach_baseline_metrics.txt"

NUM_EPISODES = 10
MAX_STEPS = 50

# Conservative workspace bound for FetchReach gripper position.
# You can tune this later after inspecting achieved_goal values.
WORKSPACE_LOW = np.array([1.0, 0.3, 0.3])
WORKSPACE_HIGH = np.array([1.6, 1.1, 0.9])

env = gym.make("FetchReach-v4", render_mode="rgb_array")
model = PPO.load(MODEL_PATH)

all_episode_metrics = []
video_frames = []

for ep in range(NUM_EPISODES):
    obs, info = env.reset()

    prev_action = np.zeros(env.action_space.shape)
    prev_pos = np.array(obs["achieved_goal"])

    ep_rewards = []
    action_norms = []
    delta_action_norms = []
    gripper_speeds = []
    workspace_violations = []
    successes = []

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if ep == 0:
            video_frames.append(frame)

        curr_pos = np.array(obs["achieved_goal"])

        action_norm = np.linalg.norm(action)
        delta_action_norm = np.linalg.norm(action - prev_action)
        gripper_speed = np.linalg.norm(curr_pos - prev_pos)

        workspace_violation = np.any(curr_pos < WORKSPACE_LOW) or np.any(curr_pos > WORKSPACE_HIGH)
        success = float(info.get("is_success", 0.0))

        ep_rewards.append(float(reward))
        action_norms.append(action_norm)
        delta_action_norms.append(delta_action_norm)
        gripper_speeds.append(gripper_speed)
        workspace_violations.append(float(workspace_violation))
        successes.append(success)

        prev_action = action
        prev_pos = curr_pos

        if terminated or truncated:
            break

    ep_metric = {
        "episode": ep,
        "total_reward": float(np.sum(ep_rewards)),
        "success": float(np.max(successes)),
        "mean_action_norm": float(np.mean(action_norms)),
        "max_action_norm": float(np.max(action_norms)),
        "mean_delta_action_norm": float(np.mean(delta_action_norms)),
        "max_delta_action_norm": float(np.max(delta_action_norms)),
        "mean_gripper_speed": float(np.mean(gripper_speeds)),
        "max_gripper_speed": float(np.max(gripper_speeds)),
        "workspace_violation_rate": float(np.mean(workspace_violations)),
    }

    all_episode_metrics.append(ep_metric)

env.close()

# Save video from first episode
imageio.mimsave(VIDEO_PATH, video_frames, fps=30)

# Aggregate metrics
summary = {}
keys = all_episode_metrics[0].keys()
for key in keys:
    if key != "episode":
        summary[key] = np.mean([m[key] for m in all_episode_metrics])

with open(METRICS_PATH, "w") as f:
    f.write("FetchReach Baseline Evaluation\n")
    f.write("===============================\n\n")

    f.write("Per-episode metrics:\n")
    for m in all_episode_metrics:
        f.write(str(m) + "\n")

    f.write("\nAverage metrics:\n")
    for k, v in summary.items():
        f.write(f"{k}: {v:.4f}\n")

print(f"Saved video to {VIDEO_PATH}")
print(f"Saved metrics to {METRICS_PATH}")
print("Average metrics:")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")