# test_maniskill_modal.py

import modal

app = modal.App("cs224r-maniskill-test")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libegl1",
        "libgles2",
        "libvulkan1",
        "vulkan-tools",
        "mesa-vulkan-drivers",
    )
    .pip_install(
        "torch",
        "mani_skill",
        "gymnasium",
        "numpy",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
)
def test_maniskill():
    import torch
    import gymnasium as gym
    import mani_skill.envs

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    env = gym.make(
        "PushCube-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
        num_envs=16,
        sim_backend="gpu",
    )

    obs, info = env.reset()
    print("Observation type:", type(obs))
    print("Action space:", env.action_space)

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward mean = {reward.float().mean().item():.4f}")

    env.close()
    print("ManiSkill GPU smoke test finished successfully.")


@app.local_entrypoint()
def main():
    test_maniskill.remote()