def compute_reward(achieved_goal, desired_goal, obs, action, info):
    distance = np.linalg.norm(achieved_goal - desired_goal)
    if distance < 0.05:  # Success threshold
        return 1.0  # Success bonus
    return -distance  # Distance-based shaping reward