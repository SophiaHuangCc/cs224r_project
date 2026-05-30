def compute_reward(achieved_goal, desired_goal, obs, action, info):
    distance = np.linalg.norm(achieved_goal - desired_goal)
    success_threshold = 0.05
    if distance < success_threshold:
        return 1.0  # Success bonus
    return -distance  # Distance-based shaping reward