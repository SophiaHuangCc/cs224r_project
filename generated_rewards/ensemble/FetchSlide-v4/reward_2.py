def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Compute Euclidean distance between achieved_goal and desired_goal
    distance = np.linalg.norm(achieved_goal - desired_goal)
    # Success threshold
    success_threshold = 0.05
    # Reward shaping: negative distance for progress
    reward = -distance
    # Success bonus if within threshold
    if distance < success_threshold:
        reward += 10.0
    return float(reward)