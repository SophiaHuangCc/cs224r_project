def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Compute the distance between the achieved goal and the desired goal
    distance = np.linalg.norm(achieved_goal - desired_goal)
    # Define success threshold
    success_threshold = 0.05  # Distance within which the task is considered successful
    # Reward shaping based on distance
    if distance <= success_threshold:
        return 10.0  # Success bonus
    else:
        return -distance * 5.0  # Stronger penalty for being far from the goal