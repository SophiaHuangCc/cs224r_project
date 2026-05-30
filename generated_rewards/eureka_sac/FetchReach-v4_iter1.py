def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Calculate the Euclidean distance between the achieved goal and the desired goal
    distance = np.linalg.norm(achieved_goal - desired_goal)
    # Define thresholds for success and shaping rewards
    success_threshold = 0.05  # Success if within 5 cm
    shaping_threshold = 0.5  # Intermediate milestone within 50 cm
    # Success bonus
    if distance <= success_threshold:
        return 1.0
    # Intermediate shaping reward
    elif distance <= shaping_threshold:
        return 0.5 - distance  # Smaller distance gives higher reward
    # Default shaping reward for being far away
    else:
        return -distance