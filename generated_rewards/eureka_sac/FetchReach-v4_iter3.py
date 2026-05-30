def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Calculate the Euclidean distance between the achieved goal and the desired goal
    distance = np.linalg.norm(achieved_goal - desired_goal)
    # Define thresholds for success and shaping rewards
    success_threshold = 0.05  # Success if within 5 cm
    shaping_threshold = 0.5  # Intermediate milestone within 50 cm
    # Success bonus
    if distance <= success_threshold:
        return 5.0  # Increased success reward for faster convergence
    # Intermediate shaping reward
    elif distance <= shaping_threshold:
        return max(1.0 - 3 * distance, -1.0)  # Stronger shaping reward with capped penalty
    # Default shaping reward for being far away
    else:
        return -3 * distance  # Increased penalty for being far away