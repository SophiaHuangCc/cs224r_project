def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Compute the Euclidean distance between achieved and desired goals
    distance = np.linalg.norm(achieved_goal - desired_goal)
    
    # Define a threshold for considering the task complete
    threshold = 0.05
    
    # Give a high reward for being within the threshold, and penalize based on distance otherwise
    if distance <= threshold:
        reward = 0.0  # Task completed
    else:
        reward = -distance  # Penalize based on distance
    
    return reward