def compute_reward(achieved_goal, desired_goal, obs, action, info):
    import numpy as np
    
    # Calculate the Euclidean distance between the achieved goal and the desired goal.
    distance = np.linalg.norm(achieved_goal - desired_goal)
    
    # Success threshold
    success_threshold = 0.05
    
    # Reward shaping based on distance
    reward = -5 * distance  # Reduce penalty to make shaping more forgiving.
    
    # Success bonus
    if distance < success_threshold:
        reward = 100.0  # Large positive reward for achieving the goal.
        return reward  # Early exit for success.
    
    # Optional intermediate milestones: Encourage faster convergence
    intermediate_thresholds = [0.5, 0.2, 0.1]
    bonus_values = [10.0, 15.0, 25.0]
    
    for threshold, bonus in zip(intermediate_thresholds, bonus_values):
        if distance < threshold:
            reward += bonus
    
    # Penalize episode length indirectly by penalizing actions
    action_penalty = -0.01 * np.sum(np.square(action))  # Discourage large, inefficient actions.
    reward += action_penalty
    
    return reward