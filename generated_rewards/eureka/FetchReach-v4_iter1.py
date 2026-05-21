def compute_reward(achieved_goal, desired_goal, obs, action, info):
    import numpy as np
    
    # Calculate the Euclidean distance between the achieved goal and the desired goal.
    distance = np.linalg.norm(achieved_goal - desired_goal)
    
    # Success threshold
    success_threshold = 0.05
    
    # Reward shaping based on distance
    reward = -distance  # Penalize the robot for being far away from the target.
    
    # Success bonus
    if distance < success_threshold:
        reward += 10.0  # Large bonus for achieving the goal.
    
    # Optional intermediate milestones: Encourage smaller distances more strongly
    intermediate_thresholds = [0.5, 0.2, 0.1]
    bonus_values = [1.0, 2.0, 4.0]
    
    for threshold, bonus in zip(intermediate_thresholds, bonus_values):
        if distance < threshold:
            reward += bonus
    
    return reward