def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Calculate the distance between achieved_goal and desired_goal
    distance = np.linalg.norm(achieved_goal - desired_goal)
    
    # Reward is negative distance, encouraging the puck to move closer to the target
    reward = -distance
    
    # Add a small penalty for excessive actions to encourage efficiency
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty
    
    return reward