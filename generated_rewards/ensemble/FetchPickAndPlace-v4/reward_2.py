def compute_reward(achieved_goal, desired_goal, obs, action, info):
    # Distance to the goal
    distance_to_goal = np.linalg.norm(achieved_goal - desired_goal)
    
    # Intermediate milestones
    block_grasped = obs[-1]  # Assuming the last element in `obs` represents grip status (1 if grasped, 0 otherwise)
    block_lifted = achieved_goal[2] > 0.05  # Check if block is lifted above a certain height
    
    # Reward shaping
    reward = -distance_to_goal  # Penalize distance to goal
    
    # Success bonus
    if distance_to_goal < 0.05:  # Success threshold
        reward += 10.0  # Large bonus for success
    
    # Bonuses for intermediate milestones
    if block_grasped:
        reward += 2.0  # Bonus for successfully grasping the block
    if block_lifted:
        reward += 1.0  # Bonus for lifting the block
    
    return reward