lives_tot = 3
function custom_rewards()
    local lives = data.lives
    local score_reward = calculate_score_reward(data)
    local movement_reward = calculate_movement_reward(data)
    local reward = score_reward + movement_reward
    if lives < lives_tot then
        lives_tot = lives
	reward = reward - 500
    end
    return reward
end

vert_dist_tot = 0
horiz_dist_tot = 0 
function calculate_movement_reward(data)
    local movement_reward = 0
    local horiz_dist_current = data.horiz_dist
    local vert_dist_current = data.vert_dist
    if vert_dist_current < 1000 then
        movement_reward = movement_reward + vert_dist_current - vert_dist_tot
        vert_dist_tot = vert_dist_current
    end
    movement_reward = movement_reward + horiz_dist_current - horiz_dist_tot
    horiz_dist_tot = horiz_dist_current
    return movement_reward
end

score_tot = 0
function calculate_score_reward(data)
    local score_current = data.score
    if score_current > score_tot then
        local score_reward = score_current - score_tot
        score_tot = score_current
	return score_reward
    else
	return 0
    end
end
