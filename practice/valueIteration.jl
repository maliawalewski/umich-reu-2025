################################################################################
# Gridworld Value Iteration (4x4)
# --------------------------------
# This script implements value iteration on a 4x4 gridworld to compute the
# optimal state-value function `V(s)` and the corresponding optimal policy.
#
# Terminal states: (1, 1) and (4, 4)
# Rewards: -1 per step, 0 for terminals
################################################################################


# Gridworld configuration
const x = 4
const y = 4
const theta = 0.1 # can change 
const gamma = 0.9 # can change 
const terminals = [(1, 1), (4, 4)]

# Initial value function
V = zeros(Float64, x, y)
V[1, 1] = 0.0  # top-left terminal
V[4, 4] = 0.0  # bottom-right terminal

# Initialize Grid
grid = -ones(Float64, x, y)
grid[1, 1] = 0.0
grid[4, 4] = 0.0

# Define actions: (row movement, column movement)
const actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # down, up, right, left

function get_actions(state::Tuple{Int64, Int64})
    # returns all valid actions from a given state
    row, col = state 
    possible_actions = Tuple{Int64, Int64}[]
    for action in actions
        new_row, new_col = row + action[1], col + action[2]
        if new_row < 1 || new_row > x || new_col < 1 || new_col > y 
            continue
        end
        push!(possible_actions, action)
    end
    return possible_actions
end

function get_reward(state::Tuple{Int64, Int64}, grid::Array{Float64, 2})
    # returns the reward associated with a given state
    return grid[state[1], state[2]]
end

function get_transitions(state::Tuple{Int64, Int64}, action::Tuple{Int64, Int64})
    new_state = (state[1] + action[1], state[2] + action[2])
    return [(1.0, new_state)] # assuming deterministic transitions for now
end

function value_update(V::Array{Float64, 2}, state::Tuple{Int64, Int64}, grid::Array{Float64, 2}, gamma::Float64)
    # updates the value according to the bellman equations
    best_value = -Inf
    for action in get_actions(state)
        total = 0.0
        for (prob, new_state) in get_transitions(state, action)
            reward = get_reward(new_state, grid)
            total += prob * (reward + gamma * V[new_state[1], new_state[2]])
        end
        best_value = max(best_value, total)
    end
    return best_value
end

function value_iteration!(V::Array{Float64, 2}, grid::Array{Float64, 2}, theta::Float64, gamma::Float64)
    while true 
        delta = 0.0
        for i in 1:x
            for j in 1:y 
                if (i, j) in terminals
                    continue # skip terminal states
                end

                v = V[i, j] # current value 
                V[i, j] = value_update(V, (i, j), grid, gamma) # bellman update
                delta = max(delta, abs(v - V[i, j])) # update delta
            end 
        end
        if delta < theta
            break 
        end 
    end 
end 

value_iteration!(V, grid, theta, gamma)

println("Final Values: ")
println(V)
