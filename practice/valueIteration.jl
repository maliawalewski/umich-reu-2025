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

# Define actions: (row movement, column movement)
const actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # down, up, right, left


function value_iteration(V::Array{Float64, 2}, theta::Float64)
    while true 
        delta = 0.0
        for i in 1:x
            for j in 1:y 

                if (i, j) in terminals
                    continue # skip terminal states
                end

                v = V[i, j] # current value 
                V[i, j] = # Bellman eqn -- write new function to solve? 
                delta = max(delta, abs(v - V[i, j])) # update delta
            end 
        end
        if delta < theta
            break 
        end 
    end 
end 