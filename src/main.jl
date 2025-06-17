using Groebner, AbstractAlgebra, Statistics
include("environment.jl")
include("data.jl")

function main() 
    num_vars = 3
    delta_noise = 1f0
    env = Environment(num_vars, delta_noise)

    # Example action
    action = [1f0, 2f0, 3f0]
    action /= sum(action)  # Normalize action to ensure it sums to 1
    println("Action: ", action)

    if in_action_space(action, env)
        act!(env, action)
    else
        println("Action is out of bounds.")
    end

    println("Current state: ", state(env))
end

# main()