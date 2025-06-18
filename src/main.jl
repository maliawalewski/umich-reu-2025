using Groebner, AbstractAlgebra, Statistics
include("environment.jl")
include("data.jl")

function main()
    num_vars = 3
    delta_noise = 1.0f0
    env = init_environment(numVars = num_vars, delta_noise = delta_noise)
    fill_ideal(env, 5, 3, 10)

    println("Initial state: ", state(env))
    println("Ideal: ", env.ideal)

    # Example action
    action = [1.0f0, 2.0f0, 3.0f0]
    action .-= mean(action)  # Normalize action to ensure it sums to 0
    println("Action: ", action)

    if in_action_space(action, env)
        act!(env, action)
    else
        println("Action is out of bounds.")
    end

    println("Current state: ", state(env))
end

main()
