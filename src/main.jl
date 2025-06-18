using Groebner, AbstractAlgebra, Statistics
include("environment.jl")
include("data.jl")

function main()
    num_vars = 3
    delta_noise = 0.1f0
    env = init_environment(numVars = num_vars, delta_noise = delta_noise)
    vars = fill_ideal(env, 5, 3, 10, 100)

    println("Initial state: ", state(env))
    println("Ideal: ")
    for p in env.ideal
        println("Polynomial: ", p)
    end

    # Example action
<<<<<<< Updated upstream
    raw_action = [1.0f0, 2.0f0, 3.0f0]
    action = make_valid_action(raw_action, env)
=======
    action = [0.01f0, 0.02f0, 0.03f0]
    action .-= mean(action)  # Normalize action to ensure it sums to 0
>>>>>>> Stashed changes
    println("Action: ", action)

    if in_action_space(action, env)
        act!(env, action, vars)
    else
        println("Action is out of bounds.")
    end

    println("Current state: ", state(env))
end

main()
