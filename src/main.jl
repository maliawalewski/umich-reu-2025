using Groebner, AbstractAlgebra, Statistics
include("environment.jl")
include("data.jl")

function main()
    num_vars = 3
    delta_noise = 0.1f0
    env = init_environment(numVars = num_vars, delta_noise = delta_noise)
    vars = fill_ideal(env, 5, 3, 10, 100)

    # Example action
    raw_action = [1.0f0, 2.0f0, 3.0f0]
    action = make_valid_action(raw_action, env)
    println("Action: ", action)

    if in_action_space(action, env)
        basis = act!(env, action)
        println("Basis: ")
        println(basis)
        println("is groebner basis? ")
        println(isgroebner(basis))
    else
        println("Action is out of bounds.")
    end

    println("Current state: ", state(env))
end

main()
