using Groebner
using AbstractAlgebra
include("data.jl")

mutable struct Environment
    numVars::Int
    delta_noise::Float32
    state::Vector{Float32}
    reward::Float32
    ideal_batch::Vector{Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}}
    variables::Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}
    is_terminated::Bool
    max_ideals::Int
end

function init_environment(;
    numVars::Int = 1,
    delta_noise::Float32 = 0.001f0,
    reward::Float32 = 0.0f0,
    ideal_batch::Vector{Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}} = Vector{Vector{
        AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}},
    }}(),
    vars::Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}} = Vector{
        AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}},
    }(),
    is_terminated::Bool = false,
    max_ideals::Int = 20,
)
    @assert numVars > 0 "Number of variables must be greater than 0."
    @assert delta_noise >= 0.0f0 "Delta noise must be non-negative."

    state_i = init_state(numVars)
    return Environment(numVars, delta_noise, state_i, reward, ideal_batch, vars, is_terminated, max_ideals)
end

function init_state(numVars::Int)
    epsilon_vector = 1 .+ rand(0.0f0:1.0f0, numVars)
    return epsilon_vector ./ sum(epsilon_vector)  # Normalize to ensure it sums to 1
end

function fill_ideal_batch(
    env::Environment,
    num_polynomials::Int,
    max_degree::Int,
    max_terms::Int,
    max_attempts::Int,
)
    ideals, vars = generate_data(
        num_ideals = env.max_ideals,
        num_polynomials = num_polynomials,
        num_variables = env.numVars,
        max_degree = max_degree,
        num_terms = max_terms,
        max_attempts = max_attempts,
    )
    env.ideal_batch = ideals
    env.variables = vars
end

function act!(env::Environment, action::Vector{Int})   
    @assert in_action_space(action, env) "Action must be a valid state and close enough to current state."

    action = [1, 2, 3]
    weights = zip(env.variables, action)
    order = WeightedOrdering(weights...)

    cur_reward = 0.0f0
    total_reward = 0.0f0
    basis_vector = Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}()

    for i in 1:length(env.ideal_batch)
        ideal = env.ideal_batch[i]
        trace, basis = groebner_learn(ideal, ordering = order)

        basis_vector = vcat(basis_vector, basis)
        cur_reward = reward(trace)
        total_reward += cur_reward
    end

    env.reward = total_reward / length(env.ideal_batch)
    env.state = action

    return basis_vector
end

function make_valid_action(env::Environment, raw_action::Vector{Float32})
    # Takes the output of the NN and makes it a valid action
    min_allowed = max.(env.state .- env.delta_noise, 0.0f0)
    max_allowed = env.state .+ env.delta_noise
    clamped_action = clamp.(raw_action, min_allowed, max_allowed)
    println("Clamped action: ", clamped_action)

    action = clamped_action ./ sum(clamped_action)  # Normalize action to ensure it sums to 1
    return action
end

function in_action_space(action::Vector{Float32}, env::Environment)
    # Checks if action is a valid state and that it is not moving too far from the current state
    return in_state_space(action, env) && all(abs.(action .- env.state) .<= env.delta_noise)
end

function reward(trace::Groebner.WrappedTrace)
    @assert length(trace.recorded_traces) == 1 "WrappedTrace struct is tracking multiple traces"
    total_reward = 0
    for (k, t) in trace.recorded_traces
        @assert length(t.critical_pair_sequence) == (length(t.matrix_infos) - 1) "length of critical_pair_sequence and matrix_infos do not match"
        for i = 1:length(t.critical_pair_sequence)
            n_cols = t.matrix_infos[i+1][3]
            pair_degree = t.critical_pair_sequence[i][1]
            pair_count = t.critical_pair_sequence[i][2]
            total_reward += (n_cols * pair_count * log(pair_degree))
        end
    end
    return -total_reward
end

function in_state_space(x::Vector{Float32}, env::Environment)
    # Checks if vector is within state space 
    @assert length(x) == env.numVars "State vector must have the same number of variables as the environment."
    return all(x .>= 0.0f0) && abs(sum(x) - 1.0f0) < 1e-6 # vector sums to 1 and all elements are non-negative
end

function state(env::Environment)
    return env.state
end

function is_terminated(env::Environment)
    # Terminates when basis has been computed
    return env.is_terminated
end

function reset!(env::Environment)
    # Resets the environment to its initial state
    env.state = init_state(env.numVars)
    env.reward = 0.0f0
    env.ideal_batch = []
    env.is_terminated = false

    fill_ideal_batch(
    env, 3, 3, 5, 10,
    )
end