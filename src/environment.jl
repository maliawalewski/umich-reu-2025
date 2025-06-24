using Groebner
using AbstractAlgebra
include("data.jl")

# Scaling parameters
ACTION_SCALE = 1e4 # Scales action to integers

mutable struct Environment
    num_vars::Int
    delta_bound::Float32
    state::Vector{Float32}
    reward::Float64
    ideal_batch::Vector{Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}}
    variables::Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}
    is_terminated::Bool
    num_ideals::Int
    iteration_count::Int # Added
    max_iterations::Int # Added
    num_terms::Int
    num_polys::Int
    monomial_matrix::Array{Vector{Any}}
end

function init_environment(;
    num_vars::Int = 1,
    delta_bound::Float32 = 0.01f0,
    reward::Float32 = 0.0f0,
    ideal_batch::Vector{Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}} = Vector{Vector{
        AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}
    }}(),
    vars::Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}} = Vector{
        AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}
    }(),
    is_terminated::Bool = false,
    num_ideals::Int = 10,
    iteration_count::Int = 0, # Added
    max_iterations::Int = 5, # Added
    num_terms::Int = num_vars + 3,
    num_polys::Int = num_vars,
    monomial_matrix::Array{Vector{Any}} = Array{Vector{Any}}(undef, num_polys, num_vars * num_terms),
)
    @assert num_vars > 0 "Number of variables must be greater than 0."
    @assert delta_bound >= 0.0f0 "Delta noise must be non-negative."

    state_i = init_state(num_vars)
    return Environment(num_vars, delta_bound, state_i, reward, ideal_batch, vars, is_terminated, num_ideals, iteration_count, max_iterations, num_terms, num_polys, monomial_matrix) # Added
end

function init_state(num_vars::Int)
    epsilon_vector = 1 .+ rand(Float32, num_vars)
    return epsilon_vector ./ sum(epsilon_vector)  # Normalize to ensure it sums to 1
end

function fill_ideal_batch(
    env::Environment,
    num_polynomials::Int,
    max_degree::Int,
    max_attempts::Int,
)
    ideals, vars, monomial_matrix = new_generate_data(
        num_ideals = env.num_ideals,
        num_polynomials = num_polynomials,
        num_variables = env.num_vars,
        max_degree = max_degree,
        num_terms = env.num_terms,
        max_attempts = max_attempts,
    )
    env.ideal_batch = ideals
    env.variables = vars
    env.monomial_matrix = monomial_matrix
end

function act!(env::Environment, action::Vector{Float32})   
    action = make_valid_action(env, action)
    
    env.state = action

    action = Int.(round.(ACTION_SCALE * action))

    weights = zip(env.variables, action)
    order = WeightedOrdering(weights...)

    cur_reward = Float64(0.0f0)
    total_reward = Float64(0.0f0)
    basis_vector = []

    for i in 1:length(env.ideal_batch)
        ideal = env.ideal_batch[i]
        trace, basis = groebner_learn(ideal, ordering = order)
        baseline_trace, baseline_basis = groebner_learn(ideal, ordering = DegRevLex())
        
        basis_vector = push!(basis_vector, basis)
        total_reward += (reward(trace) - reward(baseline_trace))
    end

    env.reward = total_reward / Float64(length(env.ideal_batch))
    env.iteration_count += 1 # Added
    if env.iteration_count >= env.max_iterations # Added
        env.is_terminated = true # Added
    end    # Added

    return basis_vector
end

function make_valid_action(env::Environment, raw_action::Vector{Float32})
    # Takes the output of the NN and makes it a valid action
    raw_action = raw_action .+ rand(Float32, env.num_vars)*(0.2f0) # Add noise to the action
    min_allowed = max.(env.state .- env.delta_bound, Float32(1 / env.num_vars^3))
    max_allowed = env.state .+ env.delta_bound
    clamped_action = clamp.(raw_action, min_allowed, max_allowed)

    action = clamped_action ./ sum(clamped_action)  # Normalize action to ensure it sums to 1

    return action
end

function in_action_space(action::Vector{Float32}, env::Environment)
    # Checks if action is a valid state and that it is not moving too far from the current state
    return in_state_space(action, env) && all(abs.(action .- env.state) .<= env.delta_bound)
end

function reward(trace::Groebner.WrappedTrace)
    @assert length(trace.recorded_traces) == 1 "WrappedTrace struct is tracking multiple traces"
    total_reward = Float64(0f0)
    for (k, t) in trace.recorded_traces
        @assert length(t.critical_pair_sequence) == (length(t.matrix_infos) - 1) "length of critical_pair_sequence and matrix_infos do not match"
        for i = 1:length(t.critical_pair_sequence)
            n_cols = Float64(t.matrix_infos[i+1][3]) / Float64(100)
            pair_degree = t.critical_pair_sequence[i][1]
            pair_count = t.critical_pair_sequence[i][2]
            total_reward += (Float64(n_cols) * Float64(pair_count) * Float64(log(pair_degree)))
        end
    end
    return -total_reward
end

function in_state_space(x::Vector{Float32}, env::Environment)
    # Checks if vector is within state space 
    @assert length(x) == env.num_vars "State vector must have the same number of variables as the environment."
    return all(x .>= 0.0f0) && abs(sum(x) - 1.0f0) < 1e-6 # vector sums to 1 and all elements are non-negative
end

function state(env::Environment)
    return env.state
end

function is_terminated(env::Environment)
    # Terminates when basis has been computed
    return env.is_terminated
end

function reset_env!(env::Environment)
    # Resets the environment to its initial state
    env.reward = Float64(0.0f0)
    env.ideal_batch = Vector{Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}}()
    env.is_terminated = false
    env.iteration_count = 0 # Added
end
