using Groebner
using AbstractAlgebra
include("data.jl")
include("basesets.jl")

# Scaling parameters
ACTION_SCALE = 1e3 # Scales action to integers 
# n_cols_list = Float64[]
# pair_degrees = Int[]
# pair_counts = Int[]

mutable struct Environment
    num_vars::Int
    delta_bound::Float32
    state::Vector{Float32}
    reward::Float64
    ideal_batch::Vector{
        Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}},
    }
    variables::Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}
    is_terminated::Bool
    num_ideals::Int
    iteration_count::Int
    max_iterations::Int
    num_terms::Int
    num_polys::Int
    monomial_matrix::Array{Vector{Any}}
end


function init_environment(;
    args::Dict{String, Any} = nothing,
    num_ideals::Int = 10,
    delta_bound::Float32 = 0.01f0,
    max_iterations::Int = 5,
    default_num_vars = 3, 
    default_num_terms = 5, 
    default_num_polys = 3,
)
    @assert delta_bound >= 0.0f0 "Delta noise must be non-negative."

    if args["baseset"] == "N_SITE_PHOSPHORYLATION_BASE_SET"
        base_sets = N_SITE_PHOSPHORYLATION_BASE_SET
    elseif args["baseset"] == "RELATIVE_POSE_BASE_SET"
        base_sets = RELATIVE_POSE_BASE_SET
    elseif args["baseset"] == "TRIANGULATION_BASE_SET"
        base_sets = TRIANGULATION_BASE_SET
    elseif args["baseset"] == "WNT_BASE_SET"
        base_sets = WNT_BASE_SET
    elseif args["baseset"] == "FOUR_PT_BASE_SET"
        base_sets = FOUR_PT_BASE_SET
    else
        base_sets = nothing
    end

    if args["baseset"] == "DEFAULT"
        println("Using default params: num_vars = $default_num_vars, num_terms = $default_num_terms, num_polys = $default_num_polys")
        num_vars, num_terms, num_polys = default_num_vars, default_num_terms, default_num_polys
    else 
        num_polys = length(base_sets)
        @assert num_polys > 0 "Empty base_sets"
        num_terms = length(base_sets[1])
        @assert all(length(p) == num_terms for p in base_sets) "polynomials have differing term counts"
        num_vars = length(base_sets[1][1])
        @assert all(length(t) == num_vars for p in base_sets for t in p) "terms do not all have the same number of variables"
    end

    monomial_matrix::Array{Vector{Any}} = Array{Vector{Any}}(
        undef,
        num_polys,
        num_vars * num_terms,
    )   

    ideal_batch::Vector{
        Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}},
    } = Vector{Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}}()
    
    vars::Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}} = Vector{
        AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}},
    }()
    
    iteration_count = 0
    is_terminated = false
    reward = 0.0f0

    state_i = init_state(num_vars, num_polys)
    return Environment(
        num_vars,
        delta_bound,
        state_i,
        reward,
        ideal_batch,
        vars,
        is_terminated,
        num_ideals,
        iteration_count,
        max_iterations,
        num_terms,
        num_polys,
        monomial_matrix,
    ) # Added
end

function init_state(num_vars::Int, num_polys::Int)
    epsilon_vector = 1 .+ rand(Float32, num_vars)
    return epsilon_vector ./ sum(epsilon_vector)  # Normalize to ensure it sums to 1
end

function init_state_less_noise(num_vars::Int)
    vector = zeros(num_vars)
    vector .+= 1.0f0 / Float32(num_vars)
    epsilon_vector = vector .+ (rand(Float32, num_vars) .* 0.01f0)
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

    # display(monomial_matrix)

    env.ideal_batch = ideals
    env.variables = vars
    env.monomial_matrix = monomial_matrix
end

function act!(env::Environment, raw_action::Vector{Float32}, use_baseline::Bool)
    # action = make_valid_action_new(env, env.state, raw_action) # Make valid action from NN output and previous state
    action = raw_action
    env.state = action

    action = Int.(round.(ACTION_SCALE * action))
    action = max.(action, 1)
    # println("weighting for groebner: ", action)

    weights = zip(env.variables, action)
    order = WeightedOrdering(weights...)

    total_reward = Float64(0.0f0)
    basis_vector = []

    for i = 1:length(env.ideal_batch)

        ideal = env.ideal_batch[i]
        trace, basis = groebner_learn(ideal, ordering = order)
        basis_vector = push!(basis_vector, basis)
        curr_reward = reward(trace)

        if use_baseline
            baseline_trace, baseline_basis = groebner_learn(ideal, ordering = DegRevLex())
            total_reward += (curr_reward - reward(baseline_trace))
        else 
            total_reward += curr_reward
        end
    end

    env.reward = total_reward / Float64(length(env.ideal_batch))
    env.iteration_count += 1
    if env.iteration_count >= env.max_iterations
        env.is_terminated = true
    end

    return basis_vector
end

function make_valid_action(
    env::Environment,
    state::Vector{Float32},
    raw_action::Vector{Float32},
)
    # Takes the output of the NN and makes it a valid action
    min_allowed = max.(state .- env.delta_bound, Float32(1 / env.num_vars^3))
    max_allowed = state .+ env.delta_bound
    clamped_action = clamp.(raw_action, min_allowed, max_allowed)

    action = clamped_action ./ sum(clamped_action)  # Normalize action to ensure it sums to 1

    return action
end

function make_valid_action_new(
    env::Environment,
    state::Vector{Float32},
    raw_action::Vector{Float32},
)
    # New valid action implemented with delta bound shift
    action = state .* (1 - env.delta_bound) + raw_action .* (env.delta_bound)
    action = max.(action, Float32(1 / env.num_vars^3)) # Ensure non-negative
    action = action ./ sum(action) # Normalize to ensure it sums to 1

    return action
end

function make_valid_action_test(
    env::Environment,
    state::Vector{Float32},
    raw_action::Vector{Float32},
)
    # Test valid action implementation
    action = max.(raw_action, Float32(1 / env.num_vars^3)) # Ensure non-negative
    
    return action
end


function in_action_space(action::Vector{Float32}, env::Environment)
    # Checks if action is a valid state and that it is not moving too far from the current state
    return in_state_space(action, env) && all(abs.(action .- env.state) .<= env.delta_bound)
end

function reward(trace::Groebner.WrappedTrace)
    @assert length(trace.recorded_traces) == 1 "WrappedTrace struct is tracking multiple traces"
    total_reward = Float64(0.0f0)
    for (k, t) in trace.recorded_traces
        @assert length(t.critical_pair_sequence) == (length(t.matrix_infos) - 1) "length of critical_pair_sequence and matrix_infos do not match"
        for i = 1:length(t.critical_pair_sequence)
            n_cols = Float64(t.matrix_infos[i+1][3]) / Float64(100)
            pair_degree = t.critical_pair_sequence[i][1]
            pair_count = t.critical_pair_sequence[i][2]

            # push!(n_cols_list, n_cols)
            # push!(pair_degrees, pair_degree)
            # push!(pair_counts, pair_count)

            total_reward +=
                (Float64(n_cols)) * Float64(log(pair_degree)) * (Float64(pair_count))
        end
    end
    return -total_reward
end

function in_state_space(x::Vector{Float32}, env::Environment)
    # Checks if vector is within state space 
    @assert length(x) == env.num_vars "State vector must have the same number of variables as the environment."
    return all(x .>= 0.0f0) && all(x .<= 1.0f0) # all elements are non-negative and less than or equal to 1
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
    env.state = init_state(env.num_vars, env.num_polys)
    env.ideal_batch =
        Vector{Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}}()
    env.is_terminated = false
    env.iteration_count = 0 # Added
end
