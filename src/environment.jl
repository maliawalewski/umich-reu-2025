using Groebner
using Random
using AbstractAlgebra
include("data.jl")
include("basesets.jl")

# Scaling parameters
ACTION_SCALE = 1e3 # Scales action to integers 
# n_cols_list = Float64[]
# pair_degrees = Int[]
# pair_counts = Int[]

mutable struct Environment{R<:AbstractRNG}
    rng::R
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
    grevlex_reward_cache::Vector{Float64}
    grevlex_time_cache_s::Vector{Float64}
    deglex_reward_cache::Vector{Float64}
    deglex_time_cache_s::Vector{Float64}
    last_agent_batch_time_s::Float64
    last_agent_mean_time_s::Float64
    last_raw_reward::Float64
    last_delta_reward::Float64
end

function init_environment(;
    args::Dict{String,Any} = nothing,
    num_ideals::Int = 10,
    delta_bound::Float32 = 0.01f0,
    max_iterations::Int = 5,
    default_num_vars = 3,
    default_num_terms = 5,
    default_num_polys = 3,
    rng::AbstractRNG = Random.default_rng(),
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
    else
        base_sets = nothing
    end

    if args["baseset"] == "DEFAULT"
        println(
            "Using default params: num_vars = $default_num_vars, num_terms = $default_num_terms, num_polys = $default_num_polys",
        )
        num_vars, num_terms, num_polys =
            default_num_vars, default_num_terms, default_num_polys
    else
        num_polys = length(base_sets)
        @assert num_polys > 0 "Empty base_sets"
        num_terms = length(base_sets[1])
        @assert all(length(p) == num_terms for p in base_sets) "polynomials have differing term counts"
        num_vars = length(base_sets[1][1])
        @assert all(length(t) == num_vars for p in base_sets for t in p) "terms do not all have the same number of variables"
    end

    monomial_matrix::Array{Vector{Any}} =
        Array{Vector{Any}}(undef, num_polys, num_vars * num_terms)

    ideal_batch::Vector{
        Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}},
    } = Vector{Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}}()

    vars::Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}} =
        Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}()

    iteration_count = 0
    is_terminated = false
    reward = 0.0

    grevlex_reward_cache = Float64[]
    grevlex_time_cache_s = Float64[]
    deglex_reward_cache = Float64[]
    deglex_time_cache_s = Float64[]
    last_agent_batch_time_s = 0.0
    last_agent_mean_time_s = 0.0
    last_raw_reward = 0.0
    last_delta_reward = 0.0

    state_i = init_state(rng, num_vars)
    return Environment(
        rng,
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
        grevlex_reward_cache,
        grevlex_time_cache_s,
        deglex_reward_cache,
        deglex_time_cache_s,
        last_agent_batch_time_s,
        last_agent_mean_time_s,
        last_raw_reward,
        last_delta_reward,
    )
end

function init_state(rng::AbstractRNG, num_vars::Int)
    epsilon_vector = 1 .+ rand(rng, Float32, num_vars)
    return epsilon_vector ./ sum(epsilon_vector)
end

function init_state_less_noise(rng::AbstractRNG, num_vars::Int)
    vector = fill(1.0f0 / Float32(num_vars), num_vars)
    epsilon_vector = vector .+ (rand(rng, Float32, num_vars) .* 0.01f0)
    return epsilon_vector ./ sum(epsilon_vector)
end

function fill_ideal_batch(
    env::Environment,
    num_polynomials::Int,
    max_degree::Int,
    max_attempts::Int,
)
    ideals, vars, monomial_matrix = new_generate_data(
        rng = env.rng,
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
    action = raw_action
    env.state = action

    action = Int.(round.(ACTION_SCALE * action))
    action = max.(action, 1)

    weights = zip(env.variables, action)
    order = WeightedOrdering(weights...)

    total_delta = 0.0
    total_raw = 0.0
    total_time_s = 0.0
    basis_vector = []

    env.last_raw_reward = 0.0
    env.last_delta_reward = use_baseline ? 0.0 : NaN

    if use_baseline
        @assert length(env.grevlex_reward_cache) == length(env.ideal_batch) "Grevlex cache missing: call precompute_baselines! after setting env.ideal_batch"
    end

    for i = 1:length(env.ideal_batch)
        ideal = env.ideal_batch[i]

        (tr, basis), t = timed(() -> groebner_learn(ideal, ordering = order))
        push!(basis_vector, basis)

        curr_reward = reward(tr)
        total_raw += curr_reward
        total_time_s += t

        if use_baseline
            delta_r = curr_reward - env.grevlex_reward_cache[i]
            total_delta += delta_r
            env.last_delta_reward += delta_r
        end
    end

    env.last_agent_batch_time_s = total_time_s
    env.last_agent_mean_time_s = total_time_s / length(env.ideal_batch)

    env.last_raw_reward = total_raw / length(env.ideal_batch)

    if use_baseline
        env.reward = total_delta / length(env.ideal_batch)
        env.last_delta_reward /= length(env.ideal_batch)
    else
        env.reward = env.last_raw_reward
    end

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
    env.reward = 0.0
    env.state = init_state(env.rng, env.num_vars)
    env.ideal_batch =
        Vector{Vector{AbstractAlgebra.Generic.MPoly{AbstractAlgebra.GFElem{Int64}}}}()
    env.is_terminated = false
    env.iteration_count = 0

    empty!(env.grevlex_reward_cache)
    empty!(env.grevlex_time_cache_s)
    empty!(env.deglex_reward_cache)
    empty!(env.deglex_time_cache_s)
    env.last_agent_batch_time_s = 0.0
    env.last_agent_mean_time_s = 0.0
    env.last_raw_reward = 0.0
    env.last_delta_reward = 0.0
end

timed(f) = begin
    t0 = time()
    v = f()
    v, (time() - t0)
end

function precompute_baselines!(env::Environment; compute_deglex::Bool = true)
    n = length(env.ideal_batch)
    resize!(env.grevlex_reward_cache, n)
    resize!(env.grevlex_time_cache_s, n)

    if compute_deglex
        resize!(env.deglex_reward_cache, n)
        resize!(env.deglex_time_cache_s, n)
    else
        empty!(env.deglex_reward_cache)
        empty!(env.deglex_time_cache_s)
    end

    for i = 1:n
        ideal = env.ideal_batch[i]

        (tr_g, _), tg = timed(() -> groebner_learn(ideal, ordering = DegRevLex()))
        env.grevlex_reward_cache[i] = reward(tr_g)
        env.grevlex_time_cache_s[i] = tg

        if compute_deglex
            (tr_d, _), td = timed(() -> groebner_learn(ideal, ordering = DegLex()))
            env.deglex_reward_cache[i] = reward(tr_d)
            env.deglex_time_cache_s[i] = td
        end
    end
end
