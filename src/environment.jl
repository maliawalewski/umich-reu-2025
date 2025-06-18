# action_space(env::YourEnv)
# state(env::YourEnv)
# state_space(env::YourEnv)
# reward(env::YourEnv)
# is_terminated(env::YourEnv)
# reset!(env::YourEnv)
# act!(env::YourEnv, action)

using Groebner
include("data.jl")

struct Environment 
    numVars::Int
    delta_noise::Float32
    state::Vector{Float32}
    reward::Float32
    ideal::Vector{Any}
end

function Environment(numVars::Int, delta_noise::Float32)
    @assert numVars > 0 "Number of variables must be greater than 0."
    @assert delta_noise >= 0f0 "Delta noise must be non-negative."

    state = zeros(Float32, numVars)
    # state /= sum(state)  # Normalize to ensure it sums to 1
    return Environment(numVars, delta_noise, state, 0f0, [])
end

function fill_ideal(env::Environment, num_polynomials::Int, max_degree::Int, max_terms::Int)
    env.ideal = generate_ideal(
        num_polynomials=num_polynomials,
        num_variables=env.numVars,
        max_degree=max_degree,
        max_terms=max_terms
    )
end

function act!(env::Environment, action::Vector{Float32})
    # @assert action in_action_space(env) "Action must be within the action space of the environment."

    weight_vector = env.state .+ action
    order = WeightedOrdering(weight_vector)
    
    trace, basis = groebner_learn(env.ideal, ordering=order)
    env.state = weight_vector
    env.reward = reward(trace)
    return basis
end

function reward(trace::Groebner.WrappedTrace)
    @assert length(trace.recorded_traces) == 1 "WrappedTrace struct is tracking multiple traces"
    total_reward = 0
    for (k, t) in trace.recorded_traces
        @assert length(t.critical_pair_sequence) == (length(t.matrix_infos) - 1) "length of critical_pair_sequence and matrix_infos do not match"
        for i in 1:length(t.critical_pair_sequence)
            n_cols = t.matrix_infos[i + 1][3]
            pair_degree = t.critical_pair_sequence[i][1]
            pair_count = t.critical_pair_sequence[i][2]
            total_reward += (n_cols * pair_count * log(pair_degree))
        end
    end
    return -total_reward
end

#checks if vector is within state space 
function in_state_space(x::Vector{Float32}, env::Environment)
    @assert length(x) == env.numVars "State vector must have the same number of variables as the environment."
    return all(x .>= 0f0) && abs(sum(x) - 1f0) < 1e-6
end

function state(env::Environment)
     return env.state
end

function in_action_space(action::Vector{Float32}, env::Environment)
    return in_state_space(action .+ env.state, env) && norm(action) <= env.delta_noise
end

# function is_terminated(env::Environment)

# end


