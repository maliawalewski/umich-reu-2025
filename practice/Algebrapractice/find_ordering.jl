using AbstractAlgebra
using Serialization
using Statistics
using Groebner

BASE_SET = Vector{Any}([
    [
        [4, 2, 0],
        [4, 0, 2],
        [3, 3, 0],
        [3, 2, 1],
        [3, 2, 0],
        [3, 1, 2],
        [3, 0, 3],
        [3, 0, 2],
        [1, 3, 2],
        [1, 2, 3],
        [1, 2, 2],
        [0, 4, 2],
        [0, 3, 3],
        [0, 3, 2],
        [0, 2, 4],
        [0, 2, 3],
        [0, 2, 2],
    ],
    [
        [4, 0, 2],
        [3, 3, 0],
        [3, 1, 2],
        [3, 0, 3],
        [3, 0, 2],
        [2, 4, 0],
        [2, 3, 1],
        [2, 3, 0],
        [2, 1, 3],
        [2, 1, 2],
        [2, 0, 4],
        [2, 0, 3],
        [2, 0, 2],
        [1, 3, 2],
        [0, 4, 2],
        [0, 3, 3],
        [0, 3, 2],
    ],
    [
        [4, 2, 0],
        [3, 3, 0],
        [3, 2, 1],
        [3, 2, 0],
        [3, 0, 3],
        [2, 4, 0],
        [2, 3, 1],
        [2, 3, 0],
        [2, 2, 1],
        [2, 2, 0],
        [2, 1, 3],
        [2, 0, 4],
        [2, 0, 3],
        [1, 2, 3],
        [0, 3, 3],
        [0, 2, 4],
        [0, 2, 3],
    ],
])

function new_generate_ideal(;
    num_variables::Integer = 3,
    num_polynomials::Integer = 3,
    num_terms::Integer = 3,
    base_sets::Vector{Any} = Vector{Any}(),
    max_attempts::Integer = 100,
)

    @assert num_variables > 0 "num_variables must be greater than 0"
    @assert length(base_sets) == num_polynomials "number of base_sets does not match the number of polynomials"
    @assert length(base_sets[1]) == num_terms "number of exponents in base_set does not match the number of terms"

    field = GF(32003)
    ring, vars = polynomial_ring(field, ["x_" * string(i) for i = 1:num_variables])
    polynomials = Vector{typeof(vars[1])}()

    for base_set in base_sets
        terms = []
        for e in base_set
            coeff = rand(field)
            c_attempts = 0
            while coeff == 0
                coeff = rand(field)
                c_attempts += 1
                @assert c_attempts <= max_attempts "failed to generate a non-zero coefficient after $max_attempts attempts"
            end
            monomial = coeff * prod(vars[i]^e[i] for i = 1:num_variables)
            push!(terms, monomial)
        end
        polynomial = sum(terms)
        push!(polynomials, polynomial)
    end

    return polynomials, vars
end

function new_generate_data(;
    num_ideals::Integer = 1000,
    num_polynomials::Integer = 3,
    num_variables::Integer = 3,
    max_degree::Integer = 4,
    num_terms::Integer = 3,
    max_attempts::Integer = 100,
    base_sets::Union{Nothing,Vector{Any}} = nothing,
    base_set_path::Union{Nothing,String} = nothing,
    should_save_base_sets::Bool = false,
)

    @assert num_ideals > 0 "num_ideals must be greater than 0"

    if base_sets === nothing
        base_sets = []
        for _ = 1:num_polynomials
            used_exponents = Set{NTuple{num_variables,Int}}()
            base_set = []
            for _ = 1:num_terms
                attempts = 0
                while true
                    exponents = rand(0:max_degree, num_variables)
                    expt_key = Tuple(exponents)
                    if !(expt_key in used_exponents)
                        push!(used_exponents, expt_key)
                        push!(base_set, exponents)
                        break
                    end
                    attempts += 1
                    @assert attempts <= max_attempts "failed to generate a unique monomial after $max_attempts attempts"
                end
            end
            push!(base_sets, base_set)
        end
        if should_save_base_sets && base_set_path !== nothing
            save_base_sets(base_sets, base_set_path)
        end
    end

    ideals = []
    variables = nothing
    for _ = 1:num_ideals
        ideal, vars = new_generate_ideal(
            num_variables = num_variables,
            num_terms = num_terms,
            base_sets = base_sets,
            max_attempts = max_attempts,
        )
        variables = vars
        push!(ideals, ideal)
    end

    return ideals, variables, base_sets
end

function save_base_sets(base_sets, path::String)
    serialize(path, base_sets)
end

function load_base_sets(path::String)
    return deserialize(path)
end

function act(
    action::AbstractVector{<:Integer},
    vars::AbstractVector{<:AbstractAlgebra.Generic.MPoly},
    ideal::AbstractVector{<:AbstractAlgebra.Generic.MPoly},
)
    weights = zip(vars, action)
    order = WeightedOrdering(weights...)
    trace, _ = groebner_learn(ideal, ordering = order)
    baseline_trace, _ = groebner_learn(ideal, ordering = DegRevLex())
    return reward(trace) - reward(baseline_trace)
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

            total_reward +=
                (Float64(n_cols)) * Float64(log(pair_degree)) * (Float64(pair_count))
        end
    end
    return -total_reward
end

base_sets = BASE_SET

ideals, vars, monomial_matrix = new_generate_data(
    num_ideals = 1,
    num_polynomials = 3,
    num_variables = 3,
    max_degree = 6,
    num_terms = 17,
    max_attempts = 100,
    base_sets = base_sets,
    base_set_path = ".",
    should_save_base_sets = base_sets === nothing,
)


reward_map = Dict{NTuple{3,Int},Float64}()

for i = 30:70
    for j = 30:70
        for k = 30:70
            order = (i, j, k)
            r = act([i, j, k], vars, ideals[1])
            println("i: $i, j: $j, k: $k, reward: $r")
            reward_map[order] = r
        end
    end
end

best_order = argmax(reward_map)
best_reward = reward_map[best_order]

worst_order = argmin(reward_map)
worst_reward = reward_map[worst_order]

println("\nBest reward:  $best_reward  at order = $best_order")
println("Worst reward: $worst_reward at order = $worst_order")

M = 10_000
ideals_test, _, _ = new_generate_data(
    num_ideals = M,
    num_polynomials = 3,
    num_variables = 3,
    max_degree = 6,
    num_terms = 17,
    base_sets = BASE_SET,
)

wvec = collect(best_order)

test_rewards = Float64[]
c = 0
for (idx, ideal) in enumerate(ideals_test)
    if idx % 100 == 0
        println("idx: $idx")
    end
    r = act(wvec, vars, ideal)
    push!(test_rewards, r)
end

avg_reward = mean(test_rewards)
min_reward = minimum(test_rewards)
max_reward = maximum(test_rewards)

println("\nEvaluation on $M new ideals with best_order: $best_order")
println("Average reward: $avg_reward")
println("Minimum reward: $min_reward")
println("Maximum reward: $max_reward")
