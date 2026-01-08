using AbstractAlgebra
using Serialization
using Statistics
using Groebner

# triangulation
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

# relative pose paper
BASE_SET = Vector{Any}([
    [
        [3, 0, 2],
        [3, 0, 1],
        [3, 0, 0],
        [2, 1, 2],
        [1, 2, 2],
        [0, 3, 2],
        [2, 1, 1],
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 0],
        [1, 1, 2],
        [2, 0, 2],
        [0, 2, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [1, 1, 1],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [0, 0, 3],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [3, 0, 2],
        [3, 0, 1],
        [3, 0, 0],
        [2, 1, 2],
        [1, 2, 2],
        [0, 3, 2],
        [2, 1, 1],
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 0],
        [1, 1, 2],
        [2, 0, 2],
        [0, 2, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [1, 1, 1],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [0, 0, 3],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [3, 0, 2],
        [3, 0, 1],
        [3, 0, 0],
        [2, 1, 2],
        [1, 2, 2],
        [0, 3, 2],
        [2, 1, 1],
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 0],
        [1, 1, 2],
        [2, 0, 2],
        [0, 2, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [1, 1, 1],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [0, 0, 3],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [3, 0, 2],
        [3, 0, 1],
        [3, 0, 0],
        [2, 1, 2],
        [1, 2, 2],
        [0, 3, 2],
        [2, 1, 1],
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 0],
        [1, 1, 2],
        [2, 0, 2],
        [0, 2, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [1, 1, 1],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [0, 0, 3],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [3, 0, 2],
        [3, 0, 1],
        [3, 0, 0],
        [2, 1, 2],
        [1, 2, 2],
        [0, 3, 2],
        [2, 1, 1],
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 0],
        [1, 1, 2],
        [2, 0, 2],
        [0, 2, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [1, 1, 1],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [0, 0, 3],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [3, 0, 2],
        [3, 0, 1],
        [3, 0, 0],
        [2, 1, 2],
        [1, 2, 2],
        [0, 3, 2],
        [2, 1, 1],
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 0],
        [1, 1, 2],
        [2, 0, 2],
        [0, 2, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [1, 1, 1],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [0, 0, 3],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [3, 0, 2],
        [3, 0, 1],
        [3, 0, 0],
        [2, 1, 2],
        [1, 2, 2],
        [0, 3, 2],
        [2, 1, 1],
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 0],
        [1, 1, 2],
        [2, 0, 2],
        [0, 2, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [1, 1, 1],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [0, 0, 3],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [3, 0, 2],
        [3, 0, 1],
        [3, 0, 0],
        [2, 1, 2],
        [1, 2, 2],
        [0, 3, 2],
        [2, 1, 1],
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 0],
        [1, 1, 2],
        [2, 0, 2],
        [0, 2, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [1, 1, 1],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [0, 0, 3],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [3, 0, 2],
        [3, 0, 1],
        [3, 0, 0],
        [2, 1, 2],
        [1, 2, 2],
        [0, 3, 2],
        [2, 1, 1],
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 0],
        [1, 1, 2],
        [2, 0, 2],
        [0, 2, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [1, 1, 1],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [0, 0, 3],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [3, 0, 2],
        [3, 0, 1],
        [3, 0, 0],
        [2, 1, 2],
        [1, 2, 2],
        [0, 3, 2],
        [2, 1, 1],
        [1, 2, 1],
        [0, 3, 1],
        [2, 1, 0],
        [1, 1, 2],
        [2, 0, 2],
        [0, 2, 2],
        [1, 0, 3],
        [1, 2, 0],
        [0, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [1, 1, 1],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [0, 0, 3],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
])


BASE_SET = Vector{Any}([
    [
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [1, 6],
        [1, 7],
        [1, 8],
        [1, 9],
        [1, 10],
        [1, 11],
        [1, 12],
        [1, 13],
        [1, 14],
    ],
    [
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [1, 6],
        [1, 7],
        [1, 8],
        [1, 9],
        [1, 10],
        [1, 11],
        [1, 12],
        [1, 13],
        [1, 14],
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
            num_polynomials = num_polynomials,
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
    grlex_trace, _ = groebner_learn(ideal, ordering = DegLex())
    # grevlex_trace, _ = groebner_learn(ideal, ordering = DegRevLex())
    return (reward(trace) - reward(grlex_trace))
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

BASE_SET_PATH = "./test_baseset.bin"

base_sets = BASE_SET

println(base_sets)

base_sets = isfile(BASE_SET_PATH) ? load_base_sets(BASE_SET_PATH) : nothing

println(base_sets)

#=
ideals, vars, monomial_matrix = new_generate_data(
    num_ideals = 1,
    num_polynomials = 2,
    num_variables = 2,
    max_degree = 15,
    num_terms = 17,
    max_attempts = 100,
    base_sets = base_sets,
    base_set_path = BASE_SET_PATH,
    should_save_base_sets = base_sets === nothing,
)

=#

ideals, vars, monomial_matrix = new_generate_data(
    num_ideals = 1,
    num_polynomials = 2,   # matches length(BASE_SET)
    num_variables = 2,
    max_degree = 15,
    num_terms = 17,        # matches length(BASE_SET[1])
    max_attempts = 100,
    base_sets = BASE_SET,  # already hardcoded above
    base_set_path = BASE_SET_PATH,
    should_save_base_sets = base_sets === nothing,
)



reward_map = Dict{NTuple{3,Int},Float64}()

baseline_trace, _ = groebner_learn(ideals[1], ordering = DegRevLex())
println("grevlex: $(reward(baseline_trace))")

baseline_trace, _ = groebner_learn(ideals[1], ordering = DegLex())
println("grlex: $(reward(baseline_trace))")

# dot_products = [(mon, mon[1]*29 + mon[2]*43 + mon[3]*27) for mon in BASE_SET[1]]
# sorted = sort(dot_products, by = x -> -x[2])
# println(sorted)

#=
for i = 53:53
    for j = 37:37
        for k = 69:69
            order = (i, j, k)
            r = act([i, j, k], vars, ideals[1])
            println("i: $i, j: $j, k: $k, reward: $r")
            reward_map[order] = r
        end
    end
end

=#

reward_map = Dict{NTuple{2,Int},Float64}()

for i = 1:100
    for j = 1:100
        order = (i, j)
        r = act([i, j], vars, ideals[1])
        println("i: $i, j: $j, reward: $r")
        reward_map[order] = r
    end
end



# (53, 37, 69)

best_order = argmax(reward_map)
best_reward = reward_map[best_order]

worst_order = argmin(reward_map)
worst_reward = reward_map[worst_order]

println("\nBest reward:  $best_reward  at order = $best_order")
println("Worst reward: $worst_reward at order = $worst_order")

base_sets = isfile(BASE_SET_PATH) ? load_base_sets(BASE_SET_PATH) : nothing

M = 100_000

#=
ideals_test, _, _ = new_generate_data(
    num_ideals = M,
    num_polynomials = 3,
    num_variables = 3,
    max_degree = 2,
    num_terms = 5,
    base_sets = base_sets,
    base_set_path = BASE_SET_PATH,
    should_save_base_sets = base_sets === nothing,
)
=#

ideals_test, _, _ = new_generate_data(
    num_ideals = M,
    num_polynomials = 2,   # matches length(BASE_SET)
    num_variables = 2,
    max_degree = 15,
    num_terms = 17,        # matches length(BASE_SET[1])
    max_attempts = 100,
    base_sets = BASE_SET,  # already hardcoded above
    base_set_path = BASE_SET_PATH,
    should_save_base_sets = base_sets === nothing,
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
