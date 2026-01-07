using AbstractAlgebra
using Serialization
using Statistics
using Groebner
using Plots

using Random
using DataFrames
using CSV

BASE_SET_PATH = "./rvt_baseset.bin"
OUT_CSV_PATH = "reward_vs_time.csv"
SEED = 1

Random.seed!(SEED)
@info "Using RNG seed = $SEED"

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

function reward(trace::Groebner.WrappedTrace)
    @assert length(trace.recorded_traces) == 1 "WrappedTrace struct is tracking multiple traces"
    total_reward = Float64(0.0)
    for (k, t) in trace.recorded_traces
        @assert length(t.critical_pair_sequence) == (length(t.matrix_infos) - 1) "length of critical_pair_sequence and matrix_infos do not match"
        for i = 1:length(t.critical_pair_sequence)
            n_cols = Float64(t.matrix_infos[i+1][3]) / Float64(100)
            pair_degree = t.critical_pair_sequence[i][1]
            pair_count = t.critical_pair_sequence[i][2]

            total_reward += (n_cols) * log(Float64(pair_degree)) * Float64(pair_count)
        end
    end
    return -total_reward
end

base_sets = isfile(BASE_SET_PATH) ? load_base_sets(BASE_SET_PATH) : nothing

ideals, vars, base_sets_used = new_generate_data(
    num_ideals = 1,
    num_polynomials = 3,
    num_variables = 3,
    max_degree = 5,
    num_terms = 6,
    max_attempts = 100,
    base_sets = base_sets,
    base_set_path = BASE_SET_PATH,
    should_save_base_sets = base_sets === nothing,
)

ideal = ideals[1]

println("ideal: $ideal")

grevlex_trace, _ = groebner_learn(ideal, ordering = DegRevLex())
baseline_reward = reward(grevlex_trace)

@info "Warming up compilation/caches..."
warm_actions = [(30, 30, 30), (50, 50, 50), (70, 70, 70)]
for a in warm_actions
    weights = zip(vars, a)
    order = WeightedOrdering(weights...)
    tr, _ = groebner_learn(ideal, ordering = order)
    _ = reward(tr)
end
GC.gc()
@info "Warmup done."

times = Float64[]
rewards = Float64[]

df = DataFrame(
    seed = Int[],
    i = Int[],
    j = Int[],
    k = Int[],
    time_s = Float64[],
    reward = Float64[],
    baseline_reward = Float64[],
    reward_delta = Float64[],
)

for i = 30:5:70, j = 30:5:70, k = 30:5:70
    action = (i, j, k)
    weights = zip(vars, action)
    order = WeightedOrdering(weights...)

    GC.gc()

    local tr::Groebner.WrappedTrace
    elapsed = @elapsed begin
        tr, _ = groebner_learn(ideal, ordering = order)
    end

    r = reward(tr)
    reward_delta = r - baseline_reward

    push!(times, elapsed)
    push!(rewards, reward_delta)

    push!(df, (SEED, i, j, k, elapsed, r, baseline_reward, reward_delta))
    @info "action=$action time=$(round(elapsed,digits=4))s, reward_delta=$(round(reward_delta,digits=3))"
end

CSV.write(OUT_CSV_PATH, df)
@info "Wrote CSV to $(abspath(OUT_CSV_PATH)) with $(nrow(df)) rows"

times_plot = times
rewards_plot = rewards

p = scatter(
    times_plot,
    rewards_plot,
    xlabel = "Computation time (s)",
    ylabel = "Reward (delta vs DegRevLex)",
    legend = false,
    markersize = 2,
    color = :green,
    markerstrokewidth = 0,
)

savefig(p, "reward_vs_time.pdf")
