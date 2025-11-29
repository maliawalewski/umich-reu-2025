using Statistics
using SymbolicUtils
using Latexify
using AbstractAlgebra
using Serialization
using Statistics
using Groebner
using SymbolicRegression
import SymbolicRegression: MultitargetSRRegressor, node_to_symbolic
import MLJ: machine, fit!, predict, report

include("ideal_ordering_pairs.jl")

function monomial_matrix_shape(mm)
    @assert !isempty(mm) "monomial_matrix is empty"
    n_polys = length(mm)
    n_mons = length(mm[1])
    @assert all(length(poly) == n_mons for poly in mm) "inconsistent number of monomials across polys"
    nvars = length(mm[1][1])
    @assert all(length(m) == nvars for poly in mm for m in poly) "inconsistent number of variables across monomials"
    return n_polys, n_mons, nvars
end

function flatten_monomial_matrix(mm)::Vector{Float64}
    n_polys, n_mons, nvars = monomial_matrix_shape(mm)
    v = Vector{Float64}(undef, n_polys * n_mons * nvars)
    k = 1
    @inbounds for p = 1:n_polys
        poly = mm[p]
        for m = 1:n_mons
            mon = poly[m]
            for j = 1:nvars
                v[k] = Float64(mon[j])
                k += 1
            end
        end
    end
    return v
end

function build_sr_dataset(samples)
    n_polys_ref, n_mons_ref, nvars_ref = monomial_matrix_shape(samples[1].monomial_matrix)
    featlen = n_polys_ref * n_mons_ref * nvars_ref
    ns = length(samples)

    X = Matrix{Float64}(undef, ns, featlen)
    W = Matrix{Float64}(undef, ns, nvars_ref)

    for (i, s) in enumerate(samples)
        n_polys_i, n_mons_i, nvars_i = monomial_matrix_shape(s.monomial_matrix)
        @assert (n_polys_i, n_mons_i, nvars_i) == (n_polys_ref, n_mons_ref, nvars_ref) "shape mismatch in sample $i"

        x_flat = flatten_monomial_matrix(s.monomial_matrix)
        @assert length(x_flat) == featlen
        X[i, :] = x_flat

        w = Float64.(s.order)
        @assert length(w) == nvars_ref "weight length $(length(w)) must equal nvars=$nvars."
        ssum = sum(w)
        @assert ssum > 0 "weights must have positive sum."
        W[i, :] = w ./ ssum
    end

    return X, W
end

function simplex_aware_sr(X, Y)
    println("\n=== Simplex-aware 2-target SR for each pair ===")

    pairs = [(1, 2), (1, 3), (2, 3)]
    best_pair_mse = Inf
    best_pair = nothing
    best_pair_data = nothing
    best_pair_mach = nothing

    for (i, j) in pairs
        all_idx = (1, 2, 3)
        k = filter(x -> x != i && x != j, all_idx)[1]

        println("\n--- Pair (w[$i], w[$j]) -> derive w[$k] ---")

        Y_pair = Y[:, [i, j]]
        model_pair = make_model()
        mach_pair = machine(model_pair, X, Y_pair)
        fit!(mach_pair)

        r_pair = report(mach_pair)

        best_i = r_pair.equations[1][r_pair.best_idx[1]]
        best_j = r_pair.equations[2][r_pair.best_idx[2]]

        sym_i = node_to_symbolic(best_i)
        sym_j = node_to_symbolic(best_j)
        sym_k = 1 - sym_i - sym_j

        println("w[$i](x) expression: ", best_i)
        println("w[$j](x) expression: ", best_j)
        println("w[$k](x) derived: 1 - w[$i](x) - w[$j](x)")
        println("Symbolic w[$i]: ", sym_i)
        println("Symbolic w[$j]: ", sym_j)
        println("Symbolic w[$k]: ", sym_k)
        println("LaTeX w[$i]: ", latexify(string(sym_i)))
        println("LaTeX w[$j]: ", latexify(string(sym_j)))
        println("LaTeX w[$k]: ", latexify(string(sym_k)))

        y_pred_pair = predict(mach_pair, X)
        @assert size(y_pred_pair, 2) == 2

        Yhat = similar(Y)
        Yhat[:, i] = y_pred_pair[:, 1]
        Yhat[:, j] = y_pred_pair[:, 2]
        Yhat[:, k] = 1 .- Yhat[:, i] .- Yhat[:, j]

        Yhat_clamped = max.(Yhat, 0.0)
        row_sums = sum(Yhat_clamped, dims = 2)
        row_sums[row_sums .== 0.0] .= 1.0
        Yhat_norm = Yhat_clamped ./ row_sums

        pair_mse = mse(Y, Yhat_norm)
        println("MSE(SR model, pair ($i,$j) with simplex) = ", pair_mse)

        if pair_mse < best_pair_mse
            best_pair_mse = pair_mse
            best_pair = (i, j, k)
            best_pair_data = (
                best_i = best_i,
                best_j = best_j,
                sym_i = sym_i,
                sym_j = sym_j,
                sym_k = sym_k,
            )
            best_pair_mach = mach_pair
        end
    end

    println("\n=== Best pair (simplex-aware) ===")
    println("Best (i, j -> k) = ", best_pair, " with MSE = ", best_pair_mse)

    if best_pair_data !== nothing
        (i, j, k) = best_pair
        println("\nBest pair symbolic equations:")
        println("w[$i](x): ", best_pair_data.sym_i)
        println("w[$j](x): ", best_pair_data.sym_j)
        println("w[$k](x): ", best_pair_data.sym_k)
    end

    return best_pair_mse, best_pair, best_pair_data, best_pair_mach
end

function predict_weights_from_best_pair(
    mach_pair,
    best_pair::NTuple{3,Int},
    monomial_matrix;
    scale::Real = 100,
)
    (i, j, k) = best_pair

    x_flat = flatten_monomial_matrix(monomial_matrix)
    X_new = reshape(x_flat, 1, :)

    y_pair = predict(mach_pair, X_new)
    @assert size(y_pair, 2) == 2

    w_raw = zeros(Float64, 3)
    w_raw[i] = y_pair[1, 1]
    w_raw[j] = y_pair[1, 2]
    w_raw[k] = 1 - w_raw[i] - w_raw[j]

    w_clamped = max.(w_raw, 0.0)
    s = sum(w_clamped)
    w_norm = s == 0 ? fill(1/3, 3) : (w_clamped ./ s)

    w_int = round.(Int, w_norm .* scale)
    w_int[w_int .<= 0] .= 1

    return w_int
end

function new_generate_ideal(;
    num_variables::Integer = 3,
    num_polynomials::Integer = 3,
    num_terms::Integer = 3,
    base_sets::AbstractVector = Any[],
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
    base_sets::Union{Nothing,AbstractVector} = nothing,
    base_set_path::Union{Nothing,String} = nothing,
    should_save_base_sets::Bool = false,
)
    @assert num_ideals > 0 "num_ideals must be greater than 0"

    if base_sets === nothing
        base_sets = Any[]
        for _ = 1:num_polynomials
            used_exponents = Set{NTuple{num_variables,Int}}()
            base_set = Any[]
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

    ideals = Any[]
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

function act_vs_grevlex(
    action::AbstractVector{<:Integer},
    vars::AbstractVector{<:AbstractAlgebra.Generic.MPoly},
    ideal::AbstractVector{<:AbstractAlgebra.Generic.MPoly},
)
    @assert length(action) == length(vars) "length(action) must equal number of variables"

    weights = zip(vars, action)
    order = WeightedOrdering(weights...)

    trace_sr, _ = groebner_learn(ideal, ordering = order)
    trace_grev, _ = groebner_learn(ideal, ordering = DegRevLex())

    return reward(trace_sr) - reward(trace_grev)
end

function evaluate_sr_bestpair_vs_grevlex(
    best_pair_mach,
    best_pair;
    M::Integer = 1000,
    max_attempts::Integer = 100,
)
    mm_template = SAMPLES[1].monomial_matrix
    n_polys, n_terms, nvars = monomial_matrix_shape(mm_template)

    @assert nvars == size(Y, 2) "SR was trained for $(size(Y,2)) weights, \
but monomial_matrix has $nvars variables."



    ideals_test, vars, base_set = new_generate_data(
        num_ideals = M,
        num_polynomials = n_polys,
        num_variables = nvars,
        max_degree = 4,
        num_terms = n_terms,
        max_attempts = max_attempts,
        base_sets = nothing,
        base_set_path = nothing,
        should_save_base_sets = false,
    )

    println("Baseset: ")
    println(base_set)

    w_int = predict_weights_from_best_pair(best_pair_mach, best_pair, mm_template)
    println("Weight Vector $w_int")

    rewards = Float64[]
    for (idx, ideal) in enumerate(ideals_test)
        if idx % 100 == 0
            println("evaluating ideal $idx / $M")
        end

        delta_r = act_vs_grevlex(w_int, vars, ideal)
        if idx % 100 == 0
            println("current reward: $delta_r")
        end
        push!(rewards, delta_r)
    end

    avg_reward = mean(rewards)
    min_reward = minimum(rewards)
    max_reward = maximum(rewards)

    println("\nsymbolic regression best-pair ordering vs grevlex on $M random ideals")
    println("Average delta_reward (symbolic - grevlex): $avg_reward")
    println("Minimum delta_reward:                $min_reward")
    println("Maximum delta_reward:                $max_reward")

    return rewards
end



function mse(A, B)
    @assert size(A) == size(B)
    return mean((A .- B) .^ 2)
end

square(x) = x^2
cube(x) = x^3

function make_model()
    return MultitargetSRRegressor(
        niterations = 300,
        populations = 100,
        binary_operators = [+, -, *, /],
        unary_operators = [square, cube],
        maxsize = 20,
        parsimony = 1e-3,
        parallelism = :multithreading,
    )
end

X, Y = build_sr_dataset(SAMPLES)
println("X dims: $(size(X)), Y dims: $(size(Y))")
@assert size(Y, 2) == 3 "Expecting exactly 3 weights/targets."

ns = size(Y, 1)

mean_vec = mean(Y, dims = 1)
Y_mean = repeat(mean_vec, ns, 1)
println("\nMSE(constant mean predictor) = ", mse(Y, Y_mean))

println("\n=== Full 3-target SR (no simplex constraint enforced) ===")

model_full = make_model()
mach_full = machine(model_full, X, Y)
fit!(mach_full)

r_full = report(mach_full)
n_targets = size(Y, 2)

for j = 1:n_targets
    best_expr = r_full.equations[j][r_full.best_idx[j]]
    println("\nTarget w[$j] (direct):")
    println("Expression: ", best_expr)

    eqn = node_to_symbolic(best_expr)
    println("Symbolic form: ", eqn)
    println("LaTeX: ", latexify(string(eqn)))
end

y_pred_full = predict(mach_full, X)
@assert size(y_pred_full) == size(Y)
println("MSE(SR model, 3 targets) = ", mse(Y, y_pred_full))


best_pair_mse, best_pair, best_pair_data, best_pair_mach = simplex_aware_sr(X, Y)

rewards_sr_vs_grev = evaluate_sr_bestpair_vs_grevlex(best_pair_mach, best_pair; M = 10_000)
