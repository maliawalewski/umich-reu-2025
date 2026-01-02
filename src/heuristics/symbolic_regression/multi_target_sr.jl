using Statistics
using SymbolicUtils
using Latexify
using AbstractAlgebra
using Serialization
using Statistics
using Groebner
using BSON
using ArgParse
using SymbolicRegression
import SymbolicRegression: MultitargetSRRegressor, node_to_symbolic
import MLJ: machine, fit!, predict, report

include("../ideal_ordering_pairs.jl")
include("../../data.jl")

BASE_DIR = @__DIR__
WEIGHTS_DIR = joinpath(BASE_DIR, "weights")
RESULTS_DIR = joinpath(BASE_DIR, "results")
SYMBOLIC_OUTPUTS_DIR = joinpath(BASE_DIR, "symbolic_outputs")

for d in (WEIGHTS_DIR, RESULTS_DIR, SYMBOLIC_OUTPUTS_DIR)
    isdir(d) || mkpath(d)
end

SR_MODEL_PATH = joinpath(WEIGHTS_DIR, "sr_model.bin")
SR_EQNS_PATH = joinpath(RESULTS_DIR, "sr_equations.bson")
SR_EQNS_TXT_PATH = joinpath(RESULTS_DIR, "sr_equations.txt")

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
        @assert length(w) == nvars_ref "weight length $(length(w)) must equal nvars=$nvars_ref."

        ssum = sum(w)
        @assert ssum > 0 "weights must have positive sum."
        W[i, :] = w ./ ssum
    end

    return X, W
end

function save_equations_text(
    path::AbstractString,
    sr_mode::AbstractString,
    sym_eqns::NTuple{3,Any},
)
    open(path, "w") do io
        println(io, "Symbolic regression mode: $sr_mode")
        println(io)

        for j = 1:3
            eq = sym_eqns[j]
            println(io, "w[$j](x): ", eq)
            println(io, "LaTeX w[$j]: ", latexify(string(eq)))
            println(io)
        end
    end
    println("Saved SR equations text to $path")
end

function two_target_sr(X, Y)
    println("\nStarting 2-target SR for each pair.")

    pairs = [(1, 2), (1, 3), (2, 3)]
    best_pair_mse = Inf
    best_pair = nothing
    best_pair_data = nothing
    best_pair_mach = nothing

    for (i, j) in pairs
        all_idx = (1, 2, 3)
        k = filter(x -> x != i && x != j, all_idx)[1]

        println("\nPair (w[$i], w[$j]) -> derive w[$k]")

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

    println("\nBest pair (2-target SR)")
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

function three_target_sr(X, Y)
    println("\nFull 3-target SR (no simplex constraint enforced)")

    model_full = make_model()
    mach_full = machine(model_full, X, Y)
    fit!(mach_full)

    r_full = report(mach_full)
    n_targets = size(Y, 2)

    sym_eqns = Vector{Any}(undef, n_targets)
    best_nodes = Vector{Any}(undef, n_targets)

    for j = 1:n_targets
        best_expr = r_full.equations[j][r_full.best_idx[j]]
        best_nodes[j] = best_expr

        println("\nTarget w[$j] (direct):")
        println("Expression: ", best_expr)

        eqn = node_to_symbolic(best_expr)
        sym_eqns[j] = eqn
        println("Symbolic form: ", eqn)
        println("LaTeX: ", latexify(string(eqn)))
    end

    y_pred_full = predict(mach_full, X)
    @assert size(y_pred_full) == size(Y)
    println("MSE(SR model, 3 targets) = ", mse(Y, y_pred_full))

    return mach_full, r_full, Tuple(sym_eqns), Tuple(best_nodes)
end

function predict_weights_from_best_pair(
    mach_pair,
    best_pair::NTuple{3,Int},
    monomial_matrix;
    scale::Real = 1000,
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

function predict_weights_from_full_model(mach_full, monomial_matrix; scale::Real = 1000)
    x_flat = flatten_monomial_matrix(monomial_matrix)
    X_new = reshape(x_flat, 1, :)

    y_full = predict(mach_full, X_new)
    @assert size(y_full, 2) == 3

    w_raw = vec(y_full)

    w_clamped = max.(w_raw, 0.0)
    s = sum(w_clamped)
    w_norm = s == 0 ? fill(1/3, 3) : (w_clamped ./ s)

    w_int = round.(Int, w_norm .* scale)
    w_int[w_int .<= 0] .= 1

    return w_int
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

    sr_reward = reward(trace_sr)
    grev_reward = reward(trace_grev)
    return sr_reward, grev_reward, sr_reward - grev_reward
end

function compute_stats(agent_rewards, baseline_rewards, name)
    total = length(agent_rewards)
    @assert total == length(baseline_rewards) "agent and baseline lengths differ"

    wins = [a > b for (a, b) in zip(agent_rewards, baseline_rewards)]
    losses = [a < b for (a, b) in zip(agent_rewards, baseline_rewards)]
    ties = [a == b for (a, b) in zip(agent_rewards, baseline_rewards)]

    win_pct = 100 * count(wins) / total
    loss_pct = 100 * count(losses) / total
    tie_pct = 100 * count(ties) / total

    improvements = [
        (a - b) / max(abs(b), 1e-8) * 100 for
        (a, b) in zip(agent_rewards, baseline_rewards) if a > b
    ]
    degradations = [
        (b - a) / max(abs(b), 1e-8) * 100 for
        (a, b) in zip(agent_rewards, baseline_rewards) if a < b
    ]

    avg_improvement = isempty(improvements) ? 0.0 : mean(improvements)
    avg_degradation = isempty(degradations) ? 0.0 : mean(degradations)

    println("agent vs $name")
    println("percent of time agent wins:  $(round(win_pct,  digits = 2)) percent")
    println("percent of time agent loses: $(round(loss_pct, digits = 2)) percent")
    println("percent of time tie:         $(round(tie_pct,  digits = 2)) percent")
    println(
        "average improvement percent (when winning):   $(round(avg_improvement, digits = 2)) percent",
    )
    println(
        "average degradation percent (when losing):    $(round(avg_degradation, digits = 2)) percent",
    )
    println()
end

function evaluate_sr_bestpair_vs_grevlex(
    best_pair_mach,
    best_pair,
    Y;
    M::Integer = 1000,
    max_attempts::Integer = 100,
)
    mm_template = SAMPLES[1].monomial_matrix
    n_polys, n_terms, nvars = monomial_matrix_shape(mm_template)

    @assert nvars == size(Y, 2) "SR was trained for $(size(Y,2)) weights, but monomial_matrix has $nvars variables."

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

    w_int = predict_weights_from_best_pair(best_pair_mach, best_pair, base_set)
    println("Weight Vector $w_int")

    rewards_sr = Float64[]
    rewards_grev = Float64[]
    delta_rewards = Float64[]

    for (idx, ideal) in enumerate(ideals_test)
        if idx % 100 == 0
            println("evaluating coefficients: $idx / $M")
        end

        sr_reward, grev_reward, delta_reward = act_vs_grevlex(w_int, vars, ideal)
        if idx % 100 == 0
            println("(SR_reward - grevlex_reward) at $idx: $delta_reward")
        end

        push!(rewards_sr, sr_reward)
        push!(rewards_grev, grev_reward)
        push!(delta_rewards, delta_reward)
    end

    avg_delta = mean(delta_rewards)
    min_delta = minimum(delta_rewards)
    max_delta = maximum(delta_rewards)

    println("\nsymbolic regression best-pair ordering vs grevlex on $M random ideals")
    println("Average delta_reward (symbolic - grevlex): $avg_delta")
    println("Minimum delta_reward:                      $min_delta")
    println("Maximum delta_reward:                      $max_delta")

    compute_stats(rewards_sr, rewards_grev, "DegRevLex")

    return delta_rewards
end

function evaluate_sr_full_vs_grevlex(
    mach_full,
    Y;
    M::Integer = 1000,
    max_attempts::Integer = 100,
)
    mm_template = SAMPLES[1].monomial_matrix
    n_polys, n_terms, nvars = monomial_matrix_shape(mm_template)

    @assert nvars == size(Y, 2) "SR was trained for $(size(Y,2)) weights, but monomial_matrix has $nvars variables."

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

    w_int = predict_weights_from_full_model(mach_full, base_set)
    println("Weight Vector $w_int")

    rewards_sr = Float64[]
    rewards_grev = Float64[]
    delta_rewards = Float64[]

    for (idx, ideal) in enumerate(ideals_test)
        if idx % 100 == 0
            println("evaluating coefficients: $idx / $M")
        end

        sr_reward, grev_reward, delta_reward = act_vs_grevlex(w_int, vars, ideal)
        if idx % 100 == 0
            println("(SR_reward - grevlex_reward) at $idx: $delta_reward")
        end

        push!(rewards_sr, sr_reward)
        push!(rewards_grev, grev_reward)
        push!(delta_rewards, delta_reward)
    end

    avg_delta = mean(delta_rewards)
    min_delta = minimum(delta_rewards)
    max_delta = maximum(delta_rewards)

    println("\nsymbolic regression full 3-target ordering vs grevlex on $M random ideals")
    println("Average delta_reward (symbolic - grevlex): $avg_delta")
    println("Minimum delta_reward:                      $min_delta")
    println("Maximum delta_reward:                      $max_delta")

    compute_stats(rewards_sr, rewards_grev, "DegRevLex")

    return delta_rewards
end

function mse(A, B)
    @assert size(A) == size(B)
    return mean((A .- B) .^ 2)
end

square(x) = x^2
cube(x) = x^3
fourth(x) = x^4
fifth(x) = x^5

function make_model()
    return MultitargetSRRegressor(
        niterations = 400,
        populations = 200,
        binary_operators = [+, -, *, /],
        unary_operators = [square, cube, fourth, fifth],
        maxsize = 40,
        parsimony = 1e-4,
        parallelism = :multithreading,
        output_directory = SYMBOLIC_OUTPUTS_DIR,
    )
end

function main()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--train"
        dest_name = "train"
        help = "If true, train a symbolic regression model."
        arg_type = Bool
        default = true

        "--test"
        dest_name = "test"
        help = "If true, evaluate the SR model vs grevlex."
        arg_type = Bool
        default = true

        "--sr_mode"
        dest_name = "sr_mode"
        help = "Symbolic regression mode: '2target' (two targets and derived third) or '3target'."
        arg_type = String
        default = "3target"

        "--load_model"
        dest_name = "load_model"
        help = "If true, load SR model from default path instead of training (SR_MODEL_PATH)."
        arg_type = Bool
        default = false

        "--num_eval_ideals"
        dest_name = "num_eval_ideals"
        help = "Number of evaluation batches of random ideals."
        arg_type = Int
        default = 10

        "--eval_coefficients_per_ideal"
        dest_name = "eval_coefficients_per_ideal"
        help = "Number of random coefficients for each ideal evaluation."
        arg_type = Int
        default = 1000
    end

    args = parse_args(s)
    mode = args["sr_mode"]

    X, Y = build_sr_dataset(SAMPLES)

    println("X dims: $(size(X)), Y dims: $(size(Y))")
    @assert size(Y, 2) == 3 "Expecting exactly 3 weights/targets."

    ns = size(Y, 1)
    mean_vec = mean(Y, dims = 1)
    Y_mean = repeat(mean_vec, ns, 1)
    println("\nMSE(constant mean predictor) = ", mse(Y, Y_mean))

    best_pair_mach = nothing
    best_pair = nothing
    best_pair_data = nothing
    best_pair_mse = NaN
    mach_full = nothing
    r_full = nothing
    sr_mode_loaded = ""
    sym_eqns = nothing

    if args["load_model"]
        if isfile(SR_MODEL_PATH)
            println("Checkpoint found. Loading SR model from $SR_MODEL_PATH")

            model_state = open(SR_MODEL_PATH) do io
                deserialize(io)
            end

            best_pair_mse = model_state.best_pair_mse
            best_pair = model_state.best_pair
            best_pair_data = model_state.best_pair_data
            best_pair_mach = model_state.best_pair_mach
            mach_full = model_state.mach_full
            r_full = model_state.r_full
            sr_mode_loaded = model_state.sr_mode_loaded
            sym_eq1 = model_state.sym_eq1
            sym_eq2 = model_state.sym_eq2
            sym_eq3 = model_state.sym_eq3

            if best_pair_mach !== nothing
                println("Loaded 2-target SR model with MSE = $best_pair_mse")
            elseif mach_full !== nothing
                println("Loaded 3-target SR model")
            else
                error("Unknown SR model format in $SR_MODEL_PATH")
            end

            if sym_eq1 !== nothing && sym_eq2 !== nothing && sym_eq3 !== nothing
                sym_eqns = (sym_eq1, sym_eq2, sym_eq3)
            else
                sym_eqns = nothing
            end
        else
            error(
                "`--load_model` was set but no SR model checkpoint was found at $SR_MODEL_PATH",
            )
        end

    elseif args["train"]
        if mode == "2target"
            best_pair_mse, best_pair, best_pair_data, best_pair_mach = two_target_sr(X, Y)

            (i, j, k) = best_pair
            tmp = Vector{Any}(undef, 3)
            tmp[i] = best_pair_data.sym_i
            tmp[j] = best_pair_data.sym_j
            tmp[k] = best_pair_data.sym_k
            sym_eqns = (tmp[1], tmp[2], tmp[3])

            sym_eq1, sym_eq2, sym_eq3 = sym_eqns

            sr_mode_loaded = mode

            model_state = (
                best_pair_mse = best_pair_mse,
                best_pair = best_pair,
                best_pair_data = best_pair_data,
                best_pair_mach = best_pair_mach,
                mach_full = mach_full,
                r_full = r_full,
                sr_mode_loaded = sr_mode_loaded,
                sym_eq1 = sym_eq1,
                sym_eq2 = sym_eq2,
                sym_eq3 = sym_eq3,
            )

            open(SR_MODEL_PATH, "w") do io
                serialize(io, model_state)
            end

            println("Saved SR model checkpoint to $SR_MODEL_PATH")

        elseif mode == "3target"
            mach_full, r_full, sym_eqns, best_nodes = three_target_sr(X, Y)

            sym_eq1, sym_eq2, sym_eq3 = sym_eqns

            sr_mode_loaded = mode

            model_state = (
                best_pair_mse = best_pair_mse,
                best_pair = best_pair,
                best_pair_data = best_pair_data,
                best_pair_mach = best_pair_mach,
                mach_full = mach_full,
                r_full = r_full,
                sr_mode_loaded = sr_mode_loaded,
                sym_eq1 = sym_eq1,
                sym_eq2 = sym_eq2,
                sym_eq3 = sym_eq3,
            )

            open(SR_MODEL_PATH, "w") do io
                serialize(io, model_state)
            end

            println("Saved SR model checkpoint to $SR_MODEL_PATH")

        else
            error("Unknown --sr_mode $(mode). Expected '2target' or '3target'.")
        end

        if sym_eqns !== nothing
            sym_eq1, sym_eq2, sym_eq3 = sym_eqns
            BSON.@save(SR_EQNS_PATH, sr_mode_loaded, sym_eq1, sym_eq2, sym_eq3)
            println("Saved SR equations to $SR_EQNS_PATH")

            save_equations_text(SR_EQNS_TXT_PATH, sr_mode_loaded, sym_eqns)
        end
    end

    if args["test"]
        num_eval_ideals = args["num_eval_ideals"]
        eval_per_ideal = args["eval_coefficients_per_ideal"]

        if sr_mode_loaded == "2target"
            if best_pair_mach === nothing || best_pair === nothing
                error(
                    "Testing requested for 2-target SR, but no best-pair model is available. " *
                    "Train with `--sr_mode 2target` or load such a model with `--load_model`.",
                )
            end

            for idx = 1:num_eval_ideals
                println("Evaluating 2-target SR model on ideal batch $idx")
                _ = evaluate_sr_bestpair_vs_grevlex(
                    best_pair_mach,
                    best_pair,
                    Y;
                    M = eval_per_ideal,
                )
            end

        elseif sr_mode_loaded == "3target"
            if mach_full === nothing
                error(
                    "Testing requested for 3-target SR, but no full model is available. " *
                    "Train with `--sr_mode 3target` or load such a model with `--load_model`.",
                )
            end

            for idx = 1:num_eval_ideals
                println("Evaluating 3-target SR model on ideal batch $idx")
                _ = evaluate_sr_full_vs_grevlex(mach_full, Y; M = eval_per_ideal)
            end

        else
            error(
                "Testing was requested, but no SR model mode is known. " *
                "Make sure you either trained (`--train`) or loaded (`--load_model`) a 2target or 3target model.",
            )
        end
    end
end

main()
