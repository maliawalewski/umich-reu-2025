using Random
using Statistics
using Serialization
using Printf

using MLJ
LassoRegressor = @load LassoRegressor pkg=MLJLinearModels verbosity=0
ElasticNetRegressor = @load ElasticNetRegressor pkg=MLJLinearModels verbosity=0
Standardizer = @load Standardizer pkg=MLJTransforms verbosity=0

using AbstractAlgebra
using Groebner

const BASE_DIR = @__DIR__

let p = joinpath(BASE_DIR, "..", "ideal_ordering_pairs.jl")
    isfile(p) || error("Could not find ideal_ordering_pairs.jl at $p")
    include(p)
end

function include_datajl_if_present()
    candidates =
        [joinpath(BASE_DIR, "..", "data.jl"), joinpath(BASE_DIR, "..", "..", "data.jl")]
    for p in candidates
        if isfile(p)
            @info "Including data.jl from $p"
            include(p)
            return true
        end
    end
    @warn "Could not find data.jl in expected locations. Evaluation vs grevlex will fail unless you include it manually."
    return false
end

const HAS_DATAJL = include_datajl_if_present()

const OUT_DIR = joinpath(BASE_DIR, "outputs")
const MODEL_PATH = joinpath(OUT_DIR, "model.bin")
const COEFS_TXT_PATH = joinpath(OUT_DIR, "equations.txt")
isdir(OUT_DIR) || mkpath(OUT_DIR)

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

function build_dataset(samples)
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

mse(A, B) = mean((A .- B) .^ 2)

function alr_transform(W::AbstractMatrix; denom_idx::Int = size(W, 2), eps::Float64 = 1e-9)
    ns, q = size(W)
    @assert q >= 2
    @assert 1 <= denom_idx <= q
    keep = [j for j = 1:q if j != denom_idx]
    Z = Matrix{Float64}(undef, ns, q-1)
    d = clamp.(W[:, denom_idx], eps, Inf)
    for (k, j) in enumerate(keep)
        num = clamp.(W[:, j], eps, Inf)
        Z[:, k] = log.(num ./ d)
    end
    return Z, keep, denom_idx
end

function alr_inverse(Z::AbstractMatrix, keep::Vector{Int}, denom_idx::Int)
    ns, qm1 = size(Z)
    q = qm1 + 1
    U = ones(Float64, ns, q)
    for (k, j) in enumerate(keep)
        U[:, j] .= exp.(Z[:, k])
    end
    W = U ./ sum(U, dims = 2)
    return W
end

@enum RegressorKind::Int8 begin
    KIND_LASSO = 1
    KIND_ELASTICNET = 2
end

function make_pipeline(kind::RegressorKind)
    if kind == KIND_LASSO
        return MLJ.Pipeline(stand = Standardizer(), reg = LassoRegressor())
    elseif kind == KIND_ELASTICNET
        return MLJ.Pipeline(stand = Standardizer(), reg = ElasticNetRegressor())
    else
        error("Unknown regressor kind $kind")
    end
end

function fit_linear_multitarget(
    X::AbstractMatrix,
    W::AbstractMatrix;
    kind::RegressorKind = KIND_ELASTICNET,
    train_frac::Float64 = 0.8,
    nfolds::Int = 5,
    rng::Int = 123,
    lambda_vals = vcat(0.0, 10.0 .^ range(-10, -2, length = 25)),
    gamma_vals = vcat(0.0, 10.0 .^ range(-10, -2, length = 25)),
    denom_idx::Int = size(W, 2),
)
    ns, _ = size(X)
    ns2, q = size(W)
    @assert ns == ns2 "X and W row counts differ"
    @assert q >= 2

    Xtbl = MLJ.table(X)
    train, test = MLJ.partition(1:ns, train_frac, shuffle = true, rng = rng)

    println("target stds = ", vec(std(W, dims = 1)))
    println("min/max per target = ", [(minimum(W[:, j]), maximum(W[:, j])) for j = 1:q])

    Z, keep, denom = alr_transform(W; denom_idx = denom_idx)
    qm1 = size(Z, 2)
    @info "ALR: fitting $(qm1) targets (keep=$(keep), denom=$(denom))"

    pipe = make_pipeline(kind)

    r_lambda = MLJ.range(pipe, :(reg.lambda), values = lambda_vals)
    r_gamma = nothing
    if kind == KIND_ELASTICNET
        r_gamma = MLJ.range(pipe, :(reg.gamma), values = gamma_vals)
    end

    machines = Vector{Any}(undef, qm1)
    best_lambda = Vector{Float64}(undef, qm1)
    best_gamma = fill(NaN, qm1)

    for j = 1:qm1
        ranges = (kind == KIND_ELASTICNET) ? [r_lambda, r_gamma] : [r_lambda]

        tm = MLJ.TunedModel(
            model = pipe,
            tuning = MLJ.Grid(),
            resampling = MLJ.CV(nfolds = nfolds, shuffle = true, rng = rng),
            range = ranges,
            measure = MLJ.rms,
            acceleration = MLJ.CPUThreads(),
        )

        mach = MLJ.machine(tm, Xtbl, Z[:, j])
        MLJ.fit!(mach, rows = train)

        machines[j] = mach

        fp = MLJ.fitted_params(mach)
        best_lambda[j] = fp.best_model.reg.lambda
        if kind == KIND_ELASTICNET
            best_gamma[j] = fp.best_model.reg.gamma
        end
    end

    predsZ = [MLJ.predict(machines[j], rows = test) for j = 1:qm1]
    Zhat = Float64.(hcat(predsZ...))

    What = alr_inverse(Zhat, keep, denom)

    return (
        kind = kind,
        machines = machines,
        best_lambda = best_lambda,
        best_gamma = best_gamma,
        train = train,
        test = test,
        keep = keep,
        denom = denom,
        What_test = What,
    )
end

function _extract_intercept_beta(best_fitted_params)
    regfit = best_fitted_params.reg
    intercept = regfit.intercept

    coefs = regfit.coefs
    if coefs isa NamedTuple
        beta = Float64.(collect(values(coefs)))
    else
        beta = Float64.(last.(coefs))
    end
    return intercept, beta
end

function fitted_linear_params(fit, j::Int)
    mach = fit.machines[j]
    fp = MLJ.fitted_params(mach)
    best = fp.best_fitted_params
    return _extract_intercept_beta(best)
end

function save_linear_equations_text(path::AbstractString, fit; topk::Int = 25)
    open(path, "w") do io
        kind_str = fit.kind == KIND_LASSO ? "LASSO" : "ELASTIC NET"
        println(io, "$(kind_str) (MLJLinearModels) summary")
        println(io, "ALR keep = ", fit.keep, " ; denom = ", fit.denom)
        println(io, "Targets fitted (ALR space): ", length(fit.machines))
        println(io, "Best lambdas per ALR-target: ", fit.best_lambda)
        if fit.kind == KIND_ELASTICNET
            println(io, "Best gammas  per ALR-target: ", fit.best_gamma)
        end
        println(io)
        println(
            io,
            "NOTE: coefficients are for STANDARDIZED features (Standardizer in pipeline).",
        )
        println(
            io,
            "Target mode: ALR log-ratios, then inverted to simplex at predict time.",
        )
        println(io)

        for j = 1:length(fit.machines)
            intercept, beta = fitted_linear_params(fit, j)
            nz = findall(!iszero, beta)
            sort!(nz, by = i -> abs(beta[i]), rev = true)

            println(io, "==============================")
            println(io, "ALR target j = $j")
            println(io, "intercept = ", intercept)
            println(io, "nonzeros  = ", length(nz), " / ", length(beta))
            println(io, "top ", min(topk, length(nz)), " coefficients by |beta|:")
            for idx in nz[1:min(topk, length(nz))]
                @printf(io, "  x[%d]  beta = %+0.6e\n", idx, beta[idx])
            end
            println(io)
        end
    end
    println("Saved equations summary to $path")
end

function predict_weights_from_fit(fit, monomial_matrix; scale::Real = 1000)
    x = flatten_monomial_matrix(monomial_matrix)
    Xnew = MLJ.table(reshape(x, 1, :))

    qm1 = length(fit.machines)
    z = Vector{Float64}(undef, qm1)
    for j = 1:qm1
        z[j] = MLJ.predict(fit.machines[j], Xnew)[1]
    end

    W = alr_inverse(reshape(z, 1, :), fit.keep, fit.denom)
    w = vec(W[1, :])

    w_int = round.(Int, w .* scale)
    w_int[w_int .<= 0] .= 1
    return w_int
end

function reward(trace::Groebner.WrappedTrace)
    @assert length(trace.recorded_traces) == 1 "WrappedTrace struct is tracking multiple traces"
    total_reward = 0.0
    for (k, t) in trace.recorded_traces
        @assert length(t.critical_pair_sequence) == (length(t.matrix_infos) - 1)
        for i = 1:length(t.critical_pair_sequence)
            n_cols = Float64(t.matrix_infos[i+1][3]) / 100.0
            pair_degree = t.critical_pair_sequence[i][1]
            pair_count = t.critical_pair_sequence[i][2]
            total_reward += n_cols * log(pair_degree) * pair_count
        end
    end
    return -total_reward
end

function act_vs_grevlex(
    action::AbstractVector{<:Integer},
    vars::AbstractVector{<:AbstractAlgebra.Generic.MPoly},
    ideal::AbstractVector{<:AbstractAlgebra.Generic.MPoly},
)
    @assert length(action) == length(vars)

    weights = zip(vars, action)
    order = WeightedOrdering(weights...)

    trace_lin, _ = groebner_learn(ideal, ordering = order)
    trace_grev, _ = groebner_learn(ideal, ordering = DegRevLex())

    lin_reward = reward(trace_lin)
    grev_reward = reward(trace_grev)
    return lin_reward, grev_reward, lin_reward - grev_reward
end

function compute_stats(agent_rewards, baseline_rewards, name)
    total = length(agent_rewards)
    @assert total == length(baseline_rewards)

    wins = [a > b for (a, b) in zip(agent_rewards, baseline_rewards)]
    losses = [a < b for (a, b) in zip(agent_rewards, baseline_rewards)]
    ties = [a == b for (a, b) in zip(agent_rewards, baseline_rewards)]

    win_pct = 100 * count(wins) / total
    loss_pct = 100 * count(losses) / total
    tie_pct = 100 * count(ties) / total

    improvements = [
        (a-b)/max(abs(b), 1e-8)*100 for
        (a, b) in zip(agent_rewards, baseline_rewards) if a>b
    ]
    degradations = [
        (b-a)/max(abs(b), 1e-8)*100 for
        (a, b) in zip(agent_rewards, baseline_rewards) if a<b
    ]

    avg_improvement = isempty(improvements) ? 0.0 : mean(improvements)
    avg_degradation = isempty(degradations) ? 0.0 : mean(degradations)

    println("agent vs $name")
    println("percent of time agent wins:  $(round(win_pct,  digits=2)) percent")
    println("percent of time agent loses: $(round(loss_pct, digits=2)) percent")
    println("percent of time tie:         $(round(tie_pct,  digits=2)) percent")
    println("average improvement % (when winning): $(round(avg_improvement, digits=2))")
    println("average degradation % (when losing):  $(round(avg_degradation, digits=2))")
    println()
end

function evaluate_fit_vs_grevlex(
    fit,
    Y;
    M::Integer = 1000,
    max_attempts::Integer = 100,
    scale::Real = 1000,
)
    @assert HAS_DATAJL "data.jl not included; cannot call new_generate_data/groebner_learn."

    mm_template = SAMPLES[1].monomial_matrix
    n_polys, n_terms, nvars = monomial_matrix_shape(mm_template)
    @assert nvars == size(Y, 2) "trained for $(size(Y,2)) weights, but monomial_matrix has $nvars vars"

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

    println("Base set:")
    println(base_set)

    w_int = predict_weights_from_fit(fit, base_set; scale = scale)
    kind_str = fit.kind == KIND_LASSO ? "LASSO" : "ELASTIC NET"
    println("$kind_str Weight Vector: ", w_int)

    rewards_lin = Float64[]
    rewards_grev = Float64[]
    delta_rewards = Float64[]

    for (idx, ideal) in enumerate(ideals_test)
        if idx % 100 == 0
            println("evaluating coefficients: $idx / $M")
        end

        lin_reward, grev_reward, delta_reward = act_vs_grevlex(w_int, vars, ideal)

        if idx % 100 == 0
            println("($kind_str reward - grevlex reward) at $idx: $delta_reward")
        end

        push!(rewards_lin, lin_reward)
        push!(rewards_grev, grev_reward)
        push!(delta_rewards, delta_reward)
    end

    println("\n$kind_str ordering vs grevlex on $M random ideals")
    println("Average delta_reward (model - grevlex): ", mean(delta_rewards))
    println("Minimum delta_reward:                   ", minimum(delta_rewards))
    println("Maximum delta_reward:                   ", maximum(delta_rewards))
    compute_stats(rewards_lin, rewards_grev, "DegRevLex")

    return delta_rewards
end

function main(;
    train::Bool = true,
    test::Bool = false,
    load_model::Bool = false,
    model_kind::Symbol = :elasticnet,
    num_eval_batches::Int = 1,
    eval_per_batch::Int = 1000,
)

    X, Y = build_dataset(SAMPLES)
    println("X dims: $(size(X)), Y dims: $(size(Y))")

    ns = size(Y, 1)
    mean_vec = mean(Y, dims = 1)
    Y_mean = repeat(mean_vec, ns, 1)
    println("MSE(constant mean predictor) = ", mse(Y, Y_mean))

    fit = nothing

    if load_model
        isfile(MODEL_PATH) || error("load_model=true but no model found at $MODEL_PATH")
        fit = open(MODEL_PATH, "r") do io
            deserialize(io)
        end
        println("Loaded model from $MODEL_PATH")
        println("kind        = ", fit.kind == KIND_LASSO ? "LASSO" : "ELASTIC NET")
        println("best_lambda = ", fit.best_lambda)
        if fit.kind == KIND_ELASTICNET
            println("best_gamma  = ", fit.best_gamma)
        end

    elseif train
        kind =
            model_kind == :lasso ? KIND_LASSO :
            model_kind == :elasticnet ? KIND_ELASTICNET :
            error("model_kind must be :lasso or :elasticnet")

        fit = fit_linear_multitarget(X, Y; kind = kind)
        println("best_lambda = ", fit.best_lambda)
        if fit.kind == KIND_ELASTICNET
            println("best_gamma  = ", fit.best_gamma)
        end

        open(MODEL_PATH, "w") do io
            serialize(io, fit)
        end
        println("Saved model to $MODEL_PATH")

        save_linear_equations_text(COEFS_TXT_PATH, fit; topk = 30)
    end

    if test
        @assert fit !== nothing "No model available to test (train or load first)."
        @assert HAS_DATAJL "Cannot test without data.jl included."

        for b = 1:num_eval_batches
            println("\nEvaluating model on ideal batch $b / $num_eval_batches")
            _ = evaluate_fit_vs_grevlex(fit, Y; M = eval_per_batch)
        end
    end

    return fit
end

main(
    train = true,
    test = true,
    load_model = false,
    model_kind = :elasticnet,
    num_eval_batches = 10,
    eval_per_batch = 1000,
)
