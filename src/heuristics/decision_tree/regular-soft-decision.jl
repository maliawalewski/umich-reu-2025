using Flux
using Statistics
using Printf
using Random
using LinearAlgebra
using Serialization
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
    @warn "Could not find data.jl in expected locations. Downstream evaluation will fail unless you include it manually."
    return false
end

const HAS_DATAJL = include_datajl_if_present()

const OUT_DIR = joinpath(BASE_DIR, "outputs_tree")
const TREE_PATH = joinpath(OUT_DIR, "soft_tree.bin")
const DOT_PATH = joinpath(OUT_DIR, "tree.dot")
const PNG_PATH = joinpath(OUT_DIR, "tree.png")
isdir(OUT_DIR) || mkpath(OUT_DIR)

function child_rng(master::AbstractRNG, tag::Integer)
    return MersenneTwister(hash((rand(master, UInt64), tag)))
end

function flatten_sample(s)
    flat_matrix = reduce(vcat, reduce(vcat, s.monomial_matrix))
    return Float32.(flat_matrix)
end

function build_XY(samples, idxs::Vector{Int})
    @assert !isempty(idxs) "build_XY got empty idxs"
    X = reduce(hcat, map(i -> flatten_sample(samples[i]), idxs))
    Y = reduce(hcat, map(i -> Float32.(samples[i].order), idxs))
    return X, Y
end

function monomial_matrix_shape(mm)
    @assert !isempty(mm) "monomial_matrix is empty"
    n_polys = length(mm)
    n_mons = length(mm[1])
    @assert all(length(poly) == n_mons for poly in mm) "inconsistent number of monomials across polys"
    nvars = length(mm[1][1])
    @assert all(length(m) == nvars for poly in mm for m in poly) "inconsistent number of variables across monomials"
    return n_polys, n_mons, nvars
end

function flatten_monomial_matrix(mm)::Vector{Float32}
    n_polys, n_mons, nvars = monomial_matrix_shape(mm)
    v = Vector{Float32}(undef, n_polys * n_mons * nvars)
    k = 1
    @inbounds for p = 1:n_polys
        poly = mm[p]
        for m = 1:n_mons
            mon = poly[m]
            for j = 1:nvars
                v[k] = Float32(mon[j])
                k += 1
            end
        end
    end
    return v
end

struct SoftNode
    dense::Dense
end
Flux.@functor SoftNode

struct Leaf
    param::AbstractVector
end
Flux.@functor Leaf

struct SoftDecisionTree
    depth::Int
    nodes::Vector{SoftNode}
    leaves::Vector{Leaf}
end
Flux.@functor SoftDecisionTree

# function SoftDecisionTree(input_dim::Int, output_dim::Int, depth::Int)
    # @assert depth >= 1 "depth must be >= 1"
    # nodes = [SoftNode(Dense(input_dim, 1, sigmoid)) for _ = 1:(2^depth-1)]
    # leaves = [Leaf(randn(Float32, output_dim)) for _ = 1:(2^depth)]
    # return SoftDecisionTree(depth, nodes, leave)
# end

function SoftDecisionTree(rng::AbstractRNG, input_dim::Int, output_dim::Int, depth::Int)
    @assert depth >= 1 "depth must be >= 1"
    nodes  = [SoftNode(Dense(input_dim, 1, sigmoid; init = Flux.glorot_uniform(rng))) for _=1:(2^depth-1)]
    leaves = [Leaf(randn(rng, Float32, output_dim)) for _=1:(2^depth)]
    return SoftDecisionTree(depth, nodes, leaves)
end

function get_leaf_probs(tree::SoftDecisionTree, X::AbstractMatrix)
    # node_probs: (NumNodes, Batch)
    node_probs = vcat([n.dense(X) for n in tree.nodes]...)

    n_leaves = 2^tree.depth

    leaf_rows = map(0:(n_leaves-1)) do l
        prob = ones(Float32, 1, size(X, 2))
        curr = 1

        for d = 1:tree.depth
            bit = (l >> (tree.depth - d)) & 1
            p_node = view(node_probs, curr:curr, :)

            prob = bit == 1 ? prob .* p_node : prob .* (1.0f0 .- p_node)
            curr = bit == 1 ? (2 * curr + 1) : (2 * curr)
        end
        return prob
    end

    return vcat(leaf_rows...)  # (NumLeaves, Batch)
end

function (tree::SoftDecisionTree)(X::AbstractMatrix)
    P = get_leaf_probs(tree, X)                 # (Leaves, Batch)
    W = hcat([l.param for l in tree.leaves]...) # (Out, Leaves)
    return W * P                                # (Out, Batch)
end

function get_feature_name(idx::Int)
    vec_idx = div(idx - 1, 3) + 1
    coord = ["x", "y", "z"][mod(idx-1, 3)+1]
    return "Vec$vec_idx.$coord"
end

function export_viz(
    tree::SoftDecisionTree;
    dot_file::AbstractString = DOT_PATH,
    png_file::AbstractString = PNG_PATH,
)
    open(dot_file, "w") do io
        write(
            io,
            """digraph Tree {
    graph [dpi=300];
    node [fontname="Arial", shape=box, style=filled, fillcolor="#f9f9f9"];
""",
        )

        function walk(node_idx::Int, current_depth::Int)
            if current_depth > tree.depth
                leaf_idx = node_idx - (2^tree.depth) + 1
                leaf = tree.leaves[leaf_idx]
                w_str = join([@sprintf("%.2f", x) for x in leaf.param], ",\\n")
                write(
                    io,
                    "    $node_idx [label=\"Weights:\\n[$w_str]\", shape=ellipse, fillcolor=\"#dff0d8\"];\n",
                )
                return
            end

            wmat = tree.nodes[node_idx].dense.weight
            feat_idx = argmax(abs.(vec(wmat)))
            feat = get_feature_name(feat_idx)

            write(io, "    $node_idx [label=\"Is $feat high?\", fillcolor=\"#d9edf7\"];\n")
            write(io, "    $node_idx -> $(2*node_idx) [label=\"No\", color=\"red\"];\n")
            write(io, "    $node_idx -> $(2*node_idx+1) [label=\"Yes\", color=\"blue\"];\n")

            walk(2*node_idx, current_depth + 1)
            walk(2*node_idx + 1, current_depth + 1)
        end

        walk(1, 1)
        write(io, "}\n")
    end

    if Sys.which("dot") !== nothing
        try
            run(`dot -Tpng $dot_file -o $png_file`)
            println("Visualization saved to: $png_file")
        catch e
            println("Error generating PNG via GraphViz: $e")
            println("DOT file still saved to: $dot_file")
        end
    else
        println("Warning: 'dot' tool not found. Install GraphViz to generate PNG.")
        println("DOT file saved to: $dot_file")
    end
end

function train_tree(X, Y; depth::Int, lr::Float64, lambda::Float64, epochs::Int, rng::AbstractRNG)
    @assert size(X, 2) == size(Y, 2) "X and Y must have same number of columns"
    input_dim = size(X, 1)
    output_dim = size(Y, 1)

    tree = SoftDecisionTree(rng, input_dim, output_dim, depth)
    opt_state = Flux.setup(Flux.Adam(lr), tree)

    for _ = 1:epochs
        grads = Flux.gradient(tree) do m
            preds = m(X)
            loss_mse = Flux.mse(preds, Y)

            probs = get_leaf_probs(m, X)
            Q = mean(probs, dims = 2)
            loss_ent = sum(Q .* log.(Q .+ 1.0f-8))

            return loss_mse + (lambda * loss_ent)
        end
        Flux.update!(opt_state, tree, grads[1])
    end

    return tree
end

function eval_tree_prediction(tree, Xtest, Ytest)
    @assert size(Xtest, 2) == size(Ytest, 2) "Xtest and Ytest must have same number of columns"
    Yhat = tree(Xtest)
    mse = Flux.mse(Yhat, Ytest)

    num = sum(Yhat .* Ytest; dims = 1)
    den = sqrt.(sum(Yhat .^ 2; dims = 1) .* sum(Ytest .^ 2; dims = 1)) .+ 1e-12
    cos = mean(vec(num ./ den))

    return (mse = Float64(mse), cos = Float64(cos))
end

function kfold_indices(n::Int, k::Int; rng = Random.default_rng(), shuffle::Bool = true)
    @assert k >= 2 "k must be >= 2"
    @assert n >= k "need n >= k"
    idx = collect(1:n)
    if shuffle
        Random.shuffle!(rng, idx)
    end
    folds = [Int[] for _ = 1:k]
    for (i, id) in enumerate(idx)
        push!(folds[mod1(i, k)], id)
    end
    out = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, k)
    for f = 1:k
        test_idx = folds[f]
        train_idx = vcat(folds[setdiff(1:k, [f])]...)
        @assert isempty(intersect(train_idx, test_idx)) "train/test overlap in fold construction"
        out[f] = (train_idx, test_idx)
    end
    return out
end


function cross_validate_base_set_prediction(
    samples;
    k::Int = 5,
    seed::Int = 0,
    depth::Int = 6,
    lr::Float64 = 1e-3,
    lambda::Float64 = 0.05,
    epochs::Int = 4000,
)
    master = MersenneTwister(seed)

    fold_rng = child_rng(master, 10_000)
    folds = kfold_indices(length(samples), k; rng = fold_rng, shuffle = true)

    mses = Float64[]
    coss = Float64[]

    for (f, (train_idx, test_idx)) in enumerate(folds)
        Xtr, Ytr = build_XY(samples, train_idx)
        Xte, Yte = build_XY(samples, test_idx)

        tr_rng = child_rng(master, 20_000 + f)

        tree = train_tree(
            Xtr, Ytr;
            depth = depth, lr = lr, lambda = lambda, epochs = epochs,
            rng = tr_rng
        )

        r = eval_tree_prediction(tree, Xte, Yte)

        println("Fold $f/$k: test MSE=$(round(r.mse, sigdigits=5)), test cos=$(round(r.cos, sigdigits=5))")
        push!(mses, r.mse)
        push!(coss, r.cos)
    end

    println("\nCV summary over $k folds (base-set prediction)")
    println("MSE: mean=$(mean(mses))  std=$(std(mses))")
    println("Cos: mean=$(mean(coss))  std=$(std(coss))")

    return (mses = mses, coss = coss)
end


function cross_validate_base_set_downstream(
    samples;
    k::Int = 5,
    seed::Int = 0,
    depth::Int = 6,
    lr::Float64 = 1e-3,
    lambda::Float64 = 0.05,
    epochs::Int = 4000,
    num_ideals_per_base::Int = 200,
    scale::Real = 1000,
    verbose_samples::Bool = true,
    max_samples_to_print_per_fold::Int = 2,
)
    @assert HAS_DATAJL "Need data.jl for new_generate_data/groebner_learn."

    master = MersenneTwister(seed)
    fold_rng = child_rng(master, 30_000)
    folds = kfold_indices(length(samples), k; rng = fold_rng, shuffle = true)

    _, Yall = build_XY(samples, collect(1:length(samples)))
    fold_means = Float64[]

    for (f, (train_idx, test_idx)) in enumerate(folds)
        println("\n==============================")
        println("Downstream CV fold $f / $k")
        println("Train size = $(length(train_idx)), Test size = $(length(test_idx))")
        println("==============================")

        Xtr, Ytr = build_XY(samples, train_idx)

        tr_rng = child_rng(master, 40_000 + f)
        tree = train_tree(
            Xtr, Ytr;
            depth = depth, lr = lr, lambda = lambda, epochs = epochs,
            rng = tr_rng
        )

        heldout_scores = Float64[]
        for (j, i) in enumerate(test_idx)
            do_print = verbose_samples && (j <= max_samples_to_print_per_fold)
            s_mean = eval_tree_downstream_on_one_base_set(
                tree,
                samples[i],
                Yall;
                sample_index = i,
                num_ideals = num_ideals_per_base,
                scale = scale,
                verbose = do_print,
            )
            push!(heldout_scores, s_mean)

            if !do_print
                println("Held-out sample index $i: mean reward_diff (tree - grevlex) = $s_mean")
            end
        end

        fold_mean = mean(heldout_scores)
        println("\nFold $f summary:")
        println("Mean reward_diff (tree - grevlex) across held-out base sets = $fold_mean")
        push!(fold_means, fold_mean)
    end

    println("\n==============================")
    println("Downstream CV summary over $k folds")
    println("Mean reward_diff (tree - grevlex): mean=$(mean(fold_means)) std=$(std(fold_means))")
    println("==============================")

    return fold_means
end


function predict_weights_from_tree(
    tree::SoftDecisionTree,
    monomial_matrix;
    scale::Real = 1000,
    eps::Real = 1e-9,
)
    x = flatten_monomial_matrix(monomial_matrix)
    X = reshape(x, :, 1)
    y = vec(tree(X))

    y64 = Float64.(y)
    y_pos = log1p.(exp.(y64)) .+ eps
    y_pos ./= sum(y_pos)

    w_int = round.(Int, y_pos .* scale)
    w_int[w_int .<= 0] .= 1
    return w_int
end

function reward(trace::Groebner.WrappedTrace)
    @assert length(trace.recorded_traces) == 1 "WrappedTrace struct is tracking multiple traces"
    total_reward = 0.0
    for (_, t) in trace.recorded_traces
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
    println("percent of time agent wins:  $(round(win_pct,  digits=2)) percent")
    println("percent of time agent loses: $(round(loss_pct, digits=2)) percent")
    println("percent of time tie:         $(round(tie_pct,  digits=2)) percent")
    println(
        "average improvement percent (when winning): $(round(avg_improvement, digits=2))",
    )
    println(
        "average degradation percent (when losing):  $(round(avg_degradation, digits=2))",
    )
    println()
end

function evaluate_tree_vs_grevlex(
    tree::SoftDecisionTree,
    Y;
    num_ideals::Integer = 1000,
    max_attempts::Integer = 100,
    scale::Real = 1000,
)
    @assert HAS_DATAJL "data.jl not included; cannot call new_generate_data/groebner_learn."
    @assert isdefined(Main, :SAMPLES) "SAMPLES not defined (did ideal_ordering_pairs.jl load?)"

    mm_template = Main.SAMPLES[1].monomial_matrix
    n_polys, n_terms, nvars = monomial_matrix_shape(mm_template)

    outdim = size(Y, 1)
    @assert outdim == nvars "tree output dim=$outdim but test generator says nvars=$nvars"

    ideals_test, vars, base_set = new_generate_data(
        num_ideals = num_ideals,
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

    w_int = predict_weights_from_tree(tree, base_set; scale = scale)
    println("Tree weight vector: ", w_int)

    rewards_tree = Float64[]
    rewards_grev = Float64[]
    reward_diffs = Float64[]

    for (idx, ideal) in enumerate(ideals_test)
        if idx % 100 == 0
            println("evaluating tree weights: $idx / $num_ideals")
        end

        r_tree, r_grev, r_diff = act_vs_grevlex(w_int, vars, ideal)

        if idx % 100 == 0
            println("(tree reward - grevlex reward) at $idx: $r_diff")
        end

        push!(rewards_tree, r_tree)
        push!(rewards_grev, r_grev)
        push!(reward_diffs, r_diff)
    end

    println("\nTree ordering vs grevlex on $num_ideals random ideals")
    println("Average reward_diff (tree - grevlex): ", mean(reward_diffs))
    println("Minimum reward_diff:                  ", minimum(reward_diffs))
    println("Maximum reward_diff:                  ", maximum(reward_diffs))
    compute_stats(rewards_tree, rewards_grev, "DegRevLex")

    return reward_diffs
end

function eval_tree_downstream_on_one_base_set(
    tree,
    sample,
    Yall;
    sample_index::Int = -1,
    num_ideals::Int = 200,
    scale::Real = 1000,
    max_attempts::Int = 100,
    verbose::Bool = true,
)
    @assert HAS_DATAJL "Need data.jl for new_generate_data/groebner_learn."

    base_set = sample.monomial_matrix
    n_polys, n_terms, nvars = monomial_matrix_shape(base_set)
    @assert size(Yall, 1) == nvars "Yall output dim does not match nvars."

    base_sets_any = Vector{Any}(base_set)

    ideals_test, vars, used_base_sets = new_generate_data(
        num_ideals = num_ideals,
        num_polynomials = n_polys,
        num_variables = nvars,
        max_degree = 4,
        num_terms = n_terms,
        max_attempts = max_attempts,
        base_sets = base_sets_any,
        base_set_path = nothing,
        should_save_base_sets = false,
    )

    @assert length(used_base_sets) == n_polys "Generator returned wrong number of base sets"

    w_int = predict_weights_from_tree(tree, base_set; scale = scale)

    if verbose
        println("\nDownstream eval on held-out base set sample index = $sample_index")
        println("Held-out base set:")
        println(base_set)
        println("Held-out target order (dataset): ", sample.order)
        println("Predicted integer weights: ", w_int)
        println(
            "Evaluating on $num_ideals new random ideals generated using this base set...",
        )
    end

    rewards_tree = Float64[]
    rewards_grev = Float64[]
    reward_diffs = Float64[]

    for ideal in ideals_test
        r_tree, r_grev, r_diff = act_vs_grevlex(w_int, vars, ideal)
        push!(rewards_tree, r_tree)
        push!(rewards_grev, r_grev)
        push!(reward_diffs, r_diff)
    end

    if verbose
        println(
            "Mean reward_diff (tree - grevlex) for this held-out sample: ",
            mean(reward_diffs),
        )
        compute_stats(rewards_tree, rewards_grev, "DegRevLex")
    end

    return mean(reward_diffs)
end


function main(;
    train::Bool = true,
    test::Bool = false,
    load_model::Bool = false,
    depth::Int = 4,
    lambda::Float64 = 0.05,
    lr::Float64 = 0.001,
    epochs::Int = 4000,
    num_eval_batches::Int = 10,
    eval_per_batch::Int = 1000,
    scale::Real = 1000,
    seed::Int = 0,
)
    @assert isdefined(Main, :SAMPLES) "SAMPLES not defined after including ideal_ordering_pairs.jl."
    samples = Main.SAMPLES
    println("Successfully loaded $(length(samples)) samples from ideal_ordering_pairs.jl")

    X = reduce(hcat, map(flatten_sample, samples))
    Y = reduce(hcat, map(s -> Float32.(s.order), samples))
    println("Data Shapes: X=$(size(X)), Y=$(size(Y))")

    tree = nothing

    if load_model
        isfile(TREE_PATH) || error("load_model=true but no model found at $TREE_PATH")
        tree = open(TREE_PATH, "r") do io
            deserialize(io)
        end
        println("Loaded tree from $TREE_PATH")

    elseif train
        rng = MersenneTwister(seed)
        tree = SoftDecisionTree(rng, size(X, 1), size(Y, 1), depth)

        opt_state = Flux.setup(Flux.Adam(lr), tree)

        println("\nStarting Training (MSE + entropy regularization)...")
        for epoch = 1:epochs
            grads = Flux.gradient(tree) do m
                preds = m(X)
                loss_mse = Flux.mse(preds, Y)

                probs = get_leaf_probs(m, X)
                Q = mean(probs, dims = 2)
                loss_ent = sum(Q .* log.(Q .+ 1.0f-8))

                return loss_mse + (lambda * loss_ent)
            end

            Flux.update!(opt_state, tree, grads[1])

            if epoch % 200 == 0
                preds = tree(X)
                mse = Flux.mse(preds, Y)
                probs = get_leaf_probs(tree, X)
                Q = mean(probs, dims = 2)
                entropy = -sum(Q .* log.(Q .+ 1.0f-8))
                @printf("Epoch %d: MSE=%.5f | Leaf Entropy=%.4f\n", epoch, mse, entropy)
            end
        end

        open(TREE_PATH, "w") do io
            serialize(io, tree)
        end
        println("Saved tree to $TREE_PATH")

        export_viz(tree)
    end

    if test
        @assert tree !== nothing "No tree available to test (train or load first)."
        @assert HAS_DATAJL "Cannot test without data.jl included."

        for b = 1:num_eval_batches
            println("\nEvaluating tree on ideal batch $b / $num_eval_batches")
            _ = evaluate_tree_vs_grevlex(tree, Y; num_ideals = eval_per_batch, scale = scale)
        end
    end

    return tree
end

main(
    train = true,
    test = true,
    load_model = false,
    num_eval_batches = 10,
    eval_per_batch = 1000,
    scale = 1000,
)

samples = Main.SAMPLES

cross_validate_base_set_prediction(
    samples;
    k = 5,
    seed = 0,
    depth = 4,
    lr = 1e-3,
    epochs = 4000,
)

cross_validate_base_set_prediction(
    samples;
    k = 5,
    seed = 0,
    depth = 6,
    lr = 1e-3,
    epochs = 4000,
)

cross_validate_base_set_downstream(
    samples;
    k = 3,
    seed = 0,
    depth = 4,
    lr = 1e-3,
    epochs = 4000,
    num_ideals_per_base = 50,
    verbose_samples = true,
    max_samples_to_print_per_fold = 2,
)

cross_validate_base_set_downstream(
    samples;
    k = 3,
    seed = 0,
    depth = 6,
    lr = 1e-3,
    epochs = 4000,
    num_ideals_per_base = 50,
    verbose_samples = true,
    max_samples_to_print_per_fold = 2,
)
