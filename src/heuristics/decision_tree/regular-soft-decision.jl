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
    @warn "Could not find data.jl in expected locations. Evaluation vs grevlex will fail unless you include it manually."
    return false
end

const HAS_DATAJL = include_datajl_if_present()

const OUT_DIR = joinpath(BASE_DIR, "outputs_tree")
const TREE_PATH = joinpath(OUT_DIR, "soft_tree.bin")
const DOT_PATH = joinpath(OUT_DIR, "tree.dot")
const PNG_PATH = joinpath(OUT_DIR, "tree.png")
isdir(OUT_DIR) || mkpath(OUT_DIR)

function flatten_sample(s)
    flat_matrix = reduce(vcat, reduce(vcat, s.monomial_matrix))
    return Float32.(flat_matrix)
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

function SoftDecisionTree(input_dim::Int, output_dim::Int, depth::Int)
    nodes = [SoftNode(Dense(input_dim, 1, sigmoid)) for _ = 1:(2^depth-1)]
    leaves = [Leaf(randn(Float32, output_dim)) for _ = 1:(2^depth)]
    return SoftDecisionTree(depth, nodes, leaves)
end

function get_leaf_probs(tree::SoftDecisionTree, X::AbstractMatrix)
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
    P = get_leaf_probs(tree, X)                       # (Leaves, Batch)
    W = hcat([l.param for l in tree.leaves]...)       # (Out, Leaves)
    return W * P                                      # (Out, Batch)
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

            wmat = tree.nodes[node_idx].dense.weight  # (1, input_dim)
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

function predict_weights_from_tree(
    tree::SoftDecisionTree,
    monomial_matrix;
    scale::Real = 1000,
    eps::Real = 1e-9,
)
    x = flatten_monomial_matrix(monomial_matrix)
    X = reshape(x, :, 1)              # (input_dim, 1)
    y = vec(tree(X))                  # Float32 length = nvars

    y64 = Float64.(y)
    y_pos = log1p.(exp.(y64)) .+ eps  # softplus + eps
    y_pos ./= sum(y_pos)

    w_int = round.(Int, y_pos .* scale)
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

function evaluate_tree_vs_grevlex(
    tree::SoftDecisionTree,
    Y;
    M::Integer = 1000,
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

    w_int = predict_weights_from_tree(tree, base_set; scale = scale)
    println("Tree Weight Vector: ", w_int)

    rewards_lin = Float64[]
    rewards_grev = Float64[]
    delta_rewards = Float64[]

    for (idx, ideal) in enumerate(ideals_test)
        if idx % 100 == 0
            println("evaluating tree weights: $idx / $M")
        end

        lin_reward, grev_reward, delta_reward = act_vs_grevlex(w_int, vars, ideal)

        if idx % 100 == 0
            println("(tree reward - grevlex reward) at $idx: $delta_reward")
        end

        push!(rewards_lin, lin_reward)
        push!(rewards_grev, grev_reward)
        push!(delta_rewards, delta_reward)
    end

    println("\nTree ordering vs grevlex on $M random ideals")
    println("Average delta_reward (tree - grevlex): ", mean(delta_rewards))
    println("Minimum delta_reward:                  ", minimum(delta_rewards))
    println("Maximum delta_reward:                  ", maximum(delta_rewards))
    compute_stats(rewards_lin, rewards_grev, "DegRevLex")

    return delta_rewards
end

function main(;
    train::Bool = true,
    test::Bool = false,
    load_model::Bool = false,
    depth::Int = 6,
    lambda::Float64 = 0.05,
    lr::Float64 = 0.001,
    epochs::Int = 4000,
    num_eval_batches::Int = 10,
    eval_per_batch::Int = 1000,
    scale::Real = 1000,
)

    @assert isdefined(Main, :SAMPLES) "SAMPLES not defined after including ideal_ordering_pairs.jl."
    samples = Main.SAMPLES
    println("Successfully loaded $(length(samples)) samples from ideal_ordering_pairs.jl")

    # X: (feat, N), Y: (out, N)
    X = reduce(hcat, map(flatten_sample, samples))
    Y = reduce(hcat, map(s -> Float32.(s.order), samples))
    println("Data Shapes: X=$(size(X)), Y=$(size(Y))")

    INPUT_DIM = size(X, 1)
    OUTPUT_DIM = size(Y, 1)

    tree = nothing

    if load_model
        isfile(TREE_PATH) || error("load_model=true but no model found at $TREE_PATH")
        tree = open(TREE_PATH, "r") do io
            deserialize(io)
        end
        println("Loaded tree from $TREE_PATH")

    elseif train
        tree = SoftDecisionTree(INPUT_DIM, OUTPUT_DIM, depth)
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
            _ = evaluate_tree_vs_grevlex(tree, Y; M = eval_per_batch, scale = scale)
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
