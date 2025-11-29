using Statistics
using SymbolicUtils
using Latexify
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
        ssum = sum(w);
        @assert ssum > 0 "weights must have positive sum."
        W[i, :] = w ./ ssum  # normalize
    end

    return X, W
end

X, Y = build_sr_dataset(SAMPLES)
println("X dims: $(size(X)), Y dims: $(size(Y))")

"""
model = MultitargetSRRegressor(
    niterations = 100,
    binary_operators = [+, -, *, /],
    unary_operators = [],
    populations = 100,
    parallelism = :multithreading,
)
"""

square(x) = x^2
cube(x) = x^3

model = MultitargetSRRegressor(
    niterations = 300,
    populations = 100,
    binary_operators = [+, -, *, /],
    unary_operators = [square, cube],
    maxsize = 20,
    parsimony = 1e-3,
    parallelism = :multithreading,
)

mach = machine(model, X, Y)
fit!(mach)

r = report(mach)
n_targets = size(Y, 2)

for j = 1:n_targets
    best_expr = r.equations[j][r.best_idx[j]]

    println("\nTarget y[$j]")
    println("Expression: ", best_expr)

    eqn = node_to_symbolic(best_expr)

    println("Symbolic form: ", eqn)
    println("LaTeX: ", latexify(string(eqn)))
end

y_pred = predict(mach, X)

function mse(A, B)
    @assert size(A) == size(B)
    mean((A .- B) .^ 2)
end

mean_vec = mean(Y, dims = 1)
Y_mean = repeat(mean_vec, size(Y, 1), 1)

println("MSE(constant mean predictor) = ", mse(Y, Y_mean))
println("MSE(SR model)                = ", mse(Y, y_pred))


println("\nPredicted vs Actual weights per sample:")
@assert size(y_pred) == size(Y) "prediction matrix size does not match target size."

for i = 1:size(Y, 1)
    println("Sample $i:")
    println("Actual: $(Y[i, :])")
    println("Predicted: $(y_pred[i, :])")
    println()
end
