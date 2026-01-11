using AbstractAlgebra
using Serialization
using Random

function old_generate_ideal(;
    rng::AbstractRNG,
    num_polynomials::Integer = 3,
    num_variables::Integer = 3,
    max_degree::Integer = 4,
    num_terms::Integer = 3,
    max_attempts::Integer = 100,
)
    @assert num_polynomials > 0
    @assert num_variables > 0
    @assert max_degree > 0
    @assert num_terms > 0

    field = GF(32003)
    ring, vars = polynomial_ring(field, ["x_" * string(i) for i = 1:num_variables])
    polynomials = Vector{typeof(vars[1])}()
    used_polys = Set{UInt64}()

    for _ = 1:num_polynomials
        p_attempts = 0
        while true
            used_exponents = Set{NTuple{num_variables,Int}}()
            terms = Any[]
            for _ = 1:num_terms
                attempts = 0
                while true
                    exponents = rand(rng, 0:max_degree, num_variables)
                    expt_key = Tuple(exponents)
                    if !(expt_key in used_exponents)
                        push!(used_exponents, expt_key)

                        coeff = rand(rng, field)
                        c_attempts = 0
                        while coeff == 0
                            coeff = rand(rng, field)
                            c_attempts += 1
                            @assert c_attempts <= max_attempts
                        end

                        monomial =
                            coeff * prod(vars[i]^exponents[i] for i = 1:num_variables)
                        push!(terms, monomial)
                        break
                    end
                    attempts += 1
                    @assert attempts <= max_attempts
                end
            end
            polynomial = sum(terms)
            poly_hash = hash(polynomial)
            if !(poly_hash in used_polys)
                push!(used_polys, poly_hash)
                push!(polynomials, polynomial)
                break
            end
            p_attempts += 1
            @assert p_attempts <= max_attempts
        end
    end

    return polynomials, vars
end

function old_generate_data(;
    rng::AbstractRNG,
    num_ideals::Integer = 1000,
    num_polynomials::Integer = 3,
    num_variables::Integer = 3,
    max_degree::Integer = 4,
    num_terms::Integer = 3,
    max_attempts::Integer = 100,
)
    @assert num_ideals > 0

    ideals = Vector{Any}()
    variables = nothing

    for _ = 1:num_ideals
        ideal, vars = old_generate_ideal(
            rng = rng,
            num_polynomials = num_polynomials,
            num_variables = num_variables,
            max_degree = max_degree,
            num_terms = num_terms,
            max_attempts = max_attempts,
        )
        variables = vars
        push!(ideals, ideal)
    end

    return ideals, variables
end

function n_site_phosphorylation_generate_ideal(;
    rng::AbstractRNG,
    num_variables::Integer = 2,
    num_polynomials::Integer = 2,
    num_terms::Integer,
    base_sets::Vector{Any},
    max_attempts::Integer = 100,
)
    @assert num_variables == 2 "n-site Eq.(2.6) generator requires num_variables == 2"
    @assert num_polynomials == 2 "n-site Eq.(2.6) generator requires num_polynomials == 2"
    @assert length(base_sets) == num_polynomials
    @assert length(base_sets[1]) == num_terms
    @assert base_sets[1] == base_sets[2] "Both polynomials must share the same support set (same ordering)"
    A = base_sets[1]

    @assert !isempty(A)
    @assert all(length(e) == 2 for e in A)
    @assert A[1] == [1, 0]
    @assert A[2] == [0, 1]
    @assert A[end] == [0, 0]

    middles = A[3:(end-1)]
    @assert all(e[1] == 1 && e[2] >= 1 for e in middles) "Middle exponents must be (1,j) with j>=1"
    js = sort!(unique(e[2] for e in middles))
    @assert js == collect(1:length(middles)) "Middle exponents must be consecutive (1..n)"
    n = length(middles)

    F = GF(32003)
    ring, vars = polynomial_ring(F, ["x_1", "x_2"])
    x1, x2 = vars

    rand_nonzero() = begin
        c = rand(rng, F)
        tries = 1
        while c == 0
            tries += 1
            @assert tries <= max_attempts
            c = rand(rng, F)
        end
        c
    end

    Stot = rand_nonzero()
    Etot = rand_nonzero()
    Ftot = rand_nonzero()

    T = [rand_nonzero() for _ = 1:n]
    K = [rand_nonzero() for _ = 1:n]

    beta = Vector{typeof(F(0))}(undef, n)
    for j = 0:(n-1)
        Tprev = (j == 0) ? F(1) : T[j]
        beta[j+1] = K[j+1] * Tprev / (Ftot ^ j)
    end

    row1 = Vector{typeof(F(0))}(undef, n + 3)
    row1[1] = F(1)
    row1[2] = F(0)
    for k = 1:n
        j = k - 1
        row1[2+k] = T[k] / (Ftot ^ (j + 1)) + beta[k]
    end
    row1[end] = -Stot

    row2 = Vector{typeof(F(0))}(undef, n + 3)
    row2[1] = F(0)
    row2[2] = F(1)
    for k = 1:n
        row2[2+k] = beta[k]
    end
    row2[end] = -Etot

    monoms = Any[]
    for e in A
        push!(monoms, (x1 ^ e[1]) * (x2 ^ e[2]))
    end

    f1 = zero(x1)
    f2 = zero(x1)
    for i in eachindex(monoms)
        f1 += row1[i] * monoms[i]
        f2 += row2[i] * monoms[i]
    end

    polynomials = Vector{typeof(vars[1])}()
    push!(polynomials, f1)
    push!(polynomials, f2)

    return polynomials, vars
end

@inline is_pad_term(e) = any(<(0), e)

function new_generate_ideal(;
    rng::AbstractRNG,
    num_variables::Integer = 3,
    num_polynomials::Integer = 3,
    num_terms::Integer = 3,
    base_sets::Vector{Any} = Vector{Any}(),
    max_attempts::Integer = 100,
)
    @assert num_variables > 0
    @assert length(base_sets) == num_polynomials

    field = GF(32003)
    ring, vars = polynomial_ring(field, ["x_" * string(i) for i = 1:num_variables])
    polynomials = Vector{typeof(vars[1])}()

    for base_set in base_sets
        terms = Any[]
        for e in base_set
            if is_pad_term(e)
                continue
            end
            coeff = rand(rng, field)
            c_attempts = 0
            while coeff == 0
                coeff = rand(rng, field)
                c_attempts += 1
                @assert c_attempts <= max_attempts
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
    rng::AbstractRNG,
    num_ideals::Integer = 1000,
    num_polynomials::Integer = 3,
    num_variables::Integer = 3,
    max_degree::Integer = 4,
    num_terms::Integer = 3,
    max_attempts::Integer = 100,
    base_sets::Union{Nothing,Vector{Any}} = nothing,
    base_set_path::Union{Nothing,String} = nothing,
    should_save_base_sets::Bool = false,
    use_n_site_phosphorylation_coeffs::Bool = false,
)
    @assert num_ideals > 0

    if base_sets === nothing
        base_sets = Any[]
        for _ = 1:num_polynomials
            used_exponents = Set{NTuple{num_variables,Int}}()
            base_set = Any[]
            for _ = 1:num_terms
                attempts = 0
                while true
                    exponents = rand(rng, 0:max_degree, num_variables)
                    expt_key = Tuple(exponents)
                    if !(expt_key in used_exponents)
                        push!(used_exponents, expt_key)
                        push!(base_set, exponents)
                        break
                    end
                    attempts += 1
                    @assert attempts <= max_attempts
                end
            end
            push!(base_sets, base_set)
        end
        if should_save_base_sets && base_set_path !== nothing
            save_base_sets(base_sets, base_set_path)
        end
    end

    ideals = Vector{Any}()
    variables = nothing

    for _ = 1:num_ideals
        if use_n_site_phosphorylation_coeffs
            ideal, vars = n_site_phosphorylation_generate_ideal(
                rng = rng,
                num_variables = num_variables,
                num_polynomials = num_polynomials,
                num_terms = num_terms,
                base_sets = base_sets,
                max_attempts = max_attempts,
            )
            variables = vars
            push!(ideals, ideal)
        else
            ideal, vars = new_generate_ideal(
                rng = rng,
                num_variables = num_variables,
                num_polynomials = num_polynomials,
                num_terms = num_terms,
                base_sets = base_sets,
                max_attempts = max_attempts,
            )
            variables = vars
            push!(ideals, ideal)
        end
    end

    return ideals, variables, base_sets
end

function save_base_sets(base_sets, path::String)
    serialize(path, base_sets)
end

function load_base_sets(path::String)
    return deserialize(path)
end
