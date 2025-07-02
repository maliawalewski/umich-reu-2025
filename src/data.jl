using AbstractAlgebra
using Serialization

# data.jl generates synthetic polynomial ideals

function old_generate_ideal(;
    num_polynomials::Integer = 3,
    num_variables::Integer = 3,
    max_degree::Integer = 4,
    num_terms::Integer = 3,
    max_attempts::Integer = 100,
)
    @assert num_polynomials > 0 "num_polynomials must be greater than 0"
    @assert num_variables > 0 "num_variables must be greater than 0"
    @assert max_degree > 0 "max_degree must be greater than 0"
    @assert num_terms > 0 "num_terms must be greater than 0"

    field = GF(32003)
    ring, vars = polynomial_ring(field, ["x_" * string(i) for i = 1:num_variables])
    polynomials = Vector{typeof(vars[1])}()
    used_polys = Set{UInt64}()

    for _ = 1:num_polynomials
        p_attempts = 0
        while true
            used_exponents = Set{NTuple{num_variables,Int}}()
            terms = []
            for _ = 1:num_terms
                attempts = 0
                while true
                    exponents = rand(0:max_degree, num_variables)
                    expt_key = Tuple(exponents)
                    if !(expt_key in used_exponents)
                        push!(used_exponents, expt_key)
                        coeff = rand(field)
                        c_attempts = 0
                        while coeff == 0
                            coeff = rand(field)
                            c_attempts += 1
                            @assert c_attempts <= max_attempts "failed to generate a non-zero coefficient after $max_attempts attempts"
                        end
                        monomial =
                            coeff * prod(vars[i]^exponents[i] for i = 1:num_variables)
                        push!(terms, monomial)
                        break
                    end
                    attempts += 1
                    @assert attempts <= max_attempts "failed to generate a unique random monomial after $max_attempts attempts"
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
            @assert p_attempts <= max_attempts "failed to generate a unique random polynomial after $max_attempts attempts"
        end
    end

    return polynomials, vars
end

function old_generate_data(;
    num_ideals::Integer = 1000,
    num_polynomials::Integer = 3,
    num_variables::Integer = 3,
    max_degree::Integer = 4,
    num_terms::Integer = 3,
    max_attempts::Integer = 100,
)
    @assert num_ideals > 0 "num_ideals must be greater than 0"

    ideals = []
    variables = nothing
    for _ = 1:num_ideals
        ideal, vars = old_generate_ideal(
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

