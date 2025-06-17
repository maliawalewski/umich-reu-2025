using AbstractAlgebra

# data.jl generates synthetic polynomial ideals

function generate_ideal(;
    num_polynomials::Integer = 3,
    num_variables::Integer = 3,
    max_degree::Integer = 8,
    max_terms::Integer = 6,
)
    @assert num_polynomials > 0 "num_polynomials must be greater than 0"
    @assert num_variables > 0 "num_variables must be greater than 0"
    @assert max_degree > 0 "max_degree must be greater than 0"
    @assert max_terms > 0 "max_terms must be greater than 0"

    field = GF(32003)
    ring, vars = polynomial_ring(field, ["x_" * string(i) for i = 1:num_variables])

    polynomials = []

    for _ = 1:num_polynomials
        terms = []
        num_terms = rand(1:max_terms)
        for _ = 1:num_terms
            coeff = rand(field)
            exponents = rand(0:max_degree, num_variables)
            monomial = coeff * prod(vars[i]^exponents[i] for i = 1:num_variables)
            push!(terms, monomial)
        end
        push!(polynomials, sum(terms))
    end

    return polynomials
end

function generate_data(;
    num_ideals::Integer = 10000,
    num_polynomials::Integer = 3,
    num_variables::Integer = 3,
    max_degree::Integer = 8,
    max_terms::Integer = 6,
)
    @assert num_ideals > 0 "num_ideals must be greater than 0"

    ideals = []

    for _ = 1:num_ideals
        ideal = generate_ideal(
            num_polynomials = num_polynomials,
            num_variables = num_variables,
            max_degree = max_degree,
            max_terms = max_terms,
        )
        push!(ideals, ideal)
    end

    return ideals
end
