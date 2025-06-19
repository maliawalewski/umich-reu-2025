using AbstractAlgebra

# data.jl generates synthetic polynomial ideals

function generate_ideal(;
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

    println("type: ")
    println(typeof(vars[1]))

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
                        monomial =
                            rand(field) * prod(vars[i]^exponents[i] for i = 1:num_variables)
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

    # TO-DO: Figure out type casting and why we are getting error with Vector{Any}
    return convert(Vector{Any}, polynomials), convert(Vector{Any}, vars)
end

function generate_data(;
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
        ideal, vars = generate_ideal(
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
