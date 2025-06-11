using Nemo
using Groebner
using AbstractAlgebra
using Random 

function groebner_basis(order::AbstractDict)
    C = CalciumField()
    R, (x, y, z) = polynomial_ring(C, ["x", "y", "z"])

    polynomials = generate_polynomials(3)  

    # polynomials = [x*z - y^2, x^3 - z^2]
    println("Polynomials: ", polynomials)


    ord = WeightedOrdering(order)
    basis = groebner(polynomials, ordering=ord)
    return [string(g) for g in basis]
end


function generate_polynomials(n::Int)
    C = CalciumField()
    R, (x, y, z) = polynomial_ring(C, ["x", "y", "z"])

    polynomials = []
    for _ in 1:n
        poly = rand((1, 50)) * x^3 + rand((1, 50)) * y^2 + rand((1, 50)) * z^1
        push!(polynomials, poly)
    end
    
    return polynomials
end
