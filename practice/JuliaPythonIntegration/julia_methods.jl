using Nemo
using Groebner
using AbstractAlgebra


function groebner_basis(order::AbstractDict)
    C = CalciumField()
    R, (x, y, z) = polynomial_ring(C, ["x", "y", "z"])
    polynomials = [x*z - y^2, x^3 - z^2]
    ord = WeightedOrdering(order)
    basis = groebner(polynomials, ordering=ord)
    return [string(g) for g in basis]
end

