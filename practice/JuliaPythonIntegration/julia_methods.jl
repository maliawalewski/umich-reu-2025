using Nemo
using Groebner
using AbstractAlgebra

function groebner_basis()
    C = CalciumField()
    R, (x, y, z) = polynomial_ring(C, ["x", "y", "z"])
    polynomials = [x*z - y^2, x^3 - z^2]
    basis = groebner(polynomials, ordering=DegLex())
    return [string(g) for g in basis]
end