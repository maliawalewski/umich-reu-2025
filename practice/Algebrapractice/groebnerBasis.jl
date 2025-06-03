using Nemo
using Groebner
using AbstractAlgebra

C = CalciumField()
R, (x, y, z) = polynomial_ring(C, ["x", "y", "z"])
polynomials = [x*z - y^2, x^3 - z^2]
basis = groebner(polynomials, ordering=DegLex())

println("Groebner Basis: ")
for g in basis
    println(g)
end
