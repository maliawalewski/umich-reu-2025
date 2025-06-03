using Nemo
using Groebner
using AbstractAlgebra

# 2.8 Example 1 (pg. 97 of Ideals, Varieties, and Algorithms)

C = CalciumField()
R, (x, y, z) = polynomial_ring(C, ["x", "y", "z"])
polynomials = [x*z - y^2, x^3 - z^2]
basis = groebner(polynomials, ordering=DegLex())

println("Groebner Basis: ")
for g in basis
    println(g)
end

# custom ordering should be the same as DegLex
weights = [[1, 1, 1], # weight all variables equally
           [3, 2, 1], # if tied, weight x > y > z
          ]

order = MatrixOrdering([x, y, z], weights)
custom_basis = groebner(polynomials, ordering=order)

println("Groebner Basis (custom ordering): ")
for g in custom_basis
    println(g)
end
