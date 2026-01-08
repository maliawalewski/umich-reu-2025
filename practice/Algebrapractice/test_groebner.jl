using AbstractAlgebra
using Groebner

field = AbstractAlgebra.GF(32003)
ring, (x_1, x_2, x_3) = AbstractAlgebra.polynomial_ring(
    field,
    ["x_1", "x_2", "x_3"],
    internal_ordering = :degrevlex,
)

polynomials = [
    2235*x_1^3*x_2^3*x_3^3 +
    20367*x_1^3*x_2^2*x_3 +
    5551*x_1^3*x_2*x_3^2 +
    7832*x_1^3*x_3 +
    20688*x_1^2*x_2^2*x_3 +
    28422*x_1^2*x_2*x_3^2 +
    15174*x_1*x_2^3*x_3^3 +
    5783*x_1*x_2^3*x_3^2 +
    1257*x_1*x_2^2*x_3^2 +
    25395*x_1*x_2*x_3,
    16538*x_1^3*x_2^3*x_3^2 +
    21349*x_1^3*x_2^3 +
    11367*x_1^3*x_2^2*x_3^2 +
    12594*x_1^3*x_3^2 +
    23215*x_1^3 +
    28261*x_1^2*x_2^2*x_3^2 +
    2666*x_1^2*x_3^3 +
    31953*x_1^2*x_3^2 +
    4826*x_2 +
    18601*x_3,
    21337*x_1^3*x_2^3*x_3^3 +
    8247*x_1^3*x_2^3*x_3 +
    11846*x_1^3*x_2*x_3^2 +
    10697*x_1^2*x_2^2*x_3^3 +
    7549*x_1^2*x_2*x_3 +
    25332*x_1^2*x_3^2 +
    16367*x_1*x_2^3*x_3^3 +
    4083*x_1*x_2^2*x_3^3 +
    19978*x_2*x_3^2 +
    18374*x_3,
    30886*x_1^3*x_2^2*x_3^3 +
    21571*x_1^3*x_3^3 +
    20314*x_1^2*x_2*x_3^3 +
    542*x_1^2*x_3 +
    7906*x_1*x_2^3*x_3^3 +
    15557*x_1*x_2*x_3^2 +
    7709*x_2^3*x_3 +
    13023*x_2^2*x_3^3 +
    10083*x_2*x_3^3 +
    4451*x_3^2,
    7939*x_1^3*x_2^2*x_3^2 +
    19202*x_1^2*x_2^3 +
    28944*x_1*x_2^3*x_3^3 +
    1622*x_1*x_2^2*x_3^3 +
    4706*x_1*x_2^2 +
    25653*x_2^3*x_3^2 +
    23149*x_2^2*x_3^3 +
    6453*x_2*x_3^3 +
    13436*x_2 +
    6642*x_3^2,
]

# tie_polys = [x_1^2+x_1*x_2, x_3^2+x_1]

order = WeightedOrdering(x_1 => 2, x_2 => 2, x_3 => 1)


trace, basis = groebner_learn(polynomials, ordering = DegRevLex())
for p in basis
    println(p)
end


println(isgroebner(basis, ordering = DegRevLex()))

println()
