using Oscar

R, (x, y, z) = polynomial_ring(QQ, [:x, :y, :z]; internal_ordering=:lex)

polynomials = [x*z - y^2, x^3 - z^2]

I = ideal(R, polynomials)
order = matrix_ordering(R, [-1 1 1; 1 1 1; 1 1 1], check = false)
println("Is global? ", is_global(order))

basis = groebner_basis(I, ordering = order)

print(basis)