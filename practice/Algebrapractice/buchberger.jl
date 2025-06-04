using AbstractAlgebra, Nemo

A = AbstractAlgebra.Generic.MPoly{CalciumFieldElem}
CC = CalciumField()
R, (x, y, z) = polynomial_ring(CC, ["x", "y", "z"], internal_ordering=:lex)

f1 = x^2 + y^2 + z^2 - 1
f2 = x^2 + z^2 - y
f3 = x - z

function buchberger(F::Vector{A})

    G = F
    G_prime = copy(G)
    changed = true

    while changed
        changed = false
        G_prime = copy(G)
        
        println("In the loop")
        for i in 1:length(G_prime)
            for j in i+1:length(G_prime)
                if G_prime[i] != G_prime[j]
                    S = S_polynomial(G_prime[i], G_prime[j])
                    r = division_alg(S, G_prime)[2]
                    if r != 0
                        push!(G, r)
                        changed = true
                        println("Added r: ", r)
                        println("Size of G: ", length(G))
                    end 
                end
            end
        end
    end
    return G
end 

function division_alg(f::A, F::Vector{A})
    r = R(0)
    q = fill(R(0), length(F))
    p = f

    while p != 0
        i = 1
        division_occurred = false 

        while i <= length(F) && division_occurred == false 
            mon_p = leading_monomial(p)
            coeff_p = leading_coefficient(p)
            mon_fi = leading_monomial(F[i])
            coeff_fi = leading_coefficient(F[i])

            if divides(mon_p, mon_fi)[1]
                mon_q = divides(mon_p, mon_fi)[2]
                coeff_q = coeff_p / coeff_fi
                q[i] += coeff_q * mon_q
                p -= coeff_q * mon_q * F[i]
                division_occurred = true
            else
                i += 1
            end
        end

        if division_occurred == false 
            r += leading_term(p)
            p -= leading_term(p)
        end

    end
    return q, r
end

function S_polynomial(f::A, g::A)
    x_gamma = lcm(leading_monomial(f), leading_monomial(g))

    mon_f = leading_monomial(f)
    coeff_f = leading_coefficient(f)

    mon_g = leading_monomial(g)
    coeff_g = leading_coefficient(g)

    S = (x_gamma / mon_f) * (1/coeff_f) * f - (x_gamma / mon_g) * (1/coeff_g) * g
    return S

end

function reduced_basis(G::Vector{A})
    #first condition
    monic_G = [p / leading_coefficient(p) for p in G]

    reduced_G = []
    #second condition
    for i in 1:length(monic_G)
        p = monic_G[i]
        G_others = [monic_G[j] for j in 1:length(monic_G) if j != i] # G \ {p}

        _, r = division_alg(p, G_others)

        if r != 0 
            push!(reduced_G, r)
        end
    end 
    return reduced_G
end

F = [f1, f2, f3]

basis = buchberger(F)
# basis = reduced_basis(collect(basis))


for b in basis
    println("Basis element: ", b)
end


