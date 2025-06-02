using Oscar 

R, (x, y) = polynomial_ring(QQ, [:x, :y]; internal_ordering=:lex)
f = x^2*y + x*y^2 + y^2
f1 = x*y - 1
f2 = y^2 - 1


# print(typeof(f), "\n")



function division_alg(f::QQMPolyRingElem, F::Vararg{QQMPolyRingElem})
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

q, r = division_alg(f, f1, f2)
println("Quotients: ", q)
print("Remainder: ", r)