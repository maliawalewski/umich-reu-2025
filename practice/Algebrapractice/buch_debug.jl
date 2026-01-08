using AbstractAlgebra, Nemo, Calcium

A = AbstractAlgebra.Generic.MPoly{CalciumFieldElem}
CC = CalciumField()
R, (x, y, z) = polynomial_ring(CC, ["x", "y", "z"], internal_ordering = :lex)

f1 = x^2 + y^2 + z^2 - 1
f2 = x^2 + z^2 - y
f3 = x - z

function is_numerically_zero(p::AbstractAlgebra.Generic.MPoly{CalciumFieldElem})
    if length(coefficients(p)) == 0 # AbstractAlgebra's representation of the zero polynomial
        return true
    end
    for c in coefficients(p)
        if !Calcium.contains_zero(c) # Key check for CalciumFieldElem
            return false
        end
    end
    return true # All coefficients are intervals containing zero
end

function division_alg(f::A, F::Vector{A})
    # Ensure R is the correct polynomial ring to create zero polynomials
    # (it should be globally defined as per your initial setup)
    r = R(0)
    q = fill(R(0), length(F))
    p = f

    while !is_numerically_zero(p) # Main loop condition uses the new zero check
        i = 1
        division_occurred = false

        while i <= length(F) && !division_occurred
            # Prevent errors if p became numerically zero due to a previous step but loop condition didn't catch it (unlikely but safe)
            if is_numerically_zero(p)
                break
            end

            mon_p = leading_monomial(p)
            coeff_p = leading_coefficient(p)
            mon_fi = leading_monomial(F[i])
            coeff_fi = leading_coefficient(F[i])

            if Calcium.contains_zero(coeff_fi) # Avoid division by a numerically zero leading coefficient
                i += 1
                continue
            end

            can_divide, mon_q = divides(mon_p, mon_fi)

            if can_divide
                coeff_q = coeff_p / coeff_fi
                q[i] += coeff_q * mon_q
                p -= coeff_q * mon_q * F[i] # Subtraction that might result in epsilon_poly
                division_occurred = true
            else
                i += 1
            end
        end

        if !division_occurred
            if is_numerically_zero(p) # If p is now zero (e.g. S-poly was epsilon), stop.
                break
            end
            r += leading_term(p)
            p -= leading_term(p)
        end
    end
    return q, r
end

"""
Buchberger's algorithm modified to use numerical zero checking and a pair list.
"""
function buchberger(F_input::Vector{A})
    # Filter out any initial polynomials that are numerically zero
    G = [p for p in F_input if !is_numerically_zero(p)]
    if isempty(G)
        return A[]
    end # Return empty list of type A

    pairs_to_process = Tuple{A,A}[] # Initialize list for pairs of polynomials
    for i = 1:length(G)
        for j = (i+1):length(G)
            push!(pairs_to_process, (G[i], G[j]))
        end
    end

    processed_idx = 0
    while processed_idx < length(pairs_to_process)
        processed_idx += 1
        p1, p2 = pairs_to_process[processed_idx]

        # println("Processing S-poly for pair: ", p1, " AND ", p2) # Optional debug
        S = S_polynomial(p1, p2)

        if is_numerically_zero(S)
            # println("S-polynomial is numerically zero.") # Optional debug
            continue
        end

        _, r = division_alg(S, G) # G is the current basis

        if !is_numerically_zero(r) # Critical check for the remainder
            # println("Non-zero remainder r: ", r) # Optional debug

            # Add new pairs involving the new basis element r to the END of the list
            for g_existing in G
                push!(pairs_to_process, (g_existing, r))
            end
            push!(G, r) # Add r to the basis G
            # println("Added to G. New size: ", length(G)) # Optional debug
            # else
            # println("Remainder r is numerically zero.") # Optional debug
        end
    end
    return G
end

function S_polynomial(f::A, g::A)
    # Ensure R is the correct polynomial ring (global)
    x_gamma = lcm(leading_monomial(f), leading_monomial(g))

    mon_f = leading_monomial(f)
    coeff_f = leading_coefficient(f)

    mon_g = leading_monomial(g)
    coeff_g = leading_coefficient(g)

    # Standard S-polynomial formula
    term1_factor = x_gamma / mon_f # This is a monomial (polynomial)
    term2_factor = x_gamma / mon_g # This is a monomial (polynomial)

    S = (term1_factor * (R(1) / coeff_f)) * f - (term2_factor * (R(1) / coeff_g)) * g
    return S
end

function reduced_basis(G::Vector{A})
    #first condition
    monic_G = [p / leading_coefficient(p) for p in G]

    reduced_G = []
    #second condition
    for i = 1:length(monic_G)
        p = monic_G[i]
        G_others = [monic_G[j] for j = 1:length(monic_G) if j != i] # G \ {p}

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
