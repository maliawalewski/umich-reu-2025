using Polynomials

f = Polynomial([2, 3, 1])  
g = Polynomial([2,1]) 

function gcd(f::Polynomial, g::Polynomial)
    h = f
    s = g

    while degree(s) >= 0
        r = rem(h, s)
        h = s
        s = r
    end

    return h
    
end

println("gcd(f, g) = ", gcd(f, g))