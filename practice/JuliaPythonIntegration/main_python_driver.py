from julia.api import Julia
Julia(compiled_modules=False)  # Must be first

from julia import Main
import numpy as np
import sympy as sp

# # Define a Julia function
# Main.eval("""
# function add(x, y)
#     return x + y
# end
# """)

# print(Main.add(2, 5))  # Should print 7

# # Linear system example
# Main.eval('A = rand(3, 3)')
# Main.b = Main.eval('A * ones(3)')
# Main.eval('x = A \\ b')

# # Check residual in Python
# residual = np.linalg.norm(np.matmul(Main.A, Main.x) - Main.b)
# print(residual)  # Should print 0.0 (or something extremely close)


# Main.eval("""
# using Nemo
# using Groebner
# using AbstractAlgebra

# function groebner_basis()
#     C = CalciumField()
#     R, (x, y, z) = polynomial_ring(C, ["x", "y", "z"])
#     polynomials = [x*z - y^2, x^3 - z^2]
#     basis = groebner(polynomials, ordering=DegLex())
#     return [string(g) for g in basis]
# end
# """)

Main.include("julia_method.jl")

result = Main.groebner_basis()
print("Groebner basis:")
for r in result:
    print(r)




