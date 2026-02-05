from julia.api import Julia

Julia(compiled_modules=False)  # Must be first

from julia import Main
import numpy as np
import sympy as sp

Main.include("julia_methods.jl")

x, y, z = 3, 2, 1

p = f"groebner_basis(Dict(:x=>{x}, :y=>{y}, :z=>{z}))"

result = Main.eval(p)

print("Groebner basis:")
for r in result:
    print(r)
