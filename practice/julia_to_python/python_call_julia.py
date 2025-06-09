from julia.api import Julia
Julia(compiled_modules=False)  # Must be first

from julia import Main
import numpy as np

# Define a Julia function
Main.eval("""
function add(x, y)
    return x + y
end
""")

print(Main.add(2, 5))  # Should print 7

# Linear system example
Main.eval('A = rand(3, 3)')
Main.b = Main.eval('A * ones(3)')
Main.eval('x = A \\ b')

# Check residual in Python
residual = np.linalg.norm(np.matmul(Main.A, Main.x) - Main.b)
print(residual)  # Should print 0.0 (or something extremely close)


