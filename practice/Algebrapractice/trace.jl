using Groebner 
using AbstractAlgebra

R, (x, y) = GF(2^31-1)["x", "y"]

trace, gb = groebner_learn([x*y^2 + x, y*x^2 + y])

println(trace)
