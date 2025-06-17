using Groebner 
using AbstractAlgebra

R, (x, y) = GF(2^31-1)["x", "y"]

trace, gb = groebner_learn([x*y^2 + x, y*x^2 + y], ordering=DegRevLex())
println(trace)

trace, gb = groebner_learn([x*y^2 + x, y*x^2 + y], ordering=DegLex())
println(trace)

trace, gb = groebner_learn([x*y^2 + x, y*x^2 + y], ordering=Lex())
println(trace)
