using Groebner 
using AbstractAlgebra

R, (x, y) = GF(2^31-1)["x", "y"]

trace, gb = groebner_learn([x*y^2 + x, y*x^2 + y], ordering=DegRevLex())

# trace, gb = groebner_learn([x*y^2 + x, y*x^2 + y], ordering=DegLex())
# println(trace)

trace, gb = groebner_learn([x*y^2 + x, y*x^2 + y], ordering=Lex())
# println(trace)

for (k, v) in trace.recorded_traces
    println(v.matrix_infos)
    for e in v.matrix_infos
        println(e[3])
    end
    
    println("degrees")
    for p in v.critical_pair_sequence 
        println(p[1])
    end
    println("pair count")
    for p in v.critical_pair_sequence
        println(p[2])
    end

    # println("Key: ", k, " (Type: ", typeof(k), ")")
    println("Value: ", v)
    println()
end
