using Groebner, AbstractAlgebra, Statistics, Dates
include("environment.jl")
include("data.jl")
include("model.jl")

# Environment parameters
NUM_VARS = 3
DELTA_BOUND = 0.1f0 # Max shift from current state
NUM_POLYS = NUM_VARS # For now, number of polynomials is equal to number of variables
NUM_IDEALS = 10 # Number of ideals per episode
NUM_TERMS = NUM_VARS # Number of terms in each polynomial
MAX_ITERATIONS = 5 # Maximum iterations per episode (i.e. steps per episode)

function main()
    # TESTING FIXED IDEAL
    field = GF(32003)
    ring, (x, y, z) = polynomial_ring(field, ["x", "y", "z"])
    ideal = [x^2 + y + z, x + x*y^2 + z^3, x^3*y + x*y + y*z^2]
    # END TESTING FIXED IDEAL

    trace, basis = groebner_learn(ideal, ordering = DegRevLex(x,y,z))
    println("Grevlex reward: ", reward(trace))

    env = init_environment(num_vars = NUM_VARS, delta_bound = DELTA_BOUND, num_ideals = NUM_IDEALS, max_iterations = MAX_ITERATIONS, num_terms = NUM_TERMS, num_polys = NUM_POLYS)

    actor_struct, critic_struct = build_td3_model(env)

    replay_buffer = CircularBuffer{Transition}(CAPACITY)

    train_td3!(actor_struct, critic_struct, env, replay_buffer, LR)
end

start_time = now()
main()
end_time = now()

elapsed_ms = Millisecond(end_time - start_time).value
elapsed_sec = elapsed_ms / 1000
println("Total runtime: $elapsed_sec seconds")
