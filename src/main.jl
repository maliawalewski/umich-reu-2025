using Groebner, AbstractAlgebra, Statistics, Dates
include("environment.jl")
include("data.jl")
include("utils.jl")
include("model.jl")

 
function main()

    # # TESTING FIXED IDEAL
    # field = GF(32003)
    # ring, (x, y, z) = polynomial_ring(field, ["x", "y", "z"])
    # ideal = [x^2 + y + z, x + x*y^2 + z^3, x^3*y + x*y + y*z^2]
    # # END TESTING FIXED IDEAL


    env = init_environment(
        num_vars = NUM_VARS,
        delta_bound = DELTA_BOUND,
        num_ideals = NUM_IDEALS,
        max_iterations = MAX_ITERATIONS,
        num_terms = NUM_TERMS,
        num_polys = NUM_POLYS,
    )

    actor_struct, critic_struct, replay_buffer = load_td3(env)

    # actor_struct, critic_struct = build_td3_model(env)

    # replay_buffer = CircularBuffer{Transition}(CAPACITY)
    # replay_buffer = PrioritizedReplayBuffer(CAPACITY, N_SAMPLES, ALPHA, BETA, BETA_INCREMENT, EPS)

    train_td3!(actor_struct, critic_struct, env, replay_buffer, ACTOR_LR, CRITIC_LR)
    
    # test_td3!(actor_struct, critic_struct, env)

end

start_time = now()
main()
end_time = now()

elapsed_ms = Millisecond(end_time - start_time).value 
elapsed_sec = elapsed_ms / 1000
println("Total runtime: $elapsed_sec seconds")
