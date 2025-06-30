using Groebner, AbstractAlgebra, Statistics, Dates
include("environment.jl")
include("data.jl")
include("utils.jl")
include("model.jl") 


function main()
    env = init_environment(num_vars = NUM_VARS, delta_bound = DELTA_BOUND, num_ideals = NUM_IDEALS, max_iterations = MAX_ITERATIONS, num_terms = NUM_TERMS, num_polys = NUM_POLYS)

    actor_struct, critic_struct = build_td3_model(env)

    # replay_buffer = CircularBuffer{Transition}(CAPACITY)
    
    replay_buffer = PrioritizedReplayBuffer(CAPACITY, N_SAMPLES, ALPHA, BETA, BETA_INCREMENT, EPS)

    train_td3!(actor_struct, critic_struct, env, replay_buffer, LR)
end

start_time = now()
main()
end_time = now()

elapsed_ms = Millisecond(end_time - start_time).value
elapsed_sec = elapsed_ms / 1000
println("Total runtime: $elapsed_sec seconds")
