using Groebner, AbstractAlgebra, Statistics, Dates
include("environment.jl")
include("data.jl")
include("model.jl")

function main()
    num_vars = 3
    delta_bound = 0.01f0
    env = init_environment(num_vars = num_vars, delta_bound = delta_bound)

    actor_struct, critic_struct = build_td3_model(env)

    replay_buffer = CircularBuffer{Transition}(CAPACITY)

    train_td3!(actor_struct, critic_struct, env, replay_buffer)
end

start_time = now()
main()
end_time = now()

elapsed_ms = Millisecond(end_time - start_time).value
elapsed_sec = elapsed_ms / 1000
println("Total runtime: $elapsed_sec seconds")
