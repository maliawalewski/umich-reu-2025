using Groebner, AbstractAlgebra, Statistics, Dates, ArgParse, Random
include("environment.jl")
include("data.jl")
include("utils.jl")
include("model.jl")
include("results.jl")

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--baseset", "--supportset"
        dest_name = "baseset"
        help = "Name of ideal baseset to use. It should be a variable from basesets.jl or DEFAULT"
        arg_type = String
        default = "N_SITE_PHOSPHORYLATION_BASE_SET"
        "--LSTM", "--lstm", "--use_lstm", "--use_LSTM"
        dest_name = "lstm"
        help = "[true] => use an LSTM for the actor [false] => use a standard feed-forward neural network for the actor."
        arg_type = Bool
        default = false
        "--PER", "--prioritized_replay", "--per", "--use_PER", "--use_per"
        dest_name = "per"
        help = "[true] => use a prioritized experience replay buffer [false] => use a uniform sampling replay buffer"
        arg_type = Bool
        default = true
        "--seed"
        dest_name = "seed"
        help = "Master RNG seed for reproducibility."
        arg_type = Int
        default = 0
    end

    args = parse_args(s)

    seed = args["seed"]
    Random.seed!(seed)

    rng_data = MersenneTwister(seed + 1)
    rng_policy = MersenneTwister(seed + 2)
    rng_buffer = MersenneTwister(seed + 3)
    rng_test = MersenneTwister(seed + 4)
    rng_env = MersenneTwister(seed + 5)


    env = init_environment(
        args = args,
        num_ideals = NUM_IDEALS,
        delta_bound = DELTA_BOUND,
        max_iterations = MAX_ITERATIONS,
        default_num_vars = DEFAULT_NUM_VARS,
        default_num_terms = DEFAULT_NUM_TERMS,
        default_num_polys = DEFAULT_NUM_POLYS,
        rng = rng_env,
    )

    actor_struct, critic_struct, replay_buffer = load_td3(env, args)

    train_td3!(
        actor_struct,
        critic_struct,
        env,
        replay_buffer,
        ACTOR_LR,
        CRITIC_LR,
        args,
        rng_data,
        rng_policy,
        rng_buffer,
        rng_env,
    )

    test_td3!(actor_struct, critic_struct, env, args, rng_test, rng_env)

    interpret_results(args)

end

start_time = now()
main()
end_time = now()

elapsed_ms = Millisecond(end_time - start_time).value
elapsed_sec = elapsed_ms / 1000
println("Total runtime: $elapsed_sec seconds")
