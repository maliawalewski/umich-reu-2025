using Groebner, AbstractAlgebra, Statistics, Dates, ArgParse
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
        default = "FOUR_PT_BASE_SET"
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
    end 

    args = parse_args(s)
    
    env = init_environment(
        args = args,
        num_ideals = NUM_IDEALS,
        delta_bound = DELTA_BOUND,
        max_iterations = MAX_ITERATIONS,
        default_num_vars = DEFAULT_NUM_VARS,
        default_num_terms = DEFAULT_NUM_TERMS,
        default_num_polys = DEFAULT_NUM_POLYS,
    )

    actor_struct, critic_struct, replay_buffer = load_td3(env, args)
  
    train_td3!(actor_struct, critic_struct, env, replay_buffer, ACTOR_LR, CRITIC_LR, args)
    
    test_td3!(actor_struct, critic_struct, env, args)

    interpret_results()

end

start_time = now()
main()
end_time = now()

elapsed_ms = Millisecond(end_time - start_time).value 
elapsed_sec = elapsed_ms / 1000
println("Total runtime: $elapsed_sec seconds")
