using Statistics
using Serialization
using DataFrames
using CSV
using Dates
using Random
using LinearAlgebra

include("environment.jl")
include("utils.jl")
include("basesets.jl")

BASE_DIR = @__DIR__
DATA_DIR = joinpath(BASE_DIR, "data")
WEIGHTS_DIR = joinpath(BASE_DIR, "weights")
RESULTS_DIR = joinpath(BASE_DIR, "results")

for d in (DATA_DIR, WEIGHTS_DIR, RESULTS_DIR)
    isdir(d) || mkpath(d)
end

# Groebner.jl determinism parameters
GROEBNER_MONOMS = :dense
GROEBNER_HOMOGENIZE = :no

DEGLEX_CACHE_EVERY = 1
CSV_FLUSH_EVERY_EPISODES = 50

# Environment parameters
DEFAULT_NUM_VARS = 3
DEFAULT_NUM_TERMS = 6
DEFAULT_NUM_POLYS = 3
DEFAULT_MAX_DEGREE = 4

DELTA_BOUND = 0.1f0
NUM_IDEALS = 10
MAX_ITERATIONS = 25
EPISODES = 10_000

# SA parameters
T_INIT = 1000.0
T_MIN = 0.1
STD = 0.002

MAX_ATTEMPTS = 100
NUM_TEST_IDEALS = 100_000
TEST_BATCH_SIZE = 100

function run_baseline!(
    method::String,
    env::Environment,
    args::Dict{String,Any},
    rng_data::AbstractRNG,
    rng_policy::AbstractRNG,
    rng_test::AbstractRNG,
    rng_cal::AbstractRNG
)
    run_tag = method * "_run_" * "baseset_" * string(args["baseset"]) * "_seed_" * string(args["seed"])
    
    train_steps_csv = joinpath(RESULTS_DIR, run_tag * "_train_agent_metrics.csv")
    train_episode_csv = joinpath(RESULTS_DIR, run_tag * "_train_baseline_metrics.csv")
    base_set_path = joinpath(DATA_DIR, run_tag * "_base_sets.bin")
    
    train_steps_df = DataFrame(
        global_timestep = Int[],
        episode = Int[],
        step_in_episode = Int[],
        raw_reward = Float64[],
        delta_vs_grevlex_reward = Float64[],
        agent_batch_time_s = Float64[],
        agent_mean_time_s = Float64[]
    )

    train_episode_df = DataFrame(
        episode = Int[],
        grevlex_mean_reward = Float64[],
        grevlex_mean_time_s = Float64[],
        deglex_mean_reward = Float64[],
        deglex_mean_time_s = Float64[]
    )
    
    base_sets = isfile(base_set_path) ? load_base_sets(base_set_path) : nothing
    if args["baseset"] == "N_SITE_PHOSPHORYLATION_BASE_SET"
        base_sets = N_SITE_PHOSPHORYLATION_BASE_SET
    elseif args["baseset"] == "RELATIVE_POSE_BASE_SET"
        base_sets = RELATIVE_POSE_BASE_SET
    elseif args["baseset"] == "TRIANGULATION_BASE_SET"
        base_sets = TRIANGULATION_BASE_SET
    elseif args["baseset"] == "WNT_BASE_SET"
        base_sets = WNT_BASE_SET
    elseif args["baseset"] == "DEFAULT"
        max_degree = DEFAULT_MAX_DEGREE
    else
        error("Unknown baseset: $(args["baseset"])")
    end

    is_n_site = args["baseset"] == "N_SITE_PHOSPHORYLATION_BASE_SET"
    if args["baseset"] != "DEFAULT"
        max_degree = max_total_degree(base_sets)
        base_sets, max_terms = pad_base_set(base_sets; max_terms = env.num_terms, num_vars = env.num_vars)
    end

    ideals, vars, monomial_matrix = new_generate_data(
        rng = rng_data,
        num_ideals = EPISODES * NUM_IDEALS,
        num_polynomials = env.num_polys,
        num_variables = env.num_vars,
        max_degree = max_degree,
        num_terms = env.num_terms,
        max_attempts = MAX_ATTEMPTS,
        base_sets = base_sets,
        base_set_path = base_set_path,
        should_save_base_sets = base_sets === nothing,
        use_n_site_phosphorylation_coeffs = is_n_site,
    )
    env.variables = vars
    env.monomial_matrix = monomial_matrix

    reset_env!(env)
    
    global_timestep = 0
    current_best_weight = init_state(rng_policy, env.num_vars)
    current_best_reward = -Inf
    
    T_decay = (T_INIT / T_MIN)^(1.0 / (EPISODES * MAX_ITERATIONS))
    T = T_INIT
    
    for i = 1:EPISODES
        reset_env!(env)
        start_idx = (i - 1) * NUM_IDEALS + 1
        end_idx = i * NUM_IDEALS
        env.ideal_batch = ideals[start_idx:end_idx]
        
        compute_deglex = ((i % DEGLEX_CACHE_EVERY) == 0)
        precompute_baselines!(env; compute_deglex = compute_deglex)
        
        push!(train_episode_df, (
            i,
            mean(env.grevlex_reward_cache),
            mean(env.grevlex_time_cache_s),
            compute_deglex ? mean(env.deglex_reward_cache) : NaN,
            compute_deglex ? mean(env.deglex_time_cache_s) : NaN,
        ))
        
        # At start of new batch, re-evaluate current_best_weight to get accurate current reward
        act!(env, current_best_weight, true)
        current_best_reward = env.reward
        
        done = false
        while !done
            global_timestep += 1
            
            if method == "rs"
                # Random Search: independent random sample
                action = init_state(rng_policy, env.num_vars)
            elseif method == "sa"
                # Simulated Annealing: perturb current best
                noise = randn(rng_policy, env.num_vars) .* STD
                action = clamp.(current_best_weight .+ noise, 1e-6, 1.0)
                action = action ./ sum(action)
            else
                error("Unknown method: $method")
            end
            
            action = Float32.(action)
            act!(env, action, true)
            r = env.reward
            
            # Acceptance logic
            if method == "rs"
                if r > current_best_reward
                    current_best_reward = r
                    current_best_weight = action
                end
            elseif method == "sa"
                if r > current_best_reward
                    current_best_reward = r
                    current_best_weight = action
                else
                    if rand(rng_policy) < exp((r - current_best_reward) / T)
                        current_best_reward = r
                        current_best_weight = action
                    end
                end
                T = max(T_MIN, T / T_decay)
            end
            
            push!(train_steps_df, (
                global_timestep,
                i,
                env.iteration_count,
                env.last_raw_reward,
                env.last_delta_reward,
                env.last_agent_batch_time_s,
                env.last_agent_mean_time_s,
            ))
            
            done = is_terminated(env)
        end
        
        if i % CSV_FLUSH_EVERY_EPISODES == 0
            CSV.write(train_steps_csv, train_steps_df; append=isfile(train_steps_csv))
            CSV.write(train_episode_csv, train_episode_df; append=isfile(train_episode_csv))
            empty!(train_steps_df)
            empty!(train_episode_df)
        end
    end
    
    CSV.write(train_steps_csv, train_steps_df; append=isfile(train_steps_csv), writeheader=!isfile(train_steps_csv))
    CSV.write(train_episode_csv, train_episode_df; append=isfile(train_episode_csv), writeheader=!isfile(train_episode_csv))
    
    # Testing phase
    test_baseline!(method, env, args, current_best_weight, rng_test, rng_cal, base_sets, base_set_path, is_n_site, max_degree)
end

function test_baseline!(method, env, args, best_weight, rng_test, rng_cal, base_sets, base_set_path, is_n_site, max_degree)
    run_tag = method * "_run_" * "baseset_" * string(args["baseset"]) * "_seed_" * string(args["seed"])
    test_csv = joinpath(RESULTS_DIR, run_tag * "_test_metrics.csv")
    order_csv = joinpath(RESULTS_DIR, run_tag * "_final_agent_weight_vector.csv")
    
    ideals_cal, vars_cal, monomial_matrix_cal = new_generate_data(
        rng = rng_cal,
        num_ideals = TEST_BATCH_SIZE,
        num_polynomials = env.num_polys,
        num_variables = env.num_vars,
        max_degree = max_degree,
        num_terms = env.num_terms,
        max_attempts = MAX_ATTEMPTS,
        base_sets = base_sets,
        base_set_path = base_set_path,
        should_save_base_sets = false,
        use_n_site_phosphorylation_coeffs = is_n_site,
    )
    env.variables = vars_cal
    
    # Final weight vector 
    CSV.write(order_csv, DataFrame(weight = best_weight))
    serialize(joinpath(RESULTS_DIR, run_tag * "_final_agent_order.bin"), best_weight)
    
    ideals, vars, monomial_matrix = new_generate_data(
        rng = rng_test,
        num_ideals = NUM_TEST_IDEALS,
        num_polynomials = env.num_polys,
        num_variables = env.num_vars,
        max_degree = max_degree,
        num_terms = env.num_terms,
        max_attempts = MAX_ATTEMPTS,
        base_sets = base_sets,
        base_set_path = base_set_path,
        should_save_base_sets = false,
        use_n_site_phosphorylation_coeffs = is_n_site,
    )
    env.variables = vars
    
    agent_rewards = Float64[]
    deglex_rewards = Float64[]
    grevlex_rewards = Float64[]
    
    test_df = DataFrame(
        idx = Int[],
        agent_reward = Float64[],
        agent_time_s = Float64[],
        deglex_reward = Float64[],
        deglex_time_s = Float64[],
        grevlex_reward = Float64[],
        grevlex_time_s = Float64[],
        agent_minus_grevlex_reward = Float64[],
        agent_time_ratio_vs_grevlex = Float64[],
        agent_minus_deglex_reward = Float64[],
        agent_time_ratio_vs_deglex = Float64[]
    )
    
    println("Testing baseline $method on $(NUM_TEST_IDEALS) ideals...")
    for (idx, ideal) in enumerate(ideals)
        reset_env!(env)
        env.ideal_batch = [ideal]
        
        precompute_baselines!(env; compute_deglex = true)
        g_rew = env.grevlex_reward_cache[1]
        g_time = env.grevlex_time_cache_s[1]
        d_rew = env.deglex_reward_cache[1]
        d_time = env.deglex_time_cache_s[1]
        
        act!(env, Float32.(best_weight), false)
        a_rew = env.reward
        a_time = env.last_agent_batch_time_s
        
        push!(agent_rewards, a_rew)
        push!(grevlex_rewards, g_rew)
        push!(deglex_rewards, d_rew)
        
        push!(test_df, (
            idx,
            a_rew, a_time,
            d_rew, d_time,
            g_rew, g_time,
            a_rew - g_rew,
            a_time / max(g_time, 1e-9),
            a_rew - d_rew,
            a_time / max(d_time, 1e-9)
        ))
        
        if idx % 1000 == 0
            println("Tested $idx / $NUM_TEST_IDEALS")
        end
    end
    
    CSV.write(test_csv, test_df)
    serialize(joinpath(RESULTS_DIR, run_tag * "_test_agent_rewards.bin"), agent_rewards)
    serialize(joinpath(RESULTS_DIR, run_tag * "_test_deglex_rewards.bin"), deglex_rewards)
    serialize(joinpath(RESULTS_DIR, run_tag * "_test_grevlex_rewards.bin"), grevlex_rewards)
end

using ArgParse

function main()
    LinearAlgebra.BLAS.set_num_threads(1)
    
    s = ArgParseSettings()
    @add_arg_table s begin
        "--method"
        help = "rs or sa"
        arg_type = String
        default = "sa"
        "--baseset"
        help = "Name of ideal baseset"
        arg_type = String
        default = "N_SITE_PHOSPHORYLATION_BASE_SET"
        "--seed"
        arg_type = Int
        default = 0
    end
    
    args = parse_args(s)
    
    seed = args["seed"]
    Random.seed!(seed)
    
    rng_data = MersenneTwister(seed + 1)
    rng_policy = MersenneTwister(seed + 2)
    rng_test = MersenneTwister(seed + 4)
    rng_env = MersenneTwister(seed + 5)
    rng_cal = MersenneTwister(seed + 7)
    groebner_seed = seed + 6
    
    env = init_environment(
        args = args,
        num_ideals = NUM_IDEALS,
        delta_bound = DELTA_BOUND,
        max_iterations = MAX_ITERATIONS,
        default_num_vars = DEFAULT_NUM_VARS,
        default_num_terms = DEFAULT_NUM_TERMS,
        default_num_polys = DEFAULT_NUM_POLYS,
        rng = rng_env,
        groebner_seed = groebner_seed,
    )
    
    run_baseline!(args["method"], env, args, rng_data, rng_policy, rng_test, rng_cal)
end

main()
