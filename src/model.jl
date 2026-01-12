using Flux
using DataStructures
using ReinforcementLearning
using Statistics
using Serialization
using Optimisers
using Plots
using LinearAlgebra
using BSON
using DataFrames
using CSV
using Dates
using Random
include("environment.jl")
include("utils.jl")
include("basesets.jl")
# plotlyjs()

BASE_DIR = @__DIR__
DATA_DIR = joinpath(BASE_DIR, "data")
WEIGHTS_DIR = joinpath(BASE_DIR, "weights")
RESULTS_DIR = joinpath(BASE_DIR, "results")
PLOTS_DIR = joinpath(BASE_DIR, "plots")

for d in (DATA_DIR, WEIGHTS_DIR, RESULTS_DIR, PLOTS_DIR)
    isdir(d) || mkpath(d)
end

# Groebner.jl determinism parameters (see: https://sumiya11.github.io/Groebner.jl/interface/#Groebner.groebner_learn)
GROEBNER_MONOMS = :dense
GROEBNER_HOMOGENIZE = :no

# Logging parameters
DEGLEX_CACHE_EVERY = 1 # how often to record DEGLEX Reward
CSV_FLUSH_EVERY_EPISODES = 50 # how often to write to CSV of results

# Environment parameters
# if a baseset is not passed to environment we have to fall back to generating random polynomials
DEFAULT_NUM_VARS = 3
DEFAULT_NUM_TERMS = 6
DEFAULT_NUM_POLYS = 3
DEFAULT_MAX_DEGREE = 4

DELTA_BOUND = 0.1f0 # Max shift from current state
NUM_IDEALS = 10 # Number of ideals per episode
MAX_ITERATIONS = 25 # Maximum iterations per episode (i.e. steps per episode)

# TD3 parameters
EPISODES = 10_000
GAMMA = 0.99 # Discount factor
TAU = 0.05 # Soft update parameter
ACTOR_LR = 1e-4 # Learning rate for actor and critics
ACTOR_MIN_LR = 1e-5 # Minimum Learning Rate
ACTOR_LR_DECAY = (ACTOR_LR - ACTOR_MIN_LR) / (EPISODES - (EPISODES / 10))
CRITIC_LR = 1e-4
CRITIC_MIN_LR = 1e-6
CRITIC_LR_DECAY = (CRITIC_LR - CRITIC_MIN_LR) / (EPISODES - (EPISODES / 10))
STD = 0.002 # Standard deviation for exploration noise
D = 100 # Update frequency for target actor and critics 
SAVE_WEIGHTS = 100

# Prioritized Experience Replay Buffer parameters
CAPACITY = 1_000_000
N_SAMPLES = 100
ALPHA = 0.6f0
BETA = 0.4f0
BETA_INCREMENT = 0.0001f0
EPS = 0.01f0

# Data parameters (used to generate ideal batch)
MAX_ATTEMPTS = 100

# test model
NUM_TEST_IDEALS = 100_000
TEST_BATCH_SIZE = 100

# NN parameters
CRITIC_HIDDEN_WIDTH = 512
ACTOR_HIDDEN_WIDTH = 512
ACTOR_DEPTH = 2 # Number of LSTM layers

struct Actor
    actor::Flux.Chain
    actor_target::Flux.Chain
    actor_opt_state::Any
end

struct Critics
    critic_1::Flux.Chain
    critic_2::Flux.Chain

    critic_1_target::Flux.Chain
    critic_2_target::Flux.Chain

    critic_1_opt_state::Any
    critic_2_opt_state::Any
end

function init_actor(actor::Flux.Chain, actor_target::Flux.Chain, actor_opt_state::Any)
    return Actor(actor, actor_target, actor_opt_state)
end

function init_critics(
    critic_1::Flux.Chain,
    critic_2::Flux.Chain,
    critic_1_target::Flux.Chain,
    critic_2_target::Flux.Chain,
    critic_1_opt_state::Any,
    critic_2_opt_state::Any,
)
    return Critics(
        critic_1,
        critic_2,
        critic_1_target,
        critic_2_target,
        critic_1_opt_state,
        critic_2_opt_state,
    )
end

function build_td3_model(env::Environment, args::Dict{String,Any})
    if args["lstm"]
        # LSTM layer actor
        actor_layers = Any[LSTM(
            ((env.num_vars * env.num_terms) + 1) * env.num_polys => ACTOR_HIDDEN_WIDTH,
        )]
        for l = 1:(ACTOR_DEPTH-1)
            layer = LSTM(ACTOR_HIDDEN_WIDTH => ACTOR_HIDDEN_WIDTH)
            push!(actor_layers, layer)
        end
        push!(actor_layers, Dense(ACTOR_HIDDEN_WIDTH, env.num_vars, sigmoid))
        actor = Flux.Chain(actor_layers...)
    else
        # Original dense layer actor
        actor = Flux.Chain(
            Dense(
                ((env.num_vars * env.num_terms) + 1) * env.num_polys,
                ACTOR_HIDDEN_WIDTH,
                relu,
            ),
            Dense(ACTOR_HIDDEN_WIDTH, ACTOR_HIDDEN_WIDTH, relu),
            Dense(ACTOR_HIDDEN_WIDTH, ACTOR_HIDDEN_WIDTH, relu),
            Dense(ACTOR_HIDDEN_WIDTH, env.num_vars),
            softmax,
        )
    end

    critic_1 = Flux.Chain(
        Dense(
            ((env.num_vars * env.num_terms) + 1) * env.num_polys + env.num_vars,
            CRITIC_HIDDEN_WIDTH,
            relu,
        ),
        Dense(CRITIC_HIDDEN_WIDTH, CRITIC_HIDDEN_WIDTH, relu),
        Dense(CRITIC_HIDDEN_WIDTH, CRITIC_HIDDEN_WIDTH, relu),
        Dense(CRITIC_HIDDEN_WIDTH, 1),
    )
    critic_2 = Flux.Chain(
        Dense(
            ((env.num_vars * env.num_terms) + 1) * env.num_polys + env.num_vars,
            CRITIC_HIDDEN_WIDTH,
            relu,
        ),
        Dense(CRITIC_HIDDEN_WIDTH, CRITIC_HIDDEN_WIDTH, relu),
        Dense(CRITIC_HIDDEN_WIDTH, CRITIC_HIDDEN_WIDTH, relu),
        Dense(CRITIC_HIDDEN_WIDTH, 1),
    )

    actor_target = deepcopy(actor)
    critic_1_target = deepcopy(critic_1)
    critic_2_target = deepcopy(critic_2)

    actor_opt = ADAM(ACTOR_LR)
    # actor_opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(10), Optimisers.ADAM(LR))
    actor_opt_state = Flux.setup(actor_opt, actor)

    critic_1_opt = ADAM(CRITIC_LR)
    critic_2_opt = ADAM(CRITIC_LR)
    # critic_1_opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(10), Optimisers.ADAM(LR))
    # critic_2_opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(10), Optimisers.ADAM(LR))

    critic_1_opt_state = Flux.setup(critic_1_opt, critic_1)
    critic_2_opt_state = Flux.setup(critic_2_opt, critic_2)

    actor_struct = Actor(actor, actor_target, actor_opt_state)
    critic_struct = Critics(
        critic_1,
        critic_2,
        critic_1_target,
        critic_2_target,
        critic_1_opt_state,
        critic_2_opt_state,
    )

    return actor_struct, critic_struct
end

function train_td3!(
    actor::Actor,
    critic::Critics,
    env::Environment,
    replay_buffer::Union{PrioritizedReplayBuffer,CircularBuffer{Transition}},
    initial_actor_lr::Float64,
    initial_critic_lr::Float64,
    args::Dict{String,Any},
    rng_data::AbstractRNG,
    rng_policy::AbstractRNG,
    rng_buffer::AbstractRNG,
    rng_env::AbstractRNG,
)

    losses = []
    rewards = []
    actions_taken = []
    losses_1 = []
    losses_2 = []

    run_tag =
        "td3_run_" * "baseset_" * string(args["baseset"]) * "_seed_" * string(args["seed"])

    train_steps_csv = joinpath(RESULTS_DIR, run_tag * "_train_agent_metrics.csv")
    train_episode_csv = joinpath(RESULTS_DIR, run_tag * "_train_baseline_metrics.csv")
    train_updates_csv = joinpath(RESULTS_DIR, run_tag * "_train_losses.csv")

    base_set_path = joinpath(DATA_DIR, run_tag * "_base_sets.bin")
    checkpoint_path = joinpath(WEIGHTS_DIR, run_tag * "_td3_checkpoint.bson")
    actor_plot_path = joinpath(PLOTS_DIR, run_tag * "_train_actor_loss_plot.png")
    reward_plot_path = joinpath(PLOTS_DIR, run_tag * "_train_reward_plot.png")
    critics_plot_path = joinpath(PLOTS_DIR, run_tag * "_train_critics_loss_plot.png")

    train_steps_df = DataFrame(
        global_timestep = Int[],
        episode = Int[],
        step_in_episode = Int[],
        raw_reward = Float64[],
        delta_vs_grevlex_reward = Float64[],
        agent_batch_time_s = Float64[],
        agent_mean_time_s = Float64[],
    )


    train_episode_df = DataFrame(
        episode = Int[],
        grevlex_mean_reward = Float64[],
        grevlex_mean_time_s = Float64[],
        deglex_mean_reward = Float64[],
        deglex_mean_time_s = Float64[],
    )

    train_updates_df = DataFrame(
        global_timestep = Int[],
        critic1_loss = Float64[],
        critic2_loss = Float64[],
        actor_loss = Float64[],
    )

    current_actor_lr = initial_actor_lr
    current_critic_lr = initial_critic_lr

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
        base_sets, max_terms =
            pad_base_set(base_sets; max_terms = env.num_terms, num_vars = env.num_vars)
        @assert max_terms == env.num_terms
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
    println("Monomial_matrix: ", env.monomial_matrix)

    @show env.num_vars env.num_terms env.num_polys
    @show size(actor.actor[1].weight)  # (hidden, expected_in_dim)
    @show ((env.num_vars * env.num_terms) + 1) * env.num_polys

    ideal0 = ideals[1]
    timing_warmup_all!(actor, critic, env, ideal0, vars, rng_policy; do_backprop = true)
    println("Timing warmup complete.")

    if args["lstm"]
        Flux.reset!(actor.actor)
        Flux.reset!(actor.actor_target)
    end

    reset_env!(env)
    env.ideal_batch = ideals[1:NUM_IDEALS]
    precompute_baselines!(env; compute_deglex = true)

    global_timestep = 0

    for i = 1:EPISODES
        reset_env!(env)

        if args["lstm"]
            Flux.reset!(actor.actor)
            Flux.reset!(actor.actor_target)
        end

        # fill_ideal_batch(env, env.num_polys, MAX_DEGREE, MAX_ATTEMPTS) # fill with random ideals

        start_idx = (i - 1) * NUM_IDEALS + 1
        end_idx = i * NUM_IDEALS
        env.ideal_batch = ideals[start_idx:end_idx]

        compute_deglex = ((i % DEGLEX_CACHE_EVERY) == 0)
        precompute_baselines!(env; compute_deglex = compute_deglex)

        grevlex_mean_reward = mean(env.grevlex_reward_cache)
        grevlex_mean_time_s = mean(env.grevlex_time_cache_s)

        deglex_mean_reward = compute_deglex ? mean(env.deglex_reward_cache) : NaN
        deglex_mean_time_s = compute_deglex ? mean(env.deglex_time_cache_s) : NaN

        push!(
            train_episode_df,
            (
                i,
                grevlex_mean_reward,
                grevlex_mean_time_s,
                deglex_mean_reward,
                deglex_mean_time_s,
            ),
        )


        s = Float32.(state(env))

        done = false

        while !done
            global_timestep += 1
            epsilon = randn(rng_policy, env.num_vars, 1) .* STD

            raw_matrix =
                vcat([collect(Iterators.flatten(row))' for row in env.monomial_matrix]...)
            matrix = normalize_columns(raw_matrix)

            @assert env.num_polys >= env.num_vars "Current state padding assumes num_polys >= num_vars"
            padded_s = vcat(s, zeros(Float32, env.num_polys - env.num_vars))
            s_input = hcat(matrix, padded_s)
            s_input =
                reshape(s_input, (((env.num_vars * env.num_terms) + 1) * env.num_polys, 1))
            s_input = Float32.(s_input)

            raw_action = vec((actor.actor(s_input)))
            action = vec(Float32.(raw_action + epsilon))

            basis = act!(env, action, true)

            s_next = Float32.(state(env))
            push!(actions_taken, s_next)

            padded_s_next = vcat(s_next, zeros(Float32, env.num_polys - env.num_vars))
            s_next_input = hcat(matrix, padded_s_next)
            s_next_input = reshape(
                s_next_input,
                (((env.num_vars * env.num_terms) + 1) * env.num_polys, 1),
            )

            r = Float32(env.reward)

            if global_timestep % 11 == 0
                println("Timestep: $global_timestep, Raw action: $raw_action, reward: $r")
            end

            push!(rewards, r)

            push!(
                train_steps_df,
                (
                    global_timestep,
                    i,
                    env.iteration_count,
                    env.last_raw_reward,
                    env.last_delta_reward,
                    env.last_agent_batch_time_s,
                    env.last_agent_mean_time_s,
                ),
            )

            done = is_terminated(env)
            s_next = done ? nothing : s_next
            s_next_input = done ? nothing : s_next_input

            if !done
                if args["per"]
                    add_experience!(
                        replay_buffer,
                        # Transition(
                        # padded_s,
                        # padded_s_next,
                        # r,
                        # padded_s_next,
                        # s_input,
                        # s_next_input,
                        # ),
                        Transition(s, action, r, s_next, s_input, s_next_input),
                        abs(r),
                    )
                else
                    push!(
                        replay_buffer,
                        Transition(s, action, r, s_next, s_input, s_next_input),
                    )
                end

            end

            s = s_next === nothing ? s : s_next
            s_input = s_next_input === nothing ? s_input : s_next_input

            if length(replay_buffer) < N_SAMPLES
                continue
            end

            if args["per"]
                batch, indices, weights = sample(rng_buffer, replay_buffer)
            else
                batch = rand(rng_buffer, replay_buffer, N_SAMPLES)
            end

            s_batch = hcat([b.s for b in batch]...)
            a_batch = hcat([b.a for b in batch]...)
            r_batch = hcat([b.r for b in batch]...)
            s_next_batch = hcat(
                [
                    b.s_next !== nothing ? b.s_next : zeros(Float32, env.num_vars) for
                    b in batch
                ]...,
            )
            s_input_batch = hcat([b.s_input for b in batch]...)
            s_next_input_batch = hcat(
                [
                    b.s_next_input !== nothing ? b.s_next_input :
                    zeros(Float32, ((env.num_vars * env.num_terms) + 1) * env.num_polys) for b in batch
                ]...,
            )
            not_done = reshape(Float32.(getfield.(batch, :s_next_input) .!== nothing), 1, :)

            epsilon = clamp.(randn(rng_policy, 1, N_SAMPLES) * STD, -0.05f0, 0.05f0)

            if args["lstm"]
                Flux.reset!(actor.actor_target)
            end

            target_action = actor.actor_target(s_next_input_batch) .+ epsilon
            # target_action = vcat(target_action, zeros(Float32, env.num_polys - env.num_vars, N_SAMPLES))
            target_action = Float32.(target_action)

            critic_1_target_val =
                critic.critic_1_target(Float32.(vcat(s_next_input_batch, target_action)))
            critic_2_target_val =
                critic.critic_2_target(Float32.(vcat(s_next_input_batch, target_action)))

            min_q = min.(critic_1_target_val, critic_2_target_val)

            y = r_batch .+ GAMMA .* not_done .* min_q
            pred = critic.critic_1(Float32.(vcat(s_input_batch, a_batch)))

            if args["per"]
                errors = vec(Float32.(abs.(pred .- y)))
                update_priorities!(replay_buffer, indices, errors)
            end

            loss1, back1 = Flux.withgradient(critic.critic_1) do model
                pred = model(Float32.(vcat(s_input_batch, a_batch)))
                Flux.mse(pred, y)
            end

            push!(losses_1, loss1)
            if i >= EPISODES - 100
                println("Critic 1 loss: $loss1")
            end

            Flux.update!(critic.critic_1_opt_state, critic.critic_1, back1[1])

            loss2, back2 = Flux.withgradient(critic.critic_2) do model
                pred = model(Float32.(vcat(s_input_batch, a_batch)))
                Flux.mse(pred, y)
            end

            push!(losses_2, loss2)

            push!(train_updates_df, (global_timestep, Float64(loss1), Float64(loss2), NaN))

            Flux.update!(critic.critic_2_opt_state, critic.critic_2, back2[1])

            # Updating every D episodes 
            if global_timestep % D == 0
                if args["lstm"]
                    Flux.reset!(actor.actor)
                end
                actor_loss, back = Flux.withgradient(actor.actor) do model
                    a_pred = model(Float32.(s_input_batch))
                    # a_pred = vcat(
                    # a_pred,
                    # zeros(Float32, env.num_polys - env.num_vars, N_SAMPLES),
                    # )
                    q_val = critic.critic_1(Float32.(vcat(s_input_batch, a_pred)))
                    -mean(q_val)
                end

                push!(losses, actor_loss)
                push!(train_updates_df, (global_timestep, NaN, NaN, Float64(actor_loss)))

                grads = back[1]
                Flux.update!(actor.actor_opt_state, actor.actor, grads)

                soft_update!(critic.critic_1_target, critic.critic_1)
                soft_update!(critic.critic_2_target, critic.critic_2)
                soft_update!(actor.actor_target, actor.actor)
            end

        end

        if i % 1 == 0
            println(
                "Episode: $i, Action Taken: ",
                actions_taken[env.max_iterations*i],
                " Reward: ",
                rewards[env.max_iterations*i],
            ) # Losses get updated every D episodes
            println()
        end

        if i % SAVE_WEIGHTS == 0
            BSON.@save(checkpoint_path, actor, critic)
            println("Saved TD3 checkpoint to $checkpoint_path at episode $i")
        end

        if i % CSV_FLUSH_EVERY_EPISODES == 0
            CSV.write(train_steps_csv, train_steps_df; append = isfile(train_steps_csv))
            CSV.write(
                train_episode_csv,
                train_episode_df;
                append = isfile(train_episode_csv),
            )
            CSV.write(
                train_updates_csv,
                train_updates_df;
                append = isfile(train_updates_csv),
            )

            empty!(train_steps_df)
            empty!(train_episode_df)
            empty!(train_updates_df)
        end


        current_actor_lr = max(ACTOR_MIN_LR, current_actor_lr - ACTOR_LR_DECAY)
        current_critic_lr = max(CRITIC_MIN_LR, current_critic_lr - CRITIC_LR_DECAY)

        Flux.adjust!(actor.actor_opt_state, current_actor_lr)
        Flux.adjust!(critic.critic_1_opt_state, current_critic_lr)
        Flux.adjust!(critic.critic_2_opt_state, current_critic_lr)

    end

    CSV.write(
        train_steps_csv,
        train_steps_df;
        append = isfile(train_steps_csv),
        writeheader = !isfile(train_steps_csv),
    )
    CSV.write(
        train_episode_csv,
        train_episode_df;
        append = isfile(train_episode_csv),
        writeheader = !isfile(train_episode_csv),
    )
    CSV.write(
        train_updates_csv,
        train_updates_df,
        append = isfile(train_updates_csv),
        writeheader = !isfile(train_updates_csv),
    )

    episodes = 1:length(losses)
    loss_plot = scatter(
        episodes,
        losses,
        title = "Actor Loss plot",
        xlabel = "Actor Update Step (every $D timesteps)",
        ylabel = "Loss",
        label = "Actor Loss",
        color = :red,
        markersize = 1,
        markerstrokewidth = 0, # Removes the border around the dots
        legend = :topleft,
    )

    savefig(loss_plot, actor_plot_path)

    episodes2 = 1:length(rewards)
    reward_plot = scatter(
        episodes2,
        rewards,
        title = "Reward plot",
        xlabel = "Time step",
        ylabel = "Reward",
        label = "Reward",
        color = :green,
        markersize = 1,
        markerstrokewidth = 0,
        legend = :bottomright,
    )

    savefig(reward_plot, reward_plot_path)

    episodes_critic1 = 1:length(losses_1)
    episodes_critic2 = 1:length(losses_2)

    critic_plot = scatter(
        [episodes_critic1 episodes_critic2],
        [losses_1 losses_2],
        layout = (2, 1),
        legend = :topright,
        markersize = 1,
        markerstrokewidth = 0,
        color = [:purple :blue],
        xlabel = "Time step",
        ylabel = "Loss",
        label = ["Critic 1 Loss" "Critic 2 Loss"],
        title = ["Critic 1" "Critic 2"],
    )

    savefig(critic_plot, critics_plot_path)




    # n_cols_plot = scatter(1:length(n_cols_list), n_cols_list,
    #     title = "n_cols over time",
    #     xlabel = "Step",
    #     ylabel = "n_cols",
    #     label = "n_cols",
    #     markersize = 1,
    #     markerstrokewidth = 0,
    #     color = :orange)

    # savefig(n_cols_plot, "n_cols_plot.pdf")

    # n_deg_plot = scatter(1:length(pair_degrees), pair_degrees,
    #     title = "Pair Degrees over time",
    #     xlabel = "Step",
    #     ylabel = "pair_degree",
    #     label = "degree",
    #     markersize = 1,
    #     markerstrokewidth = 0,
    #     color = :cyan)

    # savefig(n_deg_plot, "pair_degrees_plot.pdf")

    # n_cts_plot = scatter(1:length(pair_counts), pair_counts,
    #     title = "Pair Counts over time",
    #     xlabel = "Step",
    #     ylabel = "pair_count",
    #     label = "count",
    #     markersize = 1,
    #     markerstrokewidth = 0,
    #     color = :magenta)

    # savefig(n_cts_plot, "pair_counts_plot.pdf")

end

function test_td3!(
    actor::Actor,
    critic::Critics,
    env::Environment,
    args::Dict{String,Any},
    rng_test::AbstractRNG,
    rng_env::AbstractRNG,
    rng_cal::AbstractRNG,
    rng_perm::AbstractRNG,
)
    rewards = []
    actions_taken = []

    run_tag =
        "td3_run_" * "baseset_" * string(args["baseset"]) * "_seed_" * string(args["seed"])

    base_set_path = joinpath(DATA_DIR, run_tag * "_base_sets.bin")
    reward_cmp_path = joinpath(PLOTS_DIR, run_tag * "_train_test_comparison.png")
    test_csv = joinpath(RESULTS_DIR, run_tag * "_test_metrics.csv")
    order_csv = joinpath(RESULTS_DIR, run_tag * "_final_agent_weight_vector.csv")

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
        base_sets, max_terms =
            pad_base_set(base_sets; max_terms = env.num_terms, num_vars = env.num_vars)
        @assert max_terms == env.num_terms
    end

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
        should_save_base_sets = base_sets === nothing,
        use_n_site_phosphorylation_coeffs = is_n_site,
    )


    env.variables = vars_cal
    env.monomial_matrix = monomial_matrix_cal
    println("Monomial_matrix: ", env.monomial_matrix)

    test_batch_orders = []

    for (idx, ideal) in enumerate(ideals_cal)
        println("running inference on calibration ideal: $idx")

        last_action = nothing

        reset_env!(env)
        env.ideal_batch = [ideal]
        s = Float32.(state(env))
        done = false

        if args["lstm"]
            Flux.reset!(actor.actor)
        end

        while !done
            raw_matrix =
                vcat([collect(Iterators.flatten(row))' for row in env.monomial_matrix]...)
            matrix = normalize_columns(raw_matrix)

            @assert env.num_polys >= env.num_vars "Current state padding assumes num_polys >= num_vars"
            padded_s = vcat(s, zeros(Float32, env.num_polys - env.num_vars))
            s_input = hcat(matrix, padded_s)
            s_input =
                reshape(s_input, (((env.num_vars * env.num_terms) + 1) * env.num_polys, 1))
            s_input = Float32.(s_input)

            action = vec(actor.actor(s_input))
            last_action = action
            basis = act!(env, action, false)

            s_next = Float32.(state(env))
            r = Float32(env.reward)
            
            s = Float32.(state(env))
            done = is_terminated(env)
        end

        push!(test_batch_orders, last_action)

    end

    cand_keys = unique(map(a -> tuple(Float32.(a)...), test_batch_orders))

    reward_map = Dict{NTuple{(env.num_vars),Float32},Float32}()
    for order in cand_keys
        println("evaluating order: $order")
        reset_env!(env)
        env.ideal_batch = ideals_cal
        basis = act!(env, collect(order), false)
        r = Float32(env.reward)
        key = tuple(Float32.(order)...)
        reward_map[key] = r
    end

    best_order = argmax(reward_map)
    println("found best order: $best_order with average reward: $(reward_map[best_order])")
    best_order = collect(best_order)

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
        should_save_base_sets = base_sets === nothing,
        use_n_site_phosphorylation_coeffs = is_n_site,
    )

    env.variables = vars
    env.monomial_matrix = monomial_matrix

    agent_rewards = []
    # lex_rewards = []
    deglex_rewards = []
    grevlex_rewards = []

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
    )

    ideal0 = ideals[1]
    ideal_copy = deepcopy(ideal0)
    groebner_learn(
        ideal_copy;
        ordering = DegRevLex(),
        seed = env.groebner_seed,
        monoms = GROEBNER_MONOMS,
        homogenize = GROEBNER_HOMOGENIZE,
    )
    ideal_copy = deepcopy(ideal0)
    groebner_learn(
        ideal_copy;
        ordering = DegLex(),
        seed = env.groebner_seed,
        monoms = GROEBNER_MONOMS,
        homogenize = GROEBNER_HOMOGENIZE,
    )
    ord0 = WeightedOrdering(zip(vars, Int.(1:length(vars)))...)
    ideal_copy = deepcopy(ideal0)
    groebner_learn(
        ideal_copy;
        ordering = ord0,
        seed = env.groebner_seed,
        monoms = GROEBNER_MONOMS,
        homogenize = GROEBNER_HOMOGENIZE,
    )
    println("Timing warmup complete.")

    ps = PermScheduler(rng_perm)

    GC.gc()

    for (idx, ideal) in enumerate(ideals)
        idx % 100 == 0 && println("testing on ideal: $idx")

        if idx % 500 == 0
            GC.gc()
        end

        perm = next_perm!(ps)

        r_agent = t_agent = NaN
        r_deg = t_deg = NaN
        r_grev = t_grev = NaN

        for m in perm
            r, t = run_method(m, ideal, vars, best_order, env.groebner_seed)
            if m == :agent
                r_agent, t_agent = r, t
            elseif m == :deglex
                r_deg, t_deg = r, t
            elseif m == :degrevlex
                r_grev, t_grev = r, t
            end

        end

        ratio_grev = (t_grev > 0) ? (t_agent / t_grev) : NaN
        ratio_deg = (t_deg > 0) ? (t_agent / t_deg) : NaN

        push!(
            test_df,
            (
                idx,
                r_agent,
                t_agent,
                r_deg,
                t_deg,
                r_grev,
                t_grev,
                r_agent - r_grev,
                ratio_grev,
            ),
        )

        push!(agent_rewards, r_agent)
        push!(deglex_rewards, r_deg)
        push!(grevlex_rewards, r_grev)

    end

    CSV.write(test_csv, test_df)
    CSV.write(order_csv, DataFrame(var = 1:length(best_order), weight = best_order))

    println("Wrote test metrics CSV to $test_csv")
    println("Wrote final agent weight vector CSV to $order_csv")

    serialize(joinpath(RESULTS_DIR, run_tag * "_final_agent_order.bin"), best_order)
    serialize(joinpath(RESULTS_DIR, run_tag * "_test_agent_rewards.bin"), agent_rewards)
    serialize(joinpath(RESULTS_DIR, run_tag * "_test_deglex_rewards.bin"), deglex_rewards)
    serialize(joinpath(RESULTS_DIR, run_tag * "_test_grevlex_rewards.bin"), grevlex_rewards)

    rewards = agent_rewards

    episodes2 = 1:length(rewards)
    reward_plot = scatter(
        episodes2,
        rewards,
        title = "Reward plot",
        xlabel = "Time step",
        ylabel = "Reward",
        label = "Reward",
        color = :green,
        markersize = 2,
        markerstrokewidth = 0,
        legend = :bottomleft,
    )

    savefig(reward_plot, joinpath(PLOTS_DIR, run_tag * "_test_reward_plot.png"))

    reward_comparison = plot(
        episodes2,
        rewards,
        label = "Agent",
        color = :green,
        linewidth = 1,
        markersize = 1,
        markerstrokewidth = 0,
    )

    # plot!(
    #     episodes2, lex_rewards,
    #     label = "Lex",
    #     color = :orange,
    #     linewidth = 2,
    #     markersize = 2,
    #     markerstrokewidth = 0,
    # )

    plot!(
        episodes2,
        deglex_rewards,
        label = "Grlex",
        color = :blue,
        linewidth = 1,
        markersize = 1,
        markerstrokewidth = 0,
    )

    plot!(
        episodes2,
        grevlex_rewards,
        label = "Grevlex",
        color = :red,
        linewidth = 1,
        markersize = 1,
        markerstrokewidth = 0,
        xlabel = "Time step",
        ylabel = "Reward",
        legend = :bottomright,
    )
    # savefig("reward_comparison.pdf")
    savefig(reward_cmp_path)


end

function soft_update!(target, policy)
    for (tp, pp) in zip(Flux.params(target), Flux.params(policy))
        tp .= (1 - TAU) * tp .+ TAU * pp
    end
end

function normalize_columns(M::AbstractMatrix)
    mapslices(x -> x / (norm(x) + 1e-8), M; dims = 1)
end

function run_method(method::Symbol, ideal, vars, best_order, groebner_seed)
    if method == :agent
        return eval_order_on_ideal(ideal, vars, best_order, groebner_seed)
    elseif method == :deglex
        ideal_copy = deepcopy(ideal)
        (tr, _), t = timed(
            () -> groebner_learn(
                ideal_copy;
                ordering = DegLex(),
                seed = groebner_seed,
                monoms = GROEBNER_MONOMS,
                homogenize = GROEBNER_HOMOGENIZE,
            ),
        )
        return Float64(reward(tr)), Float64(t)
    elseif method == :degrevlex
        ideal_copy = deepcopy(ideal)
        (tr, _), t = timed(
            () -> groebner_learn(
                ideal_copy;
                ordering = DegRevLex(),
                seed = groebner_seed,
                monoms = GROEBNER_MONOMS,
                homogenize = GROEBNER_HOMOGENIZE,
            ),
        )
        return Float64(reward(tr)), Float64(t)
    else
        error("Unknown method: $method")
    end
end


function eval_order_on_ideal(ideal, vars, weights, groebner_seed)
    weights = Int.(round.(ACTION_SCALE * weights))
    weights = max.(weights, 1)
    w = zip(vars, weights)
    order = WeightedOrdering(w...)
    ideal_copy = deepcopy(ideal)
    (tr, _), t = timed(
        () -> groebner_learn(
            ideal_copy;
            ordering = order,
            seed = groebner_seed,
            monoms = GROEBNER_MONOMS,
            homogenize = GROEBNER_HOMOGENIZE,
        ),
    )
    return Float64(reward(tr)), Float64(t)
end


function timing_warmup_all!(
    actor::Actor,
    critic::Critics,
    env::Environment,
    ideal0,
    vars,
    rng_policy::AbstractRNG;
    do_backprop::Bool = true,
)
    ideal_copy = deepcopy(ideal0)
    groebner_learn(
        ideal_copy;
        ordering = DegRevLex(),
        seed = env.groebner_seed,
        monoms = GROEBNER_MONOMS,
        homogenize = GROEBNER_HOMOGENIZE,
    )

    ideal_copy = deepcopy(ideal0)
    groebner_learn(
        ideal_copy;
        ordering = DegLex(),
        seed = env.groebner_seed,
        monoms = GROEBNER_MONOMS,
        homogenize = GROEBNER_HOMOGENIZE,
    )

    ideal_copy = deepcopy(ideal0)
    ord0 = WeightedOrdering(zip(vars, Int.(1:length(vars)))...)
    groebner_learn(
        ideal_copy;
        ordering = ord0,
        seed = env.groebner_seed,
        monoms = GROEBNER_MONOMS,
        homogenize = GROEBNER_HOMOGENIZE,
    )

    raw_matrix = vcat([collect(Iterators.flatten(row))' for row in env.monomial_matrix]...)
    matrix = normalize_columns(raw_matrix)

    s = Float32.(state(env))
    @assert env.num_polys >= env.num_vars "Current state padding assumes num_polys >= num_vars"
    padded_s = vcat(s, zeros(Float32, env.num_polys - env.num_vars))
    s_input = hcat(matrix, padded_s)
    s_input = reshape(s_input, (((env.num_vars * env.num_terms) + 1) * env.num_polys, 1))
    s_input = Float32.(s_input)

    _ = actor.actor(s_input)
    _ = actor.actor_target(s_input)

    N = 4
    s_input_batch = repeat(s_input, 1, N)
    a = Float32.(actor.actor(s_input))          # (num_vars, 1)
    a_batch = repeat(a, 1, N)                   # (num_vars, N)

    _ = critic.critic_1(vcat(s_input_batch, a_batch))
    _ = critic.critic_2(vcat(s_input_batch, a_batch))
    _ = critic.critic_1_target(vcat(s_input_batch, a_batch))
    _ = critic.critic_2_target(vcat(s_input_batch, a_batch))

    if do_backprop
        y = randn(rng_policy, Float32, 1, N)

        loss1, back1 = Flux.withgradient(critic.critic_1) do model
            pred = model(Float32.(vcat(s_input_batch, a_batch)))
            Flux.mse(pred, y)
        end
        Flux.update!(critic.critic_1_opt_state, critic.critic_1, back1[1])

        loss2, back2 = Flux.withgradient(critic.critic_2) do model
            pred = model(Float32.(vcat(s_input_batch, a_batch)))
            Flux.mse(pred, y)
        end
        Flux.update!(critic.critic_2_opt_state, critic.critic_2, back2[1])

        actor_loss, backa = Flux.withgradient(actor.actor) do model
            a_pred = model(Float32.(s_input_batch))
            q_val = critic.critic_1(Float32.(vcat(s_input_batch, a_pred)))
            -mean(q_val)
        end
        Flux.update!(actor.actor_opt_state, actor.actor, backa[1])
    end

    return nothing
end

function load_td3(env::Environment, args::Dict{String,Any})
    run_tag =
        "td3_run_" * "baseset_" * string(args["baseset"]) * "_seed_" * string(args["seed"])
    checkpoint_path = joinpath(WEIGHTS_DIR, run_tag * "_td3_checkpoint.bson")
    if isfile(checkpoint_path)
        println("Checkpoint found. Loading saved model")
        actor = nothing
        critic = nothing
        if args["per"]
            replay_buffer = PrioritizedReplayBuffer(
                CAPACITY,
                N_SAMPLES,
                ALPHA,
                BETA,
                BETA_INCREMENT,
                EPS,
            )
        else
            replay_buffer = CircularBuffer{Transition}(CAPACITY)
        end
        BSON.@load(checkpoint_path, actor, critic)
        return actor, critic, replay_buffer
    else
        println("No checkpoint found. Training models from scratch")
        actor, critic = build_td3_model(env, args)
        if args["per"]
            replay_buffer = PrioritizedReplayBuffer(
                CAPACITY,
                N_SAMPLES,
                ALPHA,
                BETA,
                BETA_INCREMENT,
                EPS,
            )
        else
            replay_buffer = CircularBuffer{Transition}(CAPACITY)
        end
        return actor, critic, replay_buffer
    end
end
