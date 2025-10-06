using Flux
using DataStructures
using ReinforcementLearning
using Statistics
using Serialization
using Optimisers
using Plots
using LinearAlgebra
using BSON
include("environment.jl")
include("utils.jl")
include("basesets.jl")
# plotlyjs()

BASE_DIR = @__DIR__
DATA_DIR = joinpath(BASE_DIR, "data")
WEIGHTS_DIR = joinpath(BASE_DIR, "weights")
RESULTS_DIR = joinpath(BASE_DIR, "results")
PLOTS_DIR = joinpath(BASE_DIR, "plots")

BASE_SET_PATH = joinpath(DATA_DIR, "base_sets.bin")
CHECKPOINT_PATH = joinpath(WEIGHTS_DIR, "td3_checkpoint.bson")

ACTOR_PLOT_PATH = joinpath(PLOTS_DIR, "actor_plot.png")
REWARD_PLOT_PATH = joinpath(PLOTS_DIR, "train_reward_plot.png")
CRITICS_PLOT_PATH = joinpath(PLOTS_DIR, "critics_plot.png")
REWARD_CMP_PATH = joinpath(PLOTS_DIR, "reward_comparison.png")

for d in (DATA_DIR, WEIGHTS_DIR, RESULTS_DIR, PLOTS_DIR)
    isdir(d) || mkpath(d)
end


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

function build_td3_model(env::Environment, args::Dict{String, Any})

    if args["lstm"]
        # LSTM layer actor
        actor_layers = Any[LSTM(((env.num_vars * env.num_terms) + 1) * env.num_polys => ACTOR_HIDDEN_WIDTH)]
        for l in 1:(ACTOR_DEPTH - 1)
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
            ((env.num_vars * env.num_terms) + 2) * env.num_polys,
            CRITIC_HIDDEN_WIDTH,
            relu,
        ),
        Dense(CRITIC_HIDDEN_WIDTH, CRITIC_HIDDEN_WIDTH, relu),
        Dense(CRITIC_HIDDEN_WIDTH, CRITIC_HIDDEN_WIDTH, relu),
        Dense(CRITIC_HIDDEN_WIDTH, 1),
    )
    critic_2 = Flux.Chain(
        Dense(
            ((env.num_vars * env.num_terms) + 2) * env.num_polys,
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
    replay_buffer::Union{PrioritizedReplayBuffer, CircularBuffer{Transition}},
    initial_actor_lr::Float64,
    initial_critic_lr::Float64,
    args::Dict{String, Any},
)

    losses = []
    rewards = []
    actions_taken = []
    losses_1 = []
    losses_2 = []

    current_actor_lr = initial_actor_lr
    current_critic_lr = initial_critic_lr

    base_sets = isfile(BASE_SET_PATH) ? load_base_sets(BASE_SET_PATH) : nothing
    
    if args["baseset"] == "N_SITE_PHOSPHORYLATION_BASE_SET"
        base_sets = N_SITE_PHOSPHORYLATION_BASE_SET
    elseif args["baseset"] == "RELATIVE_POSE_BASE_SET"
        base_sets = RELATIVE_POSE_BASE_SET
    elseif args["baseset"] == "TRIANGULATION_BASE_SET"
        base_sets = TRIANGULATION_BASE_SET
    elseif args["baseset"] == "WNT_BASE_SET"
        base_sets = WNT_BASE_SET
    elseif args["baseset"] == "FOUR_PT_BASE_SET"
        base_sets = FOUR_PT_BASE_SET
    elseif args["baseset"] == "DEFAULT"
        max_degree = DEFAULT_MAX_DEGREE
    else 
        error("Unknown baseset: $(args["baseset"])")
    end

    if args["baseset"] != "DEFAULT"
        flat_terms = vcat(base_sets...)
        max_degree = maximum(sum(term) for term in flat_terms)
    end

    ideals, vars, monomial_matrix = new_generate_data(
        num_ideals = EPISODES * NUM_IDEALS,
        num_polynomials = env.num_polys,
        num_variables = env.num_vars,
        max_degree = max_degree,
        num_terms = env.num_terms,
        max_attempts = MAX_ATTEMPTS,
        base_sets = base_sets,
        base_set_path = BASE_SET_PATH,
        should_save_base_sets = base_sets === nothing,
        use_n_site_phosphorylation_coeffs = base_sets === N_SITE_PHOSPHORYLATION_BASE_SET,
    )

    env.variables = vars
    env.monomial_matrix = monomial_matrix
    println("Monomial_matrix: ", env.monomial_matrix)

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

        s = Float32.(state(env))

        done = false
        # episode_loss = []
        # critic_1_episode_loss = []
        # critic_2_episode_loss = []
        # episode_rewards = []
        # episode_actions = []

        while !done
            global_timestep += 1
            epsilon = randn(env.num_vars, 1) .* STD

            raw_matrix =
                vcat([collect(Iterators.flatten(row))' for row in env.monomial_matrix]...)
            matrix = normalize_columns(raw_matrix)

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


            done = is_terminated(env)
            s_next = done ? nothing : s_next
            s_next_input = done ? nothing : s_next_input

            if !done
                if args["per"]
                    add_experience!(
                        replay_buffer,
                        Transition(
                            padded_s,
                            padded_s_next,
                            r,
                            padded_s_next,
                            s_input,
                            s_next_input,
                        ),
                        abs(r),)
                else
                    push!(replay_buffer, Transition(s, action, r, s_next, s_input, s_next_input)) 
                end

            end

            s = s_next === nothing ? s : s_next
            s_input = s_next_input === nothing ? s_input : s_next_input

            if length(replay_buffer) < N_SAMPLES
                continue
            end

            if args["per"]
                batch, indices, weights = sample(replay_buffer)
            else 
                batch = rand(replay_buffer, N_SAMPLES)
            end

            s_batch = hcat([b.s for b in batch]...)
            a_batch = hcat([b.a for b in batch]...)
            r_batch = hcat([b.r for b in batch]...)
            s_next_batch = hcat(
                [
                    b.s_next !== nothing ? b.s_next : zeros(Float32, env.num_polys) for
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

            epsilon = clamp.(randn(1, N_SAMPLES) * STD, -0.05f0, 0.05f0)
            
            if args["lstm"]
                Flux.reset!(actor.actor_target)
            end

            target_action = actor.actor_target(s_next_input_batch) .+ epsilon
            target_action =
                vcat(target_action, zeros(Float32, env.num_polys - env.num_vars, N_SAMPLES))
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

            Flux.update!(critic.critic_2_opt_state, critic.critic_2, back2[1])

            # Updating every D episodes 
            if global_timestep % D == 0
                if args["lstm"]
                    Flux.reset!(actor.actor)
                end
                actor_loss, back = Flux.withgradient(actor.actor) do model
                    a_pred = model(Float32.(s_input_batch))
                    a_pred = vcat(
                        a_pred,
                        zeros(Float32, env.num_polys - env.num_vars, N_SAMPLES),
                    )
                    q_val = critic.critic_1(Float32.(vcat(s_input_batch, a_pred)))
                    -mean(q_val)
                end

                push!(losses, actor_loss)

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
            BSON.@save(CHECKPOINT_PATH, actor, critic)
            println("Saved TD3 checkpoint to $CHECKPOINT_PATH at episode $i")
        end

        current_actor_lr = max(ACTOR_MIN_LR, current_actor_lr - ACTOR_LR_DECAY)
        current_critic_lr = max(CRITIC_MIN_LR, current_critic_lr - CRITIC_LR_DECAY)

        Flux.adjust!(actor.actor_opt_state, current_actor_lr)
        Flux.adjust!(critic.critic_1_opt_state, current_critic_lr)
        Flux.adjust!(critic.critic_2_opt_state, current_critic_lr)

    end

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

    # savefig(loss_plot, "actor_plot_newbase.pdf")
    savefig(loss_plot, ACTOR_PLOT_PATH)

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

    # savefig(reward_plot, "reward_plot_newbase.pdf")
    savefig(reward_plot, REWARD_PLOT_PATH)

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

    # savefig(critic_plot, "critics_plot_newbase.pdf")
    savefig(critic_plot, CRITICS_PLOT_PATH)


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

function test_td3!(actor::Actor, critic::Critics, env::Environment, args::Dict{String, Any})

    rewards = []
    actions_taken = []

    base_sets = isfile(BASE_SET_PATH) ? load_base_sets(BASE_SET_PATH) : nothing
    
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

    if args["baseset"] != "DEFAULT"
        flat_terms = vcat(base_sets...)
        max_degree = maximum(sum(term) for term in flat_terms)
    end

    ideals, vars, monomial_matrix = new_generate_data(
        num_ideals = NUM_TEST_IDEALS,
        num_polynomials = env.num_polys,
        num_variables = env.num_vars,
        max_degree = max_degree,
        num_terms = env.num_terms,
        max_attempts = MAX_ATTEMPTS,
        base_sets = base_sets,
        base_set_path = BASE_SET_PATH,
        should_save_base_sets = base_sets === nothing,
        use_n_site_phosphorylation_coeffs = base_sets === N_SITE_PHOSPHORYLATION_BASE_SET,
    )

    env.variables = vars
    env.monomial_matrix = monomial_matrix
    println("Monomial_matrix: ", env.monomial_matrix)

    test_batch = rand(ideals, TEST_BATCH_SIZE)
    test_batch_orders = []
    for (idx, ideal) in enumerate(test_batch)
        println("running inference on ideal: $idx")

        reward_map = Dict{NTuple{(env.num_vars),Float32},Float32}()

        reset_env!(env)
        env.ideal_batch = [ideal]
        s = Float32.(state(env))
        done = false
        
        if args["lstm"]
            Flux.reset!(actor.actor_target)
        end
        
        while !done
            raw_matrix =
                vcat([collect(Iterators.flatten(row))' for row in env.monomial_matrix]...)
            matrix = normalize_columns(raw_matrix)

            padded_s = vcat(s, zeros(Float32, env.num_polys - env.num_vars))
            s_input = hcat(matrix, padded_s)
            s_input =
                reshape(s_input, (((env.num_vars * env.num_terms) + 1) * env.num_polys, 1))
            s_input = Float32.(s_input)
            
            action = vec(actor.actor_target(s_input))
            basis = act!(env, action, false)

            s_next = Float32.(state(env))
            r = Float32(env.reward)

            reward_map[tuple(s_next...)] = r
            done = is_terminated(env)
        end

        push!(test_batch_orders, argmax(reward_map))

    end

    reward_map = Dict{NTuple{(env.num_vars),Float32},Float32}()
    for order in test_batch_orders
        println("evaluating order: $order")
        reset_env!(env)
        env.ideal_batch = test_batch
        basis = act!(env, collect(order), false)
        s_next = Float32.(state(env))
        r = Float32(env.reward)
        reward_map[tuple(s_next...)] = r
    end

    best_order = argmax(reward_map)
    println("found best order: $best_order with average reward: $(reward_map[best_order])")
    best_order = collect(best_order)

    agent_rewards = []
    lex_rewards = []
    deglex_rewards = []
    grevlex_rewards = []

    for (idx, ideal) in enumerate(ideals)

        if idx % 100 == 0
            println("testing on ideal: $idx")
        end

        reset_env!(env)
        env.ideal_batch = [ideal]
        basis = act!(env, best_order, false)
        s_next = Float32.(state(env))
        r = Float32(env.reward)

        push!(agent_rewards, r)

        # lex_trace, lex_basis = groebner_learn(ideal, ordering = Lex())
        # push!(lex_rewards, reward(lex_trace))

        deglex_trace, deglex_basis = groebner_learn(ideal, ordering = DegLex())
        push!(deglex_rewards, reward(deglex_trace))

        grevlex_trace, grevlex_basis = groebner_learn(ideal, ordering = DegRevLex())
        push!(grevlex_rewards, reward(grevlex_trace))

    end

    serialize(joinpath(RESULTS_DIR, "agent_order.bin"), best_order)
    serialize(joinpath(RESULTS_DIR, "agent_rewards.bin"), agent_rewards)
    serialize(joinpath(RESULTS_DIR, "deglex_rewards.bin"), deglex_rewards)
    serialize(joinpath(RESULTS_DIR, "grevlex_rewards.bin"), grevlex_rewards)


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

    savefig(reward_plot, "test_reward_plot.png")

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
    savefig(REWARD_CMP_PATH)


end

function soft_update!(target, policy)
    for (tp, pp) in zip(Flux.params(target), Flux.params(policy))
        tp .= (1 - TAU) * tp .+ TAU * pp
    end
end

function normalize_columns(M::AbstractMatrix)
    mapslices(x -> x / (norm(x) + 1e-8), M; dims = 1)
end

function load_td3(env::Environment, args::Dict{String, Any})
    if isfile(CHECKPOINT_PATH)
        println("Checkpoint found. Loading saved model")
        actor = nothing
        critic = nothing
        if args["per"]
            replay_buffer = PrioritizedReplayBuffer(CAPACITY, N_SAMPLES, ALPHA, BETA, BETA_INCREMENT, EPS)
        else 
            replay_buffer = CircularBuffer{Transition}(CAPACITY)
        end
        BSON.@load(CHECKPOINT_PATH, actor, critic)
        return actor, critic, replay_buffer
    else
        println("No checkpoint found. Training models from scratch")
        actor, critic = build_td3_model(env, args)
        if args["per"]
            replay_buffer = PrioritizedReplayBuffer(CAPACITY, N_SAMPLES, ALPHA, BETA, BETA_INCREMENT, EPS)
        else
            replay_buffer = CircularBuffer{Transition}(CAPACITY)
        end
        return actor, critic, replay_buffer
    end
end
