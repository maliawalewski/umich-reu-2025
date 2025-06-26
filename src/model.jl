using Flux
using DataStructures
using ReinforcementLearning
using Statistics
using Optimisers
using Plots
using LinearAlgebra
include("environment.jl")
plotlyjs()

# TD3 parameters
CAPACITY = 1_000_000
EPISODES = 1000
N_SAMPLES = 100
GAMMA = 0.99 # Discount factor
TAU = 0.005 # Soft update parameter
LR = 1e-2 # Learning rate for actor and critics
MIN_LR = 1e-5 # Minimum Learning Rate
LR_DECAY = 4.95e-06 # LR decay Rate
STD = 0.2 # Standard deviation for exploration noise
D = 10 # Update frequency for target actor and critics 

# Data parameters (used to generate ideal batch)
MAX_DEGREE = 4
MAX_ATTEMPTS = 100

# NN parameters
CRITIC_HIDDEN_WIDTH = 256
ACTOR_HIDDEN_WIDTH = 256 
ACTOR_DEPTH = 2 # Number of LSTM layers

struct Transition
    s::Vector{Float32}
    a::Vector{Float32}
    r::Float32
    s_next::Union{Vector{Float32},Nothing}
    s_input::Array{Float32}
    s_next_input::Union{Array{Float32},Nothing}
end

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

function init_actor(actor::Flux.Chain, 
    actor_target::Flux.Chain, 
    actor_opt_state::Any
)
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
    return Critics(critic_1, critic_2, critic_1_target, critic_2_target, critic_1_opt_state, critic_2_opt_state)
end

function build_td3_model(env::Environment)
    
    actor_layers = Any[Dense(((env.num_vars * env.num_terms) + 1) * env.num_vars => ACTOR_HIDDEN_WIDTH)]
    for l in 1:(ACTOR_DEPTH - 1)
        layer = Dense(ACTOR_HIDDEN_WIDTH => ACTOR_HIDDEN_WIDTH)
        push!(actor_layers, layer)
    end
    push!(actor_layers, Dense(ACTOR_HIDDEN_WIDTH, env.num_vars))
    push!(actor_layers, sigmoid)

    actor = Flux.Chain(actor_layers...)
    
    actor = Flux.Chain(Dense(((env.num_vars * env.num_terms) + 1) * env.num_vars, CRITIC_HIDDEN_WIDTH, relu), Dense(CRITIC_HIDDEN_WIDTH, CRITIC_HIDDEN_WIDTH, relu), Dense(CRITIC_HIDDEN_WIDTH, env.num_vars, sigmoid))

    critic_1 = Flux.Chain(Dense(((env.num_vars * env.num_terms) + 2) * env.num_vars, CRITIC_HIDDEN_WIDTH, relu), Dense(CRITIC_HIDDEN_WIDTH, CRITIC_HIDDEN_WIDTH, relu), Dense(CRITIC_HIDDEN_WIDTH, 1))
    critic_2 = Flux.Chain(Dense(((env.num_vars * env.num_terms) + 2) * env.num_vars, CRITIC_HIDDEN_WIDTH, relu), Dense(CRITIC_HIDDEN_WIDTH, CRITIC_HIDDEN_WIDTH, relu), Dense(CRITIC_HIDDEN_WIDTH, 1))

    actor_target = deepcopy(actor)
    critic_1_target = deepcopy(critic_1)
    critic_2_target = deepcopy(critic_2)

    actor_opt = ADAM(LR)
    actor_opt_state = Flux.setup(actor_opt, actor)

    critic_1_opt = ADAM(LR)
    critic_2_opt = ADAM(LR)

    critic_1_opt_state = Flux.setup(critic_1_opt, critic_1)
    critic_2_opt_state = Flux.setup(critic_2_opt, critic_2)
    
    actor_struct = Actor(actor, actor_target, actor_opt_state)
    critic_struct = Critics(critic_1, critic_2, critic_1_target, critic_2_target, critic_1_opt_state, critic_2_opt_state)

    return actor_struct, critic_struct
end

function train_td3!(actor::Actor, critic::Critics, env::Environment, replay_buffer::CircularBuffer{Transition}, initial_lr::Float64)

    losses = []
    rewards = []
    actions_taken = []
    losses_1 = []
    losses_2 = []

    current_lr = initial_lr

    for i = 1:EPISODES
        reset_env!(env)
        # fill_ideal_batch(env, env.num_polys, MAX_DEGREE, MAX_ATTEMPTS) # fill with random ideals
        
        # TESTING FIXED IDEAL
        field = GF(32003)
        ring, (x, y, z) = polynomial_ring(field, ["x", "y", "z"])
        ideal = [x^2 + y + z, x + x*y^2 + z^3, x^3*y + x*y + y*z^2]
        env.ideal_batch = [ideal, ideal, ideal, ideal, ideal, ideal, ideal, ideal, ideal, ideal]
        env.monomial_matrix = [[[2,0,0],[0,1,0],[0,0,1]],
        [[1,0,0],[1,2,0],[0,0,3]],
        [[3,1,0],[1,1,0],[0,1,2]]
        ]
        env.variables = [x, y, z]
        # END TESTING FIXED IDEAL

        s = Float32.(state(env))
        done = false
        t = 0
        episode_loss = []
        critic_1_episode_loss = []
        critic_2_episode_loss = []
        episode_rewards = []
        episode_actions = []

        while !done
            epsilon = randn() * STD
            matrix = hcat([reduce(hcat, group) for group in env.monomial_matrix]...)
            s_input = hcat(matrix, s)
            s_input = reshape(s_input, (((env.num_vars * env.num_terms) + 1) * env.num_vars, 1))
            # println("S: ", s)
            # println("NN OUTPUT: ", actor.actor(s_input))
            action = vec(Float32.(actor.actor(s_input) .+ epsilon))
            action_valid = make_valid_action(env, s, action) # To store valid action in replay buffer

            basis = act!(env, action)

            s_next = Float32.(state(env))
            s_next_input = hcat(matrix, s_next)
            s_next_input = reshape(s_next_input, (((env.num_vars * env.num_terms) + 1) * env.num_vars, 1))

            r = Float32(env.reward)
            push!(episode_rewards, r)

            done = is_terminated(env)
            s_next = done ? nothing : s_next
            s_next_input = done ? nothing : s_next_input

            push!(episode_actions, s_next)
            

            # TODO: ASK ALPEREN ABOUT THIS
            if !done
                push!(replay_buffer, Transition(s, action_valid, r, s_next, s_input, s_next_input)) # STORE VALID ACTIONS
            end

            s = s_next === nothing ? s : s_next
            s_input = s_next_input === nothing ? s_input : s_next_input

            if length(replay_buffer) < N_SAMPLES
                t += 1
                continue
            end

            batch = rand(replay_buffer, N_SAMPLES)
            s_batch = hcat([b.s for b in batch]...)
            a_batch = hcat([b.a for b in batch]...)
            r_batch = hcat([b.r for b in batch]...)
            s_next_batch = hcat(
                [b.s_next !== nothing ? b.s_next : zeros(Float32, env.num_vars) for b in batch]...,
            )
            s_input_batch = hcat([b.s_input for b in batch]...)
            s_next_input_batch = hcat([b.s_next_input !== nothing ? b.s_next_input : zeros(Float32, ((env.num_vars * env.num_terms) + 1) * env.num_vars) for b in batch]...,)
            not_done = reshape(Float32.(getfield.(batch, :s_next_input) .!== nothing), 1, :)

            epsilon = clamp.(randn(1, N_SAMPLES) * STD, -0.005f0, 0.005f0)

            target_action = actor.actor_target(s_next_input_batch) .+ epsilon
            
            target_action = Float32.(target_action)

            # TODO: Unclip - critics cannot take in clipped action because of backpropogation
            for i in 1:N_SAMPLES
                target_action[:, i] = make_valid_action(env, s_next_batch[:, i], target_action[:, i])
            end

            critic_1_target_val = critic.critic_1_target(vcat(s_next_input_batch, target_action))
            critic_2_target_val = critic.critic_2_target(vcat(s_next_input_batch, target_action))

            min_q = min.(critic_1_target_val, critic_2_target_val)

            y = r_batch .+ GAMMA .* not_done .* min_q
            
            loss1, back1 = Flux.withgradient(critic.critic_1) do model
                pred = model(vcat(s_input_batch, a_batch))
                mean((pred .- y) .^ 2)
            end

            push!(critic_1_episode_loss, loss1)

            Flux.update!(critic.critic_1_opt_state, critic.critic_1, back1[1])

            loss2, back2 = Flux.withgradient(critic.critic_2) do model
                pred = model(vcat(s_input_batch, a_batch))
                mean((pred .- y) .^ 2)
            end

            push!(critic_2_episode_loss, loss2)

            Flux.update!(critic.critic_2_opt_state, critic.critic_2, back2[1])

            # TODO: t is only updating once every episode fix to update every few episodes
            if t % D == 0
                actor_loss, back = Flux.withgradient(actor.actor) do model
                    a_pred = model(s_input_batch)
                    q_val = critic.critic_1(vcat(s_input_batch, a_pred))
                    -mean(q_val)
                end

                push!(episode_loss, actor_loss)

                grads = back[1]
                Flux.update!(actor.actor_opt_state, actor.actor, grads)

                soft_update!(critic.critic_1_target, critic.critic_1)
                soft_update!(critic.critic_2_target, critic.critic_2)
                soft_update!(actor.actor_target, actor.actor)
            end

            t += 1

        end

        if length(episode_loss) != 0
            push!(losses, mean(episode_loss))
        end

        if length(critic_1_episode_loss) != 0
            push!(losses_1, mean(critic_1_episode_loss))
            push!(losses_2, mean(critic_2_episode_loss))
        end

        if length(episode_rewards) != 0
            push!(rewards, mean(episode_rewards))
            push!(actions_taken, mean(episode_actions))
        end

        # TODO: fix print statements to line up with correct losses, don't print avgs
        if i % 100 == 0
            println("Episode: $i, Action Taken: ", actions_taken[i], ", Loss: ", losses[i], " Reward: ", reward[i]) # actor losses don't line up
            println()
        end

        current_lr = max(MIN_LR, current_lr - LR_DECAY) 
        Flux.adjust!(actor.actor_opt_state, current_lr)
        Flux.adjust!(critic.critic_1_opt_state, current_lr)
        Flux.adjust!(critic.critic_2_opt_state, current_lr)

    end

    episodes = 1:length(losses)
    loss_plot = plot(episodes, losses,
        title = "Loss plot",
        xlabel = "Episode",
        ylabel = "Loss",
        label = "Loss",
        lw = 1,
        marker = false,
        legend = :topright)

    savefig(loss_plot, "loss_plot.html")

    episodes2 = 1:length(rewards)
    reward_plot = plot(episodes2, rewards,
        title = "Reward plot",
        xlabel = "Episode",
        ylabel = "Reward",
        label = "Reward",
        lw = 1,
        marker = false,
        legend = :topright)

    savefig(reward_plot, "reward_plot.html")

    episodes3 = 1:length(losses_1)
    loss_plot = plot(episodes3, losses_1,
        title = "Critic1 loss plot",
        xlabel = "Episode",
        ylabel = "Loss",
        label = "Loss",
        lw = 1,
        marker = false,
        legend = :topright)

    savefig(loss_plot, "critic_1_loss_plot.html")

    episodes4 = 1:length(losses_2)
    loss_plot = plot(episodes4, losses_2,
        title = "Critic2 loss plot",
        xlabel = "Episode",
        ylabel = "Loss",
        label = "Loss",
        lw = 1,
        marker = false,
        legend = :topright)

    savefig(loss_plot, "critic_2_loss_plot.html")
end

function soft_update!(target, policy)
    for (tp, pp) in zip(Flux.params(target), Flux.params(policy))
        tp .= (1 - TAU) * tp .+ TAU * pp
    end
end
