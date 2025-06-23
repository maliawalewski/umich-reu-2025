using Flux
using DataStructures
using ReinforcementLearning
using Statistics
using Optimisers
using Plots
include("environment.jl")

CAPACITY = 1_000_000
EPISODES = 10_000 # we want 10-20 ideals computed in 1 episode 
N_SAMPLES = 100
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
STD = 0.2
D = 10 # can change 


# Data parameters (used to generate ideal batch)
NUM_POLYS = 3
MAX_DEGREE = 4
NUM_TERMS = 3
MAX_ATTEMPTS = 100

struct Transition
    s::Vector{Float32}
    a::Vector{Float32}
    r::Float32
    s_next::Union{Vector{Float32},Nothing}
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

    actor = Flux.Chain(Dense(env.numVars, 256, relu), Dense(256, 256, relu), Dense(256, env.numVars, tanh)) # TODO fix tanh output to match action space
    critic_1 = Flux.Chain(Dense(2 * env.numVars, 256, relu), Dense(256, 256, relu), Dense(256, 1))
    critic_2 = Flux.Chain(Dense(2 * env.numVars, 256, relu), Dense(256, 256, relu), Dense(256, 1))

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

function train_td3!(actor::Actor, critic::Critics, env::Environment, replay_buffer::CircularBuffer{Transition})

    losses = []
    rewards = []
    actions_taken = []

    for i = 1:EPISODES
        reset_env!(env)
        fill_ideal_batch(env, NUM_POLYS, MAX_DEGREE, NUM_TERMS, MAX_ATTEMPTS) # fill with random ideals
        s = Float32.(state(env))
        done = false
        t = 0
        episode_loss = []

        while !done # check it out 
            epsilon = randn() * STD
            action = Float32.(actor.actor(s) .+ epsilon)
            basis = act!(env, action)

            s_next = Float32.(state(env))
            push!(actions_taken, s_next)

            r = Float32(env.reward)
            push!(rewards, r)

            done = is_terminated(env)
            s_next = done ? nothing : s_next

            push!(replay_buffer, Transition(s, action, r, s_next))

            s = s_next === nothing ? s : s_next

            if length(replay_buffer) < N_SAMPLES
                continue
            end

            batch = rand(replay_buffer, N_SAMPLES)
            s_batch = hcat([b.s for b in batch]...)
            a_batch = hcat([b.a for b in batch]...)
            r_batch = hcat([b.r for b in batch]...)

            next_s_batch = hcat(
                [b.s_next !== nothing ? b.s_next : zeros(Float32, env.numVars) for b in batch]...,
            )
            not_done = reshape(Float32.(getfield.(batch, :s_next) .!== nothing), 1, :)

            epsilon = clamp.(randn(1, N_SAMPLES) * STD, -0.5f0, 0.5f0)
            target_action = actor.actor_target(next_s_batch) .+ epsilon

            target_action = Float32.(target_action)
            cols_as_vectors = [Vector{Float32}(col) for col in eachcol(target_action)]
            target_action = [make_valid_action(env, col) for col in cols_as_vectors]
            target_action = reduce(hcat, target_action)

            critic_1_target_val = critic.critic_1_target(vcat(next_s_batch, target_action))
            critic_2_target_val = critic.critic_2_target(vcat(next_s_batch, target_action))

            min_q = min.(critic_1_target_val, critic_2_target_val)

            y = r_batch .+ GAMMA .* not_done .* min_q

            loss1, back1 = Flux.withgradient(critic.critic_1) do model
                pred = model(vcat(s_batch, a_batch))
                mean((pred .- y) .^ 2)
            end

            Flux.update!(critic.critic_1_opt_state, critic.critic_1, back1[1])

            loss2, back2 = Flux.withgradient(critic.critic_2) do model
                pred = model(vcat(s_batch, a_batch))
                mean((pred .- y) .^ 2)
            end

            Flux.update!(critic.critic_2_opt_state, critic.critic_2, back2[1])

            if t % D == 0
                actor_loss, back = Flux.withgradient(actor.actor) do model
                    a_pred = model(s_batch)
                    q_val = critic.critic_1(vcat(s_batch, a_pred))
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
            avg_loss = mean(episode_loss)
            push!(losses, avg_loss)
            if i % 100 == 0
                println("Episode: $i, Action Taken: ", actions_taken[i], " Loss: $avg_loss, Reward: ", env.reward)
                println()
            end
        end
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

    savefig(loss_plot, "loss_plot.png")

    episodes2 = 1:length(rewards)
    reward_plot = plot(episodes2, rewards,
        title = "Reward plot",
        xlabel = "Episode",
        ylabel = "Reward",
        label = "Reward",
        lw = 1,
        marker = false,
        legend = :topright)

    savefig(reward_plot, "reward_plot.png")
end

function soft_update!(target, policy)
    for (tp, pp) in zip(Flux.params(target), Flux.params(policy))
        tp .= (1 - TAU) * tp .+ TAU * pp
    end
end
