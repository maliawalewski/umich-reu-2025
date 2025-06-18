using Flux
using DataStructures
using ReinforcementLearning
using Statistics
using Optimisers
using Plots
include("environment.jl")

CAPACITY = 1_000_000
EPISODES = 1000 #number of ideals 
N_SAMPLES = 256
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
STD = 0.2
D = 2

struct Transition
    s::Vector{Float32}
    a::Float32{Float32}
    r::Float32
    s_next::Union{Vector{Float32},Nothing}
end

struct Actor
    actor::Flux.Chain
    actor_target::Flux.Chain
    actor_opt_state::Optimisers.OptimiserState
end

struct Critics
    critic_1::Flux.Chain
    critic_2::Flux.Chain

    critic_1_target::Flux.Chain
    critic_2_target::Flux.Chain

    critic_1_opt_state::Optimisers.OptimiserState
    critic_2_opt_state::Optimisers.OptimiserState
end

function build_td3_model(env::Environment)
    actor = Flux.Chain(Dense(env.numVars, 128, relu), Dense(128, 128, relu), Dense(128, env.numVars, tanh)) # TODO fix tanh output to match action space

    critic_1 = Flux.Chain(Dense(2 * env.numVars, 128, relu), Dense(128, 128, relu), Dense(128, 1))
    critic_2 = Flux.Chain(Dense(2 * env.numVars, 128, relu), Dense(128, 128, relu), Dense(128, 1))

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



function soft_update!(target, policy)
    for (tp, pp) in zip(Flux.params(target), Flux.params(policy))
        tp .= (1 - TAU) * tp .+ TAU * pp
    end
end

function main()
    
    losses = []

    replay_buffer = CircularBuffer{Transition}(CAPACITY)

    for i = 1:EPISODES
        reset!(env)
        s = Float32.(state(env))
        done = false
        total_reward = 0.0f0

        t = 0
        episode_loss = []
        while !done
            epsilon = randn() * STD
            action = actor(s) .+ epsilon
            action = make_valid_action(action, env)

            act!(env, action)

            s_next = Float32.(state(env))
            r = Float32(reward(env))
            done = is_terminated(env)
            s_next = done ? nothing : s_next

            push!(replay_buffer, Transition(s, action, r, s_next))

            s = s_next === nothing ? s : s_next
            total_reward += r

            if length(replay_buffer) < N_SAMPLES
                continue
            end

            batch = rand(replay_buffer, N_SAMPLES)
            s_batch = hcat([b.s for b in batch]...)
            a_batch = hcat([b.a for b in batch]...)
            r_batch = hcat([b.r for b in batch]...)

            next_s_batch = hcat(
                [b.s_next !== nothing ? b.s_next : zeros(Float32, 3) for b in batch]...,
            )
            not_done = reshape(Float32.(getfield.(batch, :s_next) .!== nothing), 1, :)

            epsilon = clamp.(randn(1, N_SAMPLES) * STD, -0.5f0, 0.5f0)
            target_action =
                clamp.(2.0f0 * actor_target(next_s_batch) .+ epsilon, -2.0f0, 2.0f0)

            critic_1_target_val = critic_1_target(vcat(next_s_batch, target_action))
            critic_2_target_val = critic_2_target(vcat(next_s_batch, target_action))

            min_q = min.(critic_1_target_val, critic_2_target_val)

            y = r_batch .+ GAMMA .* not_done .* min_q

            loss1, back1 = Flux.withgradient(critic_1) do model
                pred = model(vcat(s_batch, a_batch))
                mean((pred .- y) .^ 2)
            end

            Flux.update!(critic_1_opt_state, critic_1, back1[1])

            loss2, back2 = Flux.withgradient(critic_2) do model
                pred = model(vcat(s_batch, a_batch))
                mean((pred .- y) .^ 2)
            end

            Flux.update!(critic_2_opt_state, critic_2, back2[1])

            if t % D == 0
                actor_loss, back = Flux.withgradient(actor) do model
                    a_pred = model(s_batch)
                    q_val = critic_1(vcat(s_batch, a_pred))
                    -mean(q_val)
                end

                push!(episode_loss, actor_loss)
                grads = back[1]
                Flux.update!(opt_state, actor, grads)

                soft_update!(critic_1_target, critic_1)
                soft_update!(critic_2_target, critic_2)
                soft_update!(actor_target, actor)
            end

            t += 1

        end

        if length(episode_loss) != 0
            push!(losses, mean(episode_loss))
        end

        if i % 1 == 0 && length(losses) > 0 && i > 1
            i_loss = losses[i-1]
            println("Episode: $i, Loss: $i_loss, Reward: $total_reward")
        end
    end

    episodes = 1:length(losses)
    p = plot(
        episodes,
        losses,
        title = "Loss/t",
        xlabel = "Episode",
        ylabel = "Loss",
        label = "Loss",
        lw = 2,
        marker = :circle,
        legend = :topright,
    )

    savefig(p, "loss_plot.png")

end

main()
