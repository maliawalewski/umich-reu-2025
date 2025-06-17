using Flux
using DataStructures
using ReinforcementLearning
using Statistics
using Optimisers
using Plots

env = PendulumEnv()

CAPACITY = 1_000_000
EPISODES = 1000
N_SAMPLES = 256
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
STD = 0.2
D = 2

struct Transition
    s::Vector{Float32}
    a::Float32
    r::Float32
    next_s::Union{Vector{Float32},Nothing}
end

function soft_update!(target, policy)
    for (tp, pp) in zip(Flux.params(target), Flux.params(policy))
        tp .= (1 - TAU) * tp .+ TAU * pp
    end
end

function plot_pendulum(env::PendulumEnv; kwargs...)
    s = state(env) 
    cosθ, sinθ, θ̇ = s

    θ = atan(sinθ, cosθ)
    l = 1.0

    x = l * sin(θ)
    y = -l * cos(θ)

    plot(
        xlims = (-l - 0.2, l + 0.2),
        ylims = (-l - 0.2, l + 0.2),
        aspect_ratio = :equal,
        legend = false,
        title = "Pendulum",
        size = (300, 300),
        background_color = :white,
        grid = false,
        framestyle = :none
    )

    plot!([0, x], [0, y]; lw = 3, color = :black)
    scatter!([x], [y]; markersize = 10, color = :red)

    gui()
end

function main() 
    actor = Flux.Chain(Dense(3, 128, relu), Dense(128, 128, relu), Dense(128, 1, tanh))
  
    q_theta_1 = Flux.Chain(Dense(4, 128, relu), Dense(128, 128, relu), Dense(128, 1))
    q_theta_2 = Flux.Chain(Dense(4, 128, relu), Dense(128, 128, relu), Dense(128, 1))

    target_actor = deepcopy(actor)
    target_q_theta_1 = deepcopy(q_theta_1)
    target_q_theta_2 = deepcopy(q_theta_2)

    opt = ADAM(LR)
    opt_state = Flux.setup(opt, actor)

    critic_opt1 = ADAM(LR)
    critic_opt2 = ADAM(LR)
    
    critic_state1 = Flux.setup(critic_opt1, q_theta_1)
    critic_state2 = Flux.setup(critic_opt2, q_theta_2)

    losses = []
    
    replay_buffer = CircularBuffer{Transition}(CAPACITY)

    for i in 1:EPISODES
        ReinforcementLearning.reset!(env)
        s = Float32.(state(env))
        done = false
        total_reward = 0.0f0
        
        t = 0
        episode_loss = []
        while !done
            epsilon = randn() * STD
            action = 2.0f0 * actor(s)[1] + epsilon
            action = clamp(action, -2.0f0, 2.0f0)

            act!(env, action)
            plot_pendulum(env)

            s_next = Float32.(state(env))
            r = Float32(reward(env))
            done = is_terminated(env)
            next_s = done ? nothing : s_next

            push!(replay_buffer, Transition(s, action, r, next_s))

            s = next_s === nothing ? s : next_s
            total_reward += r

            if length(replay_buffer) < N_SAMPLES
                continue
            end

            batch = rand(replay_buffer, N_SAMPLES)
            s_batch = hcat([b.s for b in batch]...)
            a_batch = hcat([b.a for b in batch]...)
            r_batch = hcat([b.r for b in batch]...)
            
            next_s_batch = hcat([b.next_s !== nothing ? b.next_s : zeros(Float32, 3) for b in batch]...)
            not_done = reshape(Float32.(getfield.(batch, :next_s) .!== nothing), 1, :)

            epsilon = clamp.(randn(1, N_SAMPLES) * STD, -0.5f0, 0.5f0)
            target_action = clamp.(2.0f0 * target_actor(next_s_batch) .+ epsilon, -2f0, 2f0)

            q_val_1 = target_q_theta_1(vcat(next_s_batch, target_action))
            q_val_2 = target_q_theta_2(vcat(next_s_batch, target_action))

            min_q = min.(q_val_1, q_val_2)

            y = r_batch .+ GAMMA .* not_done .* min_q

            online_q_input = vcat(s_batch, a_batch)
            q_online_val_1 = q_theta_1(online_q_input)
            q_online_val_2 = q_theta_2(online_q_input)

            loss1, back1 = Flux.withgradient(q_theta_1) do model
                pred = model(vcat(s_batch, a_batch))
                mean((pred .- y).^2)
            end

            Flux.update!(critic_state1, q_theta_1, back1[1])

            loss2, back2 = Flux.withgradient(q_theta_2) do model
                pred = model(vcat(s_batch, a_batch))
                mean((pred .- y).^2)
            end
            
            Flux.update!(critic_state2, q_theta_2, back2[1])

            if t % D == 0
                actor_loss, back = Flux.withgradient(actor) do model
                    a_pred = model(s_batch)
                    q_val = q_theta_1(vcat(s_batch, a_pred))
                    -mean(q_val)
                end

                push!(episode_loss, actor_loss)
                grads = back[1]
                Flux.update!(opt_state, actor, grads)

                soft_update!(target_q_theta_1, q_theta_1)
                soft_update!(target_q_theta_2, q_theta_2)
                soft_update!(target_actor, actor)
            end

            t += 1

        end

        if length(episode_loss) != 0 
            push!(losses, mean(episode_loss))
        end

        if i % 1 == 0 && length(losses) > 0 && i > 1
            i_loss = losses[i - 1]
            println("Episode: $i, Loss: $i_loss, Reward: $total_reward")
        end
    end

    episodes = 1:length(losses)
    p = plot(episodes, losses,
        title = "Loss/t",
        xlabel = "Episode",
        ylabel = "Loss",
        label = "Loss",
        lw = 2,
        marker = :circle,
        legend = :topright)

    savefig(p, "loss_plot.png")

end

main()
