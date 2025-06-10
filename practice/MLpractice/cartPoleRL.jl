using ReinforcementLearning
import ReinforcementLearning: reset!, act!, state, reward, is_terminated, action_space
using DataStructures
using Flux
using Random
import Plots
import .Plots.gui
import .Plots.plot
import .Plots.plot!
import .Plots.annotate!

env = CartPoleEnv(max_steps=500, thetathreshold=30)
NUM_STATES = length(state(env))
NUM_ACTIONS = length(action_space(env))

CAPACITY = 10_000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000.0
TAU = 0.005
LR = 1e-4
NUM_EPISODES = 600

struct Transition
    s::Vector{Float32}
    a::Int
    next_s::Union{Vector{Float32},Nothing}
    r::Float32
end

replay_buffer = CircularBuffer{Transition}(CAPACITY)

function select_action(policy_net, s, steps)
    eps = EPS_END + (EPS_START - EPS_END) * exp(-steps / EPS_DECAY)
    if rand() > eps
        q = policy_net(s)
        a = argmax(q)
    else
        a = rand(action_space(env))
    end
    return a, steps + 1
end

function soft_update!(target, policy)
    for (tp, pp) in zip(Flux.params(target), Flux.params(policy))
        tp .= (1 - TAU) * tp .+ TAU * pp
    end
end

function optimize_model!(policy_net, target_net, opt_state)
    if length(replay_buffer) < BATCH_SIZE
        return opt_state
    end

    batch = rand(replay_buffer, BATCH_SIZE)

    state_batch = hcat([t.s for t in batch]...)
    reward_batch = Float32[t.r for t in batch]
    action_batch = [t.a for t in batch]
    nonfinal_mask = [t.next_s !== nothing for t in batch]
    next_states = hcat([t.next_s for t in batch if t.next_s !== nothing]...)

    loss, back = Flux.withgradient(policy_net) do m
        q_vals = m(state_batch)
        q_sa = [q_vals[action_batch[i], i] for i = 1:BATCH_SIZE]

        q_next =
            isempty(next_states) ? Float32[] :
            vec(maximum(target_net(next_states), dims = 1))

        mask_int = Int.(nonfinal_mask)
        counts = cumsum(mask_int)
        y = [
            nonfinal_mask[i] ? reward_batch[i] + GAMMA * q_next[counts[i]] :
            reward_batch[i] for i = 1:BATCH_SIZE
        ]

        sum(Flux.Losses.huber_loss.(q_sa, y)) / BATCH_SIZE
    end

    Flux.update!(opt_state, policy_net, back[1])

    return opt_state
end

function plot_env(env::CartPoleEnv; kwargs...)
    a, d = env.action, env.done
    x, xdot, theta, thetadot = env.state
    l = 2 * env.params.halflength
    xthreshold = env.params.xthreshold
    plot(
        xlims = (-xthreshold, xthreshold),
        ylims = (-0.1, l + 0.1),
        legend = false,
        border = :none,
    )
    plot!([x - 0.5, x - 0.5, x + 0.5, x + 0.5], [-0.05, 0, 0, -0.05]; seriestype = :shape)
    plot!([x, x + l * sin(theta)], [0, l * cos(theta)]; linewidth = 3)
    if isa(action_space(env), Base.OneTo)
        a = a == 2 ? 1 : -1
    end
    plot!(
        [x + sign(a)*0.5, x + sign(a)*0.7],
        [-0.025, -0.025];
        linewidth = 3,
        arrow = true,
        color = :black,
    )
    if d
        color = :pink
        if env.t > env.params.max_steps
            color = :green
        end
        plot!(
            [xthreshold - 0.2],
            [l];
            marker = :circle,
            markersize = 20,
            markerstrokewidth = 0.0,
            color = color,
        )
    end

    p = plot!(; kwargs...)
    gui(p)
    p
end

function main()
    policy_net =
        Chain(Dense(NUM_STATES, 128, relu), Dense(128, 128, relu), Dense(128, NUM_ACTIONS))
    target_net = deepcopy(policy_net)

    opt = ADAM(LR)
    opt_state = Flux.setup(opt, policy_net)
    steps = 0
    episode_rewards = Float32[]

    for episode = 1:NUM_EPISODES
        reset!(env)
        s = Float32.(state(env))
        done = false
        total_reward = 0.0f0

        while !done
            a, steps = select_action(policy_net, s, steps)
            act!(env, a)

            s_next = Float32.(state(env))
            r = Float32(reward(env))
            done = is_terminated(env)
            next_s = done ? nothing : s_next

            push!(replay_buffer, Transition(s, a, next_s, r))
            s = next_s === nothing ? s : next_s
            total_reward += r

            opt_state = optimize_model!(policy_net, target_net, opt_state)
            soft_update!(target_net, policy_net)
            plot_env(env)
        end

        push!(episode_rewards, total_reward)
        println("Episode $episode: Reward = $total_reward")

    end

    plot(
        episode_rewards,
        title = "DQN on CartPole",
        xlabel = "Episode",
        ylabel = "Reward",
        legend = false,
    )
end

main()
