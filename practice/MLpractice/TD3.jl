using Flux
using DataStructures
using ReinforcementLearning

env = PendulumEnv()

CAPACITY = 10000
EPISODES = 1
N_SAMPLES = 100
GAMMA = 0.99
TAU = 0.005
LR = 0.001
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

function main() 
    actor = Flux.Chain(Dense(3, 128, relu), Dense(128, 128, relu), Dense(128, 1, tanh))
  
    q_theta_1 = Flux.Chain(Dense(4, 128, relu), Dense(128, 128, relu), Dense(128, 1))
    q_theta_2 = Flux.Chain(Dense(4, 128, relu), Dense(128, 128, relu), Dense(128, 1))

    target_actor = deepcopy(actor)
    target_q_theta_1 = deepcopy(q_theta_1)
    target_q_theta_2 = deepcopy(q_theta_2)

    opt = ADAM(LR)
    opt_state = Flux.setup(opt, actor)
    
    replay_buffer = CircularBuffer{Transition}(CAPACITY)

    for i in 1:EPISODES
        ReinforcementLearning.reset!(env)
        s = Float32.(state(env))
        done = false
        total_reward = 0.0f0
        
        t = 0
        while !done
            epsilon = randn() * STD
            action = 2.0f0 * actor(s)[1] + epsilon
            action = clamp(action, -2.0f0, 2.0f0)

            act!(env, action)

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
            next_s_batch = hcat([b.next_s for b in batch if b.next_s !== nothing]...)
            
            epsilon = clamp.(randn(1, N_SAMPLES) * STD, -0.5f0, 0.5f0)
            target_action = 2.0f0 * target_actor(next_s_batch) .+ epsilon
            target_action = clamp.(target_action, -2.0f0, 2.0f0)

            q_input = vcat(next_s_batch, target_action)

            q_val_1 = target_q_theta_1(q_input)
            q_val_2 = target_q_theta_2(q_input)

            q1 = q_val_1[1]
            q2 = q_val_2[1]

            y = r_batch .+ GAMMA * min.(q1, q2)                

            online_q_input = vcat(s_batch, a_batch)
            q_online_val_1 = q_theta_1(online_q_input)
            q_online_val_2 = q_theta_2(online_q_input)

            sum_q_online_1 = (1 / N_SAMPLES) * sum((y - q_online_val_1) .^ 2)
            sum_q_online_2 = (1 / N_SAMPLES) * sum((y - q_online_val_2) .^ 2)
            
            if (sum_q_online_1 < sum_q_online_2)
                q_theta_2 = deepcopy(q_theta_1)
            else
                q_theta_1 = deepcopy(q_theta_2)
            end

            if t % D == 0 
                
                # TODO: actor update 

                soft_update!(target_q_theta_1, q_theta_1)
                soft_update!(target_q_theta_2, q_theta_2)
                soft_update!(target_actor, actor)
            end

            t += 1

        end
    end
end

main()
