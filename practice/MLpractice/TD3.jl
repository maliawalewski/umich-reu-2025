using Flux
using DataStructures
using ReinforcementLearning

env = PendulumEnv()

CAPACITY = 10000
EPISODES = 60000
N_SAMPLES = 100
GAMMA = 0.99
LR = 0.001
STD = 0.2

struct Transition
    s::Vector{Float32}
    a::Float32
    r::Float32
    next_s::Union{Vector{Float32},Nothing}
end


function main() 
    actor = Flux.Chain(Dense(3, 128, relu), Dense(128, 128, relu), Dense(128, 1, tanh))
  
    q_theta_1 = Flux.Chain(Dense(4, 128, relu), Dense(128, 128, relu), Dense(128, 1))
    q_theta_2 = Flux.Chain(Dense(4, 128, relu), Dense(128, 128, relu), Dense(128, 1))

    target_actor = deepcopy(actor)
    target_q_theta_1 = deepcopy(q_theta_1)
    target_q_theta_2 = deepcopy(q_theta_2)
    
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

            
            batch = rand(replay_buffer, N_SAMPLES)
            # TODO: use batch below

            epsilon = clamp(randn() * STD, -0.5f0, 0.5f0)
            target_action = 2.0f0 * target_actor(next_s)[1] + epsilon
            target_action = clamp(target_action, -2.0f0, 2.0f0)

            println(next_s, target_action)

            q_input = vcat(next_s, Float32(target_action))
            q_input = reshape(q_input, :, 1)

            q_val_1 = target_q_theta_1(q_input)
            q_val_2 = target_q_theta_2(q_input)

            q1 = q_val_1[1]
            q2 = q_val_2[1]

            y = r + GAMMA * min(q1, q2)



            if t % 2 == 0 
                # update 

            end

            t += 1

        end
    end
end

main()
