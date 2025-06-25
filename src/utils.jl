using StatsBase

# Reference: https://github.com/JuliaPOMDP/DeepQLearning.jl/blob/master/src/prioritized_experience_replay.jl
# Reference: https://github.com/Ullar-Kask/TD3-PER/blob/master/Pytorch/src/PER.py

struct Transition
    s::Array{Float32}
    a::Vector{Float32}
    r::Float32
    s_next::Union{Array{Float32}, Nothing}
end

mutable struct PrioritizedReplayBuffer
    capacity::Int64
    batch_size::Int64
    alpha::Float32
    beta::Float32
    beta_increment::Float32
    eps::Float32
    _priorities::Vector{Float32}
    _experiences::Vector{Transition}
    _weights_batch::Vector{Float32}
    _curr_size::Int64
    _curr_idx::Int64
end

function PrioritizedReplayBuffer(capacity::Int64, batch_size::Int64, alpha::Float32, beta::Float32, beta_increment::Float32, eps::Float32)
    @assert capacity >= batch_size "cannot initialize buffer where capacity is less than batch_size"
    curr_size, curr_idx = 0, 1 
    priorities::Vector{Float32} = Vector{Float32}(undef, capacity)
    experiences::Vector{Transition} = Vector{Transition}(undef, capacity)
    weights_batch = zeros(Float32, batch_size)
    return PrioritizedReplayBuffer(capacity, batch_size, alpha, beta, beta_increment, eps, priorities, experiences, weights_batch, curr_size, curr_idx)
end

function add_experience!(buffer::PrioritizedReplayBuffer, experience::Transition, td_error::Float32)
    @assert td_error + buffer.eps > 0 "td_error + epsilon must be greater than 0"
    priority = (td_error + buffer.eps) ^ buffer.alpha
    buffer._experiences[buffer._curr_idx] = experience 
    buffer._priorities[buffer._curr_idx] = priority
    buffer._curr_idx = ((buffer._curr_idx + 1) % buffer.capacity)
    if buffer._curr_size < buffer.capacity
        buffer._curr_size += 1
    end
end

function update_priorities!(buffer::PrioritizedReplayBuffer, indices::Vector{Int64}, td_errors::Vector{Float32})
    @assert length(indices) == length(td_errors) "length of indices and td_errors are not equal"
    priorities = (abs.(td_errors) .+ buffer.eps) .^ buffer.alpha
    @assert all(priorities .> 0f0) "td_errors + epsilon must be greater than 0"
    buffer._priorities[indices] = priorities
end

function StatsBase.sample(buffer::PrioritizedReplayBuffer)
    @assert buffer.curr_size >= buffer.batch_size "unable to return " + buffer.batch_size + " samples from replay buffer of size " + buffer.curr_size
    sample_indices = sample(1:buffer._curr_size, StatsBase.Weights(buffer._priorities[1:buffer._curr_size]), buffer.batch_size, replace=false)
    return get_batch(buffer, sample_indices)
end

function get_batch(buffer::PrioritizedReplayBuffer, sample_indices::Vector{Int64})
    # TODO: do we want to sample without replacement? 
    @assert length(sample_indices) == length(buffer.batch_size) "length of sample_indices does not match batch_size"
    samples = buffer._experiences[sample_indices]
    buffer._weights_batch[sample_indices] = buffer._priorities[sample_indices]
    p = buffer._weights_batch ./ sum(buffer._priorities[1:buffer._curr_size])
    weights = (r._curr_size * p) .^ (-buffer.beta)
    buffer.beta = min(1, buffer.beta + buffer.beta_increment)
    return samples, weights
end
