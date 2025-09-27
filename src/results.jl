using Statistics
using Serialization
include("model.jl")
include("environment.jl")

function compute_stats(agent_rewards, baseline_rewards, name)
    total = length(agent_rewards)
    wins = [a > b for (a, b) in zip(agent_rewards, baseline_rewards)]
    win_pct = 100 * count(wins) / total

    improvements = [
        (a - b) / abs(b + 1e-8) * 100 for
        (a, b) in zip(agent_rewards, baseline_rewards) if a > b
    ]
    avg_improvement = isempty(improvements) ? 0.0 : mean(improvements)

    println("Agent vs $name:")
    println("Percent of time agent wins: $(round(win_pct, digits=2))%")
    println("Average improvement percent: $(round(avg_improvement, digits=2))%")
    println()
end

function interpret_results()
    if !@isdefined(RESULTS_DIR)
        BASE_DIR = @__DIR__
        RESULTS_DIR = joinpath(BASE_DIR, "results")
    end

    best_order = deserialize(joinpath(RESULTS_DIR, "agent_order.bin"))
    agent_rewards = deserialize(joinpath(RESULTS_DIR, "agent_rewards.bin"))
    deglex_rewards = deserialize(joinpath(RESULTS_DIR, "deglex_rewards.bin"))
    grevlex_rewards = deserialize(joinpath(RESULTS_DIR, "grevlex_rewards.bin"))

    println("Agent order: $best_order")

    int_best_order = Int.(round.(ACTION_SCALE * best_order))
    int_best_order = max.(int_best_order, 1)
    println("Agent order (int): $int_best_order")

    agent_rewards = Float64.(agent_rewards)
    deglex_rewards = Float64.(deglex_rewards)
    grevlex_rewards = Float64.(grevlex_rewards)

    compute_stats(agent_rewards, deglex_rewards, "DegLex")
    compute_stats(agent_rewards, grevlex_rewards, "DegRevLex")
end
