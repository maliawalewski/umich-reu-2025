using POMDPs, Crux, Flux
import POMDPTools: Deterministic, ImplicitDistribution, EpsGreedyPolicy, LinearDecaySchedule
import QuickPOMDPs: QuickPOMDP
import ReinforcementLearningEnvironments: PendulumEnv, state, act!, reset!, is_terminated, reward
import Random
using Functors
import Zygote: Grads

actor_net() = ContinuousNetwork(Chain(Dense(3, 64, relu), 
    Dense(64, 64, relu), 
    Dense(64, 1, tanh), 
    x -> 2f0 * x), 
    1)

critic_net() = ContinuousNetwork(Chain(Dense(4, 64, relu),
    Dense(64, 64, relu), 
    Dense(64, 1)))

# Define POMDP version of PendulumEnv
mdp = QuickPOMDP(
    actions = [-2.0f0, -1.0f0, 0.0f0, 1.0f0, 2.0f0],  # discrete actions for Crux
    discount = 0.99f0,
    gen = function (s, a, rng)
        sp = deepcopy(s)
        act!(sp, a)
        o = state(sp)
        r = reward(sp)
        (;sp, o, r)
    end,
    initialstate = ImplicitDistribution(rng -> PendulumEnv()),
    isterminal = is_terminated,
    initialobs = s -> Deterministic(state(s))
)

S = ContinuousSpace(3; σ=[1f0, 1f0, 1f0])

c_o = Adam(1e-4)
c_opt_state = Flux.setup(c_o, critic_net())

a_o = Adam(1e-3)
a_opt_state = Flux.setup(a_o, actor_net())

off_policy = (S=S,
              ΔN=50,
              N=60000,
              buffer_size=Int(5e5),
              buffer_init=1000,
              c_opt=(batch_size=100, optimizer=c_o),
              a_opt=(batch_size=100, optimizer=a_o),
              π_explore=GaussianNoiseExplorationPolicy(0.5f0, a_min=[-2.0], a_max=[2.0]))

solver = TD3(;π=Crux.ActorCritic(actor_net(), DoubleNetwork(critic_net(), critic_net())), S=S, off_policy...)

@functor Grads
td3 = solve(solver, mdp)
