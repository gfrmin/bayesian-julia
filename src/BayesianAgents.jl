"""
    BayesianAgents

A framework for Bayes-Adaptive POMDPs — agents that maintain uncertainty over
both world state and world dynamics, planning in a way that naturally balances
exploration and exploitation.

All behaviour derives from expected utility maximisation. No hacks.
"""
module BayesianAgents

using Random
using Distributions
using LinearAlgebra
using Logging

# ============================================================================
# CORE ABSTRACT TYPES
# ============================================================================

"""
    World

Abstract interface for environments the agent can interact with.
Implementations: JerichoWorld, GridWorld, GymnasiumWorld, etc.
"""
abstract type World end

"""
    Sensor

Abstract interface for information sources with learnable reliability.
Implementations: LLMSensor, HeuristicSensor, OracleSensor, etc.
"""
abstract type Sensor end

"""
    WorldModel

Abstract interface for Bayesian models of world dynamics.
Implementations: TabularWorldModel, GPWorldModel, NeuralWorldModel, etc.
"""
abstract type WorldModel end

"""
    Planner

Abstract interface for planning algorithms.
Implementations: ThompsonMCTS, POMCP, ValueIteration, etc.
"""
abstract type Planner end

"""
    StateAbstractor

Abstract interface for learning state equivalence classes.
Implementations: BisimulationAbstractor, IdentityAbstractor, etc.
"""
abstract type StateAbstractor end

# ============================================================================
# WORLD INTERFACE
# ============================================================================

"""
    reset!(world::World) → observation

Reset the world to its initial state and return the initial observation.
"""
function reset! end

"""
    step!(world::World, action) → (observation, reward, done, info)

Execute an action in the world and return the result.
"""
function step! end

"""
    actions(world::World, observation) → Vector

Return the available actions given the current observation.
"""
function actions end

"""
    render(world::World) → String

Return a human-readable representation of the world state.
Optional — defaults to empty string.
"""
render(::World) = ""

"""
    seed!(world::World, seed::Int)

Set the random seed for the world. Optional.
"""
seed!(::World, ::Int) = nothing

# ============================================================================
# SENSOR INTERFACE
# ============================================================================

"""
    query(sensor::Sensor, state, question) → answer

Query the sensor about a question given the current state.
"""
function query end

"""
    tpr(sensor::Sensor) → Float64

Return the true positive rate: P(positive | true).
"""
function tpr end

"""
    fpr(sensor::Sensor) → Float64

Return the false positive rate: P(positive | false).
"""
function fpr end

"""
    update_reliability!(sensor::Sensor, predicted::Bool, actual::Bool)

Update the sensor's reliability estimates from ground truth.
"""
function update_reliability! end

"""
    posterior(sensor::Sensor, prior::Float64, answer::Bool) → Float64

Compute the posterior probability given prior and sensor answer.
Uses Bayes' rule with learned TPR/FPR.
"""
function posterior(sensor::Sensor, prior::Float64, answer::Bool)::Float64
    t = tpr(sensor)
    f = fpr(sensor)
    
    if answer  # Sensor said "yes"
        numerator = t * prior
        denominator = t * prior + f * (1 - prior)
    else  # Sensor said "no"
        numerator = (1 - t) * prior
        denominator = (1 - t) * prior + (1 - f) * (1 - prior)
    end
    
    return denominator > 0 ? numerator / denominator : prior
end

# ============================================================================
# WORLD MODEL INTERFACE
# ============================================================================

"""
    sample_dynamics(model::WorldModel) → sampled_model

Sample a concrete dynamics model from the posterior for Thompson Sampling.
"""
function sample_dynamics end

"""
    update!(model::WorldModel, s, a, r, s′)

Update the model posterior with an observed transition.
"""
function update! end

"""
    transition_dist(model::WorldModel, s, a) → Distribution

Return the posterior predictive distribution over next states.
"""
function transition_dist end

"""
    reward_dist(model::WorldModel, s, a) → Distribution

Return the posterior predictive distribution over rewards.
"""
function reward_dist end

"""
    entropy(model::WorldModel) → Float64

Return the entropy of the model posterior (uncertainty measure).
"""
function entropy end

# ============================================================================
# STATE ABSTRACTOR INTERFACE
# ============================================================================

"""
    abstract_state(abstractor::StateAbstractor, observation) → abstract_state

Map a concrete observation to an abstract state.
"""
function abstract_state end

"""
    record_transition!(abstractor::StateAbstractor, s, a, r, s′)

Record a transition for learning equivalence classes.
"""
function record_transition! end

"""
    check_contradiction(abstractor::StateAbstractor) → Option{Contradiction}

Check for contradictions (same abstract state, different outcomes).
"""
function check_contradiction end

"""
    refine!(abstractor::StateAbstractor, contradiction)

Refine the abstraction to resolve a contradiction.
"""
function refine! end

# ============================================================================
# PLANNER INTERFACE
# ============================================================================

"""
    plan(planner::Planner, belief, world_model, actions) → action

Plan and return the best action given current belief and model.
"""
function plan end

# ============================================================================
# VALUE OF INFORMATION
# ============================================================================

"""
    compute_voi(sensor::Sensor, prior::Float64, actions, eu_func) → Float64

Compute the value of information for querying a binary sensor.

VOI = E[max_a EU(a) | after asking] - max_a EU(a) | now

Arguments:
- sensor: The sensor to query
- prior: Current belief P(proposition is true)
- actions: Available actions
- eu_func: Function mapping (action, belief) → expected utility
"""
function compute_voi(sensor::Sensor, prior::Float64, actions, eu_func)
    t = tpr(sensor)
    f = fpr(sensor)
    
    # Current best EU
    current_best = maximum(a -> eu_func(a, prior), actions)
    
    # Probability sensor says "yes"
    p_yes = t * prior + f * (1 - prior)
    p_no = 1 - p_yes
    
    # Posterior if sensor says "yes"
    posterior_yes = posterior(sensor, prior, true)
    best_after_yes = maximum(a -> eu_func(a, posterior_yes), actions)
    
    # Posterior if sensor says "no"
    posterior_no = posterior(sensor, prior, false)
    best_after_no = maximum(a -> eu_func(a, posterior_no), actions)
    
    # Expected best after asking
    expected_best_after = p_yes * best_after_yes + p_no * best_after_no
    
    return expected_best_after - current_best
end

# ============================================================================
# CONFIGURATION
# ============================================================================

"""
    AgentConfig

Configuration for the Bayesian agent.
"""
Base.@kwdef struct AgentConfig
    # Planning
    planning_depth::Int = 10
    mcts_iterations::Int = 100
    discount::Float64 = 0.99
    ucb_c::Float64 = 2.0
    
    # Sensors
    sensor_cost::Float64 = 0.01
    max_queries_per_step::Int = 10
    
    # Priors
    transition_prior_strength::Float64 = 0.1
    reward_prior_mean::Float64 = 0.0
    reward_prior_variance::Float64 = 1.0
    
    # State abstraction
    abstraction_threshold::Float64 = 0.95
    
    # Intrinsic motivation
    use_intrinsic_reward::Bool = true
    intrinsic_scale::Float64 = 0.1
end

# ============================================================================
# AGENT
# ============================================================================

"""
    BayesianAgent

The main agent that ties everything together.
"""
mutable struct BayesianAgent{W<:World, M<:WorldModel, P<:Planner, A<:StateAbstractor}
    world::W
    model::M
    planner::P
    abstractor::A
    sensors::Vector{Sensor}
    config::AgentConfig

    # State
    current_observation::Any
    current_abstract_state::Any
    step_count::Int
    episode_count::Int
    total_reward::Float64

    # History for credit assignment
    trajectory::Vector{NamedTuple{(:s, :a, :r, :s′), Tuple{Any, Any, Float64, Any}}}

    # Persistent action beliefs: (abstract_state, action) → P(action helps)
    # Survives across steps and episodes so sensor learning is not lost
    action_belief_cache::Dict{Tuple{Any,Any}, Float64}
end

"""
    BayesianAgent(world, model, planner, abstractor; sensors=[], config=AgentConfig())

Construct a new Bayesian agent.
"""
function BayesianAgent(
    world::World,
    model::WorldModel,
    planner::Planner,
    abstractor::StateAbstractor;
    sensors::Vector{<:Sensor} = Sensor[],
    config::AgentConfig = AgentConfig()
)
    return BayesianAgent(
        world, model, planner, abstractor, convert(Vector{Sensor}, sensors), config,
        nothing, nothing, 0, 0, 0.0,
        NamedTuple{(:s, :a, :r, :s′), Tuple{Any, Any, Float64, Any}}[],
        Dict{Tuple{Any,Any}, Float64}()
    )
end

"""
    reset!(agent::BayesianAgent)

Reset the agent for a new episode.
"""
function reset!(agent::BayesianAgent)
    agent.current_observation = reset!(agent.world)
    agent.current_abstract_state = abstract_state(agent.abstractor, agent.current_observation)
    agent.step_count = 0
    agent.episode_count += 1
    agent.total_reward = 0.0
    empty!(agent.trajectory)
    return agent.current_observation
end

"""
    act!(agent::BayesianAgent) → (action, observation, reward, done)

Choose and execute an action via expected utility maximisation.
"""
function act!(agent::BayesianAgent)
    s = agent.current_abstract_state
    available_actions = actions(agent.world, agent.current_observation)
    
    # VOI-gated sensor queries
    # Maintain per-action belief P(action helps) — prior is 1/N (most actions don't help)
    n_actions = length(available_actions)
    @debug "act! start" state=s n_actions step=agent.step_count

    # Load cached beliefs or default to 1/N
    action_beliefs = Dict(
        a => get(agent.action_belief_cache, (s, a), 1.0 / n_actions)
        for a in available_actions
    )

    # Track queries for ground truth updates: (sensor, action, answer)
    sensor_queries = Tuple{Sensor, Any, Bool}[]

    # EU function: expected utility of an action given belief that it helps.
    # Actions believed helpful get a bonus proportional to belief.
    eu_func = (a, belief) -> begin
        rd = reward_dist(agent.model, s, a)
        mean_reward = Distributions.mean(rd)
        mean_reward + belief
    end

    for sensor in agent.sensors
        queries_made = 0
        while queries_made < agent.config.max_queries_per_step
            best_voi = 0.0
            best_action_to_ask = nothing

            for a in available_actions
                prior = action_beliefs[a]
                (prior < 0.01 || prior > 0.99) && continue

                voi = compute_voi(sensor, prior, available_actions,
                    (act, belief) -> eu_func(act, act == a ? belief : action_beliefs[act]))

                if voi > best_voi
                    best_voi = voi
                    best_action_to_ask = a
                end
            end

            (best_voi <= agent.config.sensor_cost || isnothing(best_action_to_ask)) && break

            question = "Will action \"$(best_action_to_ask)\" help make progress?"
            answer = query(sensor, agent.current_observation, question)

            action_beliefs[best_action_to_ask] = posterior(
                sensor, action_beliefs[best_action_to_ask], answer)

            push!(sensor_queries, (sensor, best_action_to_ask, answer))
            queries_made += 1
            @debug "sensor query" sensor=sensor.name action=best_action_to_ask answer voi=best_voi belief=action_beliefs[best_action_to_ask]
        end
    end

    # Persist updated beliefs back to cache
    for (a, belief) in action_beliefs
        agent.action_belief_cache[(s, a)] = belief
    end

    # Normalize beliefs as action priors for the planner
    belief_total = sum(values(action_beliefs))
    action_priors = if belief_total > 0
        Dict(a => b / belief_total for (a, b) in action_beliefs)
    else
        Dict(a => 1.0 / n_actions for a in available_actions)
    end

    action = plan_with_priors(agent.planner, s, agent.model, available_actions, action_priors)
    @debug "action selected" action beliefs=action_beliefs n_sensor_queries=length(sensor_queries)

    # Post-planning query: if the chosen action wasn't already queried, ask about it
    # so we get a sensor prediction to compare against ground truth
    already_queried = any(qa -> qa[2] == action, sensor_queries)
    if !already_queried && !isempty(agent.sensors)
        sensor = first(agent.sensors)
        question = "Will action \"$(action)\" help make progress?"
        answer = query(sensor, agent.current_observation, question)
        push!(sensor_queries, (sensor, action, answer))
        @debug "post-planning query" sensor=sensor.name action answer
    end

    # Execute action
    obs, reward, done, info = step!(agent.world, action)
    s′ = abstract_state(agent.abstractor, obs)
    @debug "step result" action reward new_state=s′ done

    # Update model
    update!(agent.model, s, action, reward, s′)
    
    # Record for abstraction learning
    record_transition!(agent.abstractor, s, action, reward, s′)
    
    # Check for contradictions and refine if needed
    contradiction = check_contradiction(agent.abstractor)
    if !isnothing(contradiction)
        refine!(agent.abstractor, contradiction)
    end
    
    # Update sensor reliability from ground truth (reward > 0 means action helped)
    actually_helped = reward > 0
    for (sensor, queried_action, said_yes) in sensor_queries
        if queried_action == action
            update_reliability!(sensor, said_yes, actually_helped)
        end
    end

    # Record transition
    push!(agent.trajectory, (s=s, a=action, r=reward, s′=s′))

    # Update state
    agent.current_observation = obs
    agent.current_abstract_state = s′
    agent.step_count += 1
    agent.total_reward += reward

    return action, obs, reward, done
end

"""
    run_episode!(agent::BayesianAgent; max_steps=1000) → total_reward

Run a complete episode and return the total reward.
"""
function run_episode!(agent::BayesianAgent; max_steps::Int = 1000)
    reset!(agent)
    
    for _ in 1:max_steps
        _, _, _, done = act!(agent)
        done && break
    end
    
    return agent.total_reward
end

# ============================================================================
# EXPORTS
# ============================================================================

export World, Sensor, WorldModel, Planner, StateAbstractor
export reset!, step!, actions, render, seed!
export query, tpr, fpr, update_reliability!, posterior
export sample_dynamics, update!, transition_dist, reward_dist, entropy
export abstract_state, record_transition!, check_contradiction, refine!
export plan
export compute_voi
export AgentConfig, BayesianAgent, act!, run_episode!

# Include implementations
include("models/tabular_world_model.jl")
include("models/binary_sensor.jl")
include("planners/thompson_mcts.jl")
include("abstractors/identity_abstractor.jl")
include("abstractors/bisimulation_abstractor.jl")

# World adapters
include("worlds/gridworld.jl")

# Jericho requires PyCall - load conditionally
try
    include("worlds/jericho.jl")
catch e
    @warn "Jericho world not available (PyCall or Jericho not installed)" exception=e
end

# Additional exports
export GridWorld, spawn_food!
export TabularWorldModel, sample_dynamics, information_gain
export ThompsonMCTS, MCTSNode, plan_with_priors
export IdentityAbstractor, BisimulationAbstractor
export BinarySensor, LLMSensor, format_observation_for_llm

end # module
