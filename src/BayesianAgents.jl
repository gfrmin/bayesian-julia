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

    # Stage 2: Variable Discovery
    enable_variable_discovery::Bool = false
    variable_discovery_frequency::Int = 10  # Every N steps
    variable_bic_threshold::Float64 = 0.0   # BIC improvement threshold

    # Stage 3: Structure Learning
    enable_structure_learning::Bool = false
    structure_learning_frequency::Int = 50  # Every N transitions
    max_parents::Int = 3

    # Stage 4: Action Schemas
    enable_action_schemas::Bool = false
    schema_discovery_frequency::Int = 100   # Every N transitions

    # Stage 5: Goal Planning
    enable_goal_planning::Bool = false
    goal_rollout_bias::Float64 = 0.5       # Blend: (1-bias)*random + bias*goal-directed
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

    # Pending sensor queries awaiting ground truth: (sensor, action, said_yes, step)
    pending_sensor_queries::Vector{Tuple{Sensor, Any, Bool, Int}}

    # Step of last nonzero reward (for windowed credit assignment)
    last_reward_step::Int

    # Observation history: (action, observation_text) for building LLM context
    observation_history::Vector{Tuple{Any, String}}

    # State analysis cache: observation_hash → StateAnalysis
    state_analysis_cache::Dict{UInt64, Any}
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
        Dict{Tuple{Any,Any}, Float64}(),
        Tuple{Sensor, Any, Bool, Int}[],
        0,
        Tuple{Any, String}[],
        Dict{UInt64, Any}()
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
    empty!(agent.pending_sensor_queries)
    agent.last_reward_step = 0
    empty!(agent.observation_history)
    empty!(agent.state_analysis_cache)
    return agent.current_observation
end

"""
    extract_observation_text(obs) → String

Extract a text representation from an observation for storing in history.
"""
function extract_observation_text(obs)
    if obs isa NamedTuple && hasproperty(obs, :text)
        return string(obs.text)
    else
        return string(obs)
    end
end

"""
    build_llm_context(agent::BayesianAgent) → String

Build rich context string for LLM selection queries. Includes:
- Current observation (location, text, inventory, score)
- Recent trajectory with outcomes (what happened when actions were tried)
- World model knowledge for current state (tried actions, observed rewards)
- Game progress (step count, score, states discovered)
"""
function build_llm_context(agent::BayesianAgent)
    parts = String[]
    s = agent.current_abstract_state

    # Current observation
    push!(parts, format_observation_for_llm(agent.current_observation))

    # Recent history with outcomes
    if !isempty(agent.observation_history)
        push!(parts, "")
        push!(parts, "Recent history:")
        n = min(15, length(agent.observation_history))
        for (action, outcome) in agent.observation_history[end-n+1:end]
            short = length(outcome) > 120 ? outcome[1:120] * "..." : outcome
            # Collapse whitespace for readability
            short = replace(short, r"\s+" => " ")
            push!(parts, "  > $action → $short")
        end
    end

    # Confirmed useless actions from this state (null outcomes)
    confirmed_useless = String[]
    for (state, action) in agent.model.confirmed_selfloops
        if state == s
            push!(confirmed_useless, "'$action'")
        end
    end
    if !isempty(confirmed_useless)
        push!(parts, "")
        push!(parts, "Confirmed useless from this location:")
        for a in confirmed_useless
            push!(parts, "  $a (no effect)")
        end
    end

    # Globally effective actions: (state, action) pairs with positive average reward anywhere
    globally_effective = String[]
    for (key, p) in agent.model.reward_posterior
        n_obs = Int(p.κ - agent.model.reward_prior.κ)
        if n_obs > 0 && p.μ > 0.1  # Saw positive rewards
            action_name = key[2]
            state_name = key[1]
            avg_r = round(p.μ, digits=1)
            push!(globally_effective, "'$action_name' at $state_name → avg reward $avg_r")
        end
    end
    if !isempty(globally_effective)
        push!(parts, "")
        push!(parts, "Actions that worked elsewhere:")
        for e in globally_effective[1:min(5, length(globally_effective))]
            push!(parts, "  $e")
        end
    end

    # World model knowledge: what actions have been tried from this state
    tried = String[]
    for (key, p) in agent.model.reward_posterior
        if key[1] == s
            n_obs = Int(p.κ - agent.model.reward_prior.κ)
            if n_obs > 0
                avg_r = round(p.μ, digits=2)
                push!(tried, "'$(key[2])' (tried $(n_obs)×, avg reward $avg_r)")
            end
        end
    end
    if !isempty(tried)
        push!(parts, "")
        push!(parts, "Previously tried from this location:")
        for t in tried
            push!(parts, "  $t")
        end
    end

    # Game progress
    push!(parts, "")
    # Count states: works for both TabularWorldModel and FactoredWorldModel
    n_states = if hasfield(typeof(agent.model), :known_states)
        length(agent.model.known_states)
    else
        length(agent.model.known_locations) + length(agent.model.known_objects)
    end
    push!(parts, "Step $(agent.step_count), total reward $(agent.total_reward), $n_states state elements discovered")

    return join(parts, "\n")
end

"""
    act!(agent::BayesianAgent) → (action, observation, reward, done)

Choose and execute an action via expected utility maximisation.
"""
function act!(agent::BayesianAgent)
    s = agent.current_abstract_state
    available_actions = actions(agent.world, agent.current_observation)

    # Filter out confirmed self-loops if there are alternatives
    viable_actions = filter(a -> !is_selfloop(agent.model, s, a), available_actions)
    if !isempty(viable_actions)
        available_actions = viable_actions
    end

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

    # Query state analysis from LLM if beliefs are uncertain
    for sensor in agent.sensors
        sensor isa LLMSensor || continue

        obs_hash = hash(extract_observation_text(agent.current_observation))
        if haskey(agent.state_analysis_cache, obs_hash)
            current_analysis = agent.state_analysis_cache[obs_hash]
            apply_state_analysis_priors!(current_analysis, available_actions, action_beliefs, sensor)
        else
            # VOI gate: only analyze if beliefs are uncertain
            max_uncertainty = maximum([min(b, 1-b) for b in values(action_beliefs)]; init=0.0)
            if max_uncertainty > 0.1
                context = build_llm_context(agent)
                current_analysis = query_state_analysis(sensor, context, available_actions)
                agent.state_analysis_cache[obs_hash] = current_analysis
                apply_state_analysis_priors!(current_analysis, available_actions, action_beliefs, sensor)
            end
        end
        break  # Only use state analysis from first LLM sensor
    end

    for sensor in agent.sensors
        if sensor isa LLMSensor
            # LLM sensor: single selection query ("which action is best?")
            # VOI check: use max single-action VOI as proxy for selection query value
            max_voi = 0.0
            for a in available_actions
                prior = action_beliefs[a]
                (prior < 0.01 || prior > 0.99) && continue
                voi = compute_voi(sensor, prior, available_actions,
                    (act, belief) -> eu_func(act, act == a ? belief : action_beliefs[act]))
                max_voi = max(max_voi, voi)
            end

            if max_voi > agent.config.sensor_cost
                context = build_llm_context(agent)
                selected = query_selection(sensor, context, available_actions)

                if !isnothing(selected)
                    update_beliefs_from_selection!(sensor, available_actions, selected, action_beliefs)
                    push!(sensor_queries, (sensor, selected, true))
                    @debug "LLM selection" sensor=sensor.name selected voi=max_voi
                end
            end
        else
            # Binary sensors (oracle, heuristic): VOI-gated yes/no queries
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
                @debug "binary query" sensor=sensor.name action=best_action_to_ask answer voi=best_voi belief=action_beliefs[best_action_to_ask]
            end
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

    # If the planner picks a different action than the LLM selected, store a
    # negative prediction for the executed action — the LLM implicitly said
    # "this is NOT the best action". This gives us learning signal even when
    # the planner overrides the LLM.
    llm_selected = isempty(sensor_queries) ? nothing : sensor_queries[end][2]
    if !isnothing(llm_selected) && action != llm_selected
        llm_sensor = sensor_queries[end][1]
        push!(sensor_queries, (llm_sensor, action, false))
        @debug "implicit negative" sensor=llm_sensor.name selected=llm_selected executed=action
    end

    # Execute action
    obs, reward, done, info = step!(agent.world, action)
    s′ = abstract_state(agent.abstractor, obs)
    @debug "step result" action reward new_state=s′ done

    # Store observation text for LLM context history
    push!(agent.observation_history, (action, extract_observation_text(obs)))

    # Update model
    update!(agent.model, s, action, reward, s′)

    # Record for abstraction learning
    record_transition!(agent.abstractor, s, action, reward, s′)

    # ================================================================
    # STAGES 2-5: ADVANCED LEARNING (Optional, config-gated)
    # ================================================================
    # Note: These features are available and tested, but require different
    # state representations in some cases. For now, we demonstrate Stage 5
    # (goal planning) which works directly with MinimalState.

    # Stage 3: Structure Learning (action scope identification)
    if agent.config.enable_structure_learning && agent.step_count % agent.config.structure_learning_frequency == 0
        try
            if isa(agent.model, FactoredWorldModel)
                scope = compute_action_scope(agent.model, action)
                if !isempty(scope)
                    @debug "Stage 3: Action Scope" action scope=scope
                end
            end
        catch e
            @debug "Stage 3 error" exception=e
        end
    end

    # Stage 4: Action Schemas (clustering similar actions)
    if agent.config.enable_action_schemas && agent.step_count % agent.config.schema_discovery_frequency == 0
        try
            if isa(agent.model, FactoredWorldModel)
                schemas = discover_schemas(agent.model)
                if !isempty(schemas)
                    @debug "Stage 4: Action Schemas discovered" num_schemas=length(schemas)
                end
            end
        catch e
            @debug "Stage 4 error" exception=e
        end
    end

    # Stage 5: Goal-Directed Planning (extract and track goals from observation text)
    if agent.config.enable_goal_planning
        try
            obs_text = extract_observation_text(obs)
            goals = extract_goals_from_text(obs_text)
            if !isempty(goals)
                # Update goal achievement status based on current state
                if isa(s′, MinimalState)
                    update_goal_status!(goals, s′)
                    achieved_count = count(g -> g.achieved for g in goals)
                    @debug "Stage 5: Goal Planning" num_goals=length(goals) achieved=achieved_count
                end
            end
        catch e
            @debug "Stage 5 error" exception=e
        end
    end

    # ================================================================
    # END STAGES 2-5
    # ================================================================

    # Check for contradictions and refine if needed
    contradiction = check_contradiction(agent.abstractor)
    if !isnothing(contradiction)
        refine!(agent.abstractor, contradiction)
    end
    
    # Store sensor queries for delayed credit assignment
    for (sensor, queried_action, said_yes) in sensor_queries
        push!(agent.pending_sensor_queries, (sensor, queried_action, said_yes, agent.step_count))
    end

    # Null-action ground truth: if observation text unchanged, action was unhelpful
    if is_null_outcome(agent.current_observation, obs)
        resolve_null_action_queries!(agent, action)
        mark_selfloop!(agent.model, s, action)
        agent.action_belief_cache[(s, action)] = 0.001
    end

    # When reward != 0, resolve pending queries with discounted trajectory credit
    if reward != 0.0
        resolve_pending_queries!(agent, reward)
        agent.last_reward_step = agent.step_count
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
    resolve_pending_queries!(agent::BayesianAgent, reward::Float64)

Resolve pending sensor queries using discounted trajectory credit.

Actions close to the reward event get strong credit (γ^1), distant actions get
weak credit (γ^20). The proposition was "action helps make progress" — temporal
proximity determines how much credit the action receives. Very distant actions
(|discounted| < 0.001) are skipped entirely to avoid noise.
"""
function resolve_pending_queries!(agent::BayesianAgent, reward::Float64)
    γ = agent.config.discount
    resolved = Int[]
    for (i, (sensor, action, said_yes, step)) in enumerate(agent.pending_sensor_queries)
        if step > agent.last_reward_step
            steps_elapsed = agent.step_count - step
            discounted = reward * γ^steps_elapsed
            # Skip if credit is negligible (avoids noise from very distant actions)
            if abs(discounted) < 0.001
                continue
            end
            actually_helped = discounted > 0.0
            update_reliability!(sensor, said_yes, actually_helped)
            push!(resolved, i)
            @debug "trajectory credit" action said_yes actually_helped discount=γ^steps_elapsed
        end
    end
    deleteat!(agent.pending_sensor_queries, sort(resolved))
end

"""
    resolve_null_action_queries!(agent::BayesianAgent, null_action)

Resolve pending sensor queries about an action that produced no change.

When observation text is identical before and after, the action was definitively
unhelpful: P(obs_unchanged | action_helpful) ≈ 0. This provides negative ground
truth to calibrate the sensor's FPR without waiting for sparse rewards.
"""
function resolve_null_action_queries!(agent::BayesianAgent, null_action)
    resolved = Int[]
    for (i, (sensor, queried_action, said_yes, step)) in enumerate(agent.pending_sensor_queries)
        if queried_action == null_action
            update_reliability!(sensor, said_yes, false)
            push!(resolved, i)
            @debug "null action ground truth" action=null_action said_yes
        end
    end
    deleteat!(agent.pending_sensor_queries, sort(resolved))
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
export AgentConfig, BayesianAgent, act!, run_episode!, resolve_pending_queries!, resolve_null_action_queries!

# ============================================================================
# FOUNDATIONAL COMPONENTS (needed by both legacy and Stage 1)
# ============================================================================

# Legacy components (depended on by Stage 1)
include("models/tabular_world_model.jl")
include("models/binary_sensor.jl")
include("planners/thompson_mcts.jl")
include("abstractors/identity_abstractor.jl")
include("abstractors/bisimulation_abstractor.jl")
include("abstractors/minimal_state_abstractor.jl")

# ============================================================================
# STAGE 1: MVBN (Minimum Viable Bayesian Network) - NEW COMPONENTS
# ============================================================================

# Probability foundations
include("probability/cpd.jl")

# State representation
include("state/minimal_state.jl")
include("state/state_belief.jl")
include("state/variable_discovery.jl")  # Stage 2

# Factored world model
include("models/factored_world_model.jl")

# State inference
include("inference/bayesian_update.jl")
include("inference/hidden_variable_inference.jl")

# Factored planning
include("planning/factored_mcts.jl")

# Stage 3: Structure Learning
include("structure/structure_learning.jl")

# Stage 4: Action Schemas
include("actions/action_schema.jl")

# Stage 5: Goal-Directed Planning
include("planning/goal_planning.jl")

# World adapters
include("worlds/gridworld.jl")

# Jericho requires PyCall - load conditionally
try
    include("worlds/jericho.jl")
catch e
    @warn "Jericho world not available (PyCall or Jericho not installed)" exception=e
end

# Stage 1: MVBN exports
export DirichletCategorical, update!, predict, entropy, expected_entropy, mode
export MinimalState, extract_minimal_state
export StateBelief, add_object!, update_from_state!, sample_state, predict_state, posterior_prob, loglikelihood
export FactoredWorldModel, SampledFactoredDynamics, add_location!, sample_next_state, mark_selfloop!, is_selfloop
export FactoredMCTS, FactoredMCTSNode, mcts_search
export update_location_belief!, update_inventory_belief!, bayesian_update_belief!, predict_from_likelihood

# Stage 2: Variable Discovery exports
export VariableCandidate, extract_candidate_variables, compute_bic, compute_bic_delta
export should_accept_variable, discover_variables!, update_state_belief_with_discovery!

# Stage 3: Structure Learning exports
export DirectedGraph, add_edge!, remove_edge!, get_parents, is_acyclic
export bde_score, learn_structure_greedy, LearnedStructure
export learn_action_structure, compute_action_scope

# Stage 4: Action Schemas exports
export ActionSchema, ActionInstance
export extract_action_type, cluster_actions, infer_schema_from_cluster
export discover_schemas, apply_schema, zero_shot_transfer_likelihood

# Stage 5: Goal-Directed Planning exports
export Goal, extract_goals_from_text, compute_goal_progress, expected_goal_progress
export goal_biased_action_selection, update_goal_status!
export intrinsic_motivation_reward
export ValueOfInformation, compute_voi_for_query

# Legacy exports
export GridWorld, spawn_food!
export TabularWorldModel, NormalGammaPosterior, SampledDynamics, get_reward, information_gain
export ThompsonMCTS, MCTSNode, plan_with_priors, select_rollout_action
export IdentityAbstractor, BisimulationAbstractor, MinimalStateAbstractor, abstraction_summary
export BinarySensor, LLMSensor, format_observation_for_llm, query_selection, update_beliefs_from_selection!, is_null_outcome, StateAnalysis, query_state_analysis, parse_state_analysis, apply_state_analysis_priors!
export extract_observation_text, build_llm_context
export action_features, collect_posteriors, combine_posteriors

end # module
