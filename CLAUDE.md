# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

Julia ≥ 1.9 required. Project.toml is at the repo root.

    # Install/resolve dependencies
    julia --project=. -e 'using Pkg; Pkg.instantiate()'

    # Load the package in REPL
    julia --project=.
    julia> using BayesianAgents

    # Run examples
    julia --project=. examples/gridworld_example.jl
    julia --project=. examples/gridworld_agent.jl

    # Run tests (once test/ exists)
    julia --project=. -e 'using Pkg; Pkg.test()'

## Playing Interactive Fiction Games

The agent can play text adventure games (IF) via Jericho. Requires: `pip install jericho`

    # Play Enchanter game without LLM
    julia --project=. examples/jericho_agent.jl /path/to/enchanter.z3 --episodes 5 --steps 100

    # Play with LLM guidance (requires Ollama running: ollama serve)
    # First pull a model: ollama pull llama3.1
    julia --project=. examples/jericho_agent.jl /path/to/enchanter.z3 --llm --model llama3.1 --episodes 5 --steps 100

    # Common options:
    --llm                 # Enable LLM sensor for state analysis and action selection
    --model llama3.1      # Specify LLM model (requires ollama to have it installed)
    --episodes 5          # Number of episodes to run (default 3)
    --steps 100           # Max steps per episode (default 50)
    --verbose             # Print detailed debug output
    --mcts-iterations 100 # MCTS samples per decision (default 60)
    --mcts-depth 10       # Planning horizon (default 8)

Game files location: `/home/g/Sync/git/bayesian-agents/bayesian-if-agent/games/`
Available games: enchanter.z3, (others in games/ directory)

How it works:
1. Agent abstracts game state to location|inventory_hash to learn world model
2. Thompson Sampling MCTS plans 8-20 steps ahead
3. LLM sensor (if enabled) analyzes game text, identifies obstacles and promising actions
4. VOI-gated queries: only asks LLM when action beliefs are uncertain
5. Null outcome detection: marks repeated "look X" commands as confirmed no-ops to avoid looping

## Architecture

Core module: `src/BayesianAgents.jl` — defines 5 abstract interfaces and the agent loop (`act!`, `run_episode!`).

| Interface | Methods | Implementations |
|---|---|---|
| `World` | `reset!`, `step!`, `actions` | `GridWorld` (`src/worlds/gridworld.jl`), `JerichoWorld` (`src/worlds/jericho.jl`) |
| `WorldModel` | `update!`, `sample_dynamics`, `transition_dist`, `reward_dist` | `TabularWorldModel` (`src/models/tabular_world_model.jl`) — Dirichlet-Categorical transitions, Normal-Gamma rewards |
| `Planner` | `plan` | `ThompsonMCTS` (`src/planners/thompson_mcts.jl`) — MCTS with posterior-sampled dynamics |
| `StateAbstractor` | `abstract_state`, `record_transition!`, `check_contradiction`, `refine!` | `IdentityAbstractor`, `BisimulationAbstractor` (`src/abstractors/`) |
| `Sensor` | `query`, `tpr`, `fpr`, `update_reliability!` | `BinarySensor`, `LLMSensor` (`src/models/binary_sensor.jl`) — Beta posteriors for TPR/FPR |

**Data flow**: World → StateAbstractor → WorldModel → Planner → Decision. Sensors provide VOI-gated observations via `compute_voi()`. Agent config is `AgentConfig` struct.

**Dependencies**: Distributions.jl, StatsBase.jl, DataStructures.jl, LinearAlgebra (stdlib), PyCall.jl (optional, for Jericho IF games).

**Specs**: `MASTER_SPEC.md` has the complete mathematical framework. `UNIFIED_SPEC.md` has additional theory. `ARCHITECTURE.md` has system design.

---

## Purpose

Build a general Bayesian agent framework in Julia that:

1. Maintains **hierarchical uncertainty** over state, dynamics, structure, and meta-parameters
2. Plans over **trajectories**, not just single actions (critical for sparse-reward domains)
3. Learns **state equivalence classes** via bisimulation to avoid loops
4. Treats **asking questions** and **taking actions** in a **unified decision space**
5. Supports **meta-learning**: learning priors across tasks
6. Connects to arbitrary worlds (IF games, grid worlds, real data)

**All behaviour derives from expected utility maximisation. No hacks.**

## Critical Lessons Learned

These insights came from debugging a failed IF agent:

| Problem | Wrong Approach | Correct Approach |
|---------|---------------|------------------|
| Agent loops on equivalent states | Add loop detection | Fix state abstraction (bisimulation) |
| Agent never asks questions | Separate "should ask?" logic | Unified EU comparison |
| Agent always asks questions | Heuristic thresholds | VOI > cost (principled) |
| Myopic planning fails | One-step lookahead | Trajectory-level planning |
| Ground truth for sensor learning | State changed | Reward > 0 |
| Prior P(action helps) | 0.5 (unrealistic) | 1/N (most actions don't help) |

---

## NON-NEGOTIABLE DESIGN PRINCIPLES

### Principle 1: Unified Decision Space

The agent chooses between MULTIPLE types of options at each step:

```
Options = {take(a) for a in game_actions} 
        ∪ {ask(q,k) for q in questions, k in sensors}
        ∪ {plan_more(n)}
        ∪ {refine_abstraction()}
```

ALL are evaluated by expected utility. The agent picks `argmax EU(option)`.

**This is the core insight. Do not separate "should I ask?" from "what should I do?"**

### Principle 2: Trajectory-Level Planning

One-step lookahead is insufficient. IF games require ~20 steps to solve puzzles.

The agent plans over **trajectories** (sequences of state-action pairs):

```
τ = (s₀, a₀, s₁, a₁, ..., sₜ)
V(τ) = Σ γᵗ r(sₜ, aₜ)
```

**Value of Information must be trajectory-level**, not myopic:

```
TVOI(question) = E[max_τ V(τ) | after_answer] - max_τ V(τ) | now
```

### Principle 3: Hierarchical Uncertainty

Uncertainty exists at multiple levels:

```
Level 0: State          — Where am I? P(s | observations)
Level 1: Dynamics       — How does the world work? P(θ | history)  
Level 2: Structure      — What state representation? P(φ | contradictions)
Level 3: Meta           — What priors work? P(hyperparameters | tasks)
```

### Principle 4: No Exploration Bonuses

Exploration emerges naturally from uncertainty. An untried action has high variance → Thompson Sampling will sometimes pick it. **Never add ε-greedy, UCB bonuses, or exploration terms.**

### Principle 5: No Loop Detection

If the agent loops, the model is wrong. Fix the model (improve state abstraction via bisimulation), don't add loop detection hacks.

### Principle 6: LLM is a Sensor, Not a Decision-Maker

The LLM provides **observations** (with learned reliability). The Bayesian machinery makes **decisions**. Never let the LLM directly choose actions.

### Principle 7: Meta-Learning Across Tasks

Priors should improve across tasks. After playing many IF games:
- Prior on action success rate sharpens
- Prior on LLM reliability concentrates
- Prior on dynamics sparsity calibrates

---

## MATHEMATICAL FRAMEWORK

### The Hierarchy of Models

```julia
struct AgentBeliefs
    # Level 0: Current state belief
    state_belief::Distribution  # P(s | observations)
    
    # Level 1: World dynamics
    dynamics_model::WorldModel  # P(θ | history) over transition/reward params
    
    # Level 2: State abstraction
    abstractor::StateAbstractor  # Mapping φ: concrete → abstract states
    
    # Level 3: Meta-parameters
    meta::MetaPriors  # Hyperparameters learned across tasks
end
```

### Trajectory Representation

```julia
struct Trajectory
    states::Vector{Any}
    actions::Vector{Any}
    rewards::Vector{Float64}
    
    # Computed
    total_value::Float64  # Σ γᵗ rₜ
    first_action::Any
end
```

### Expected Utility of Game Actions (Trajectory-Level)

For action `a` in state `s`, we compute EU over **trajectories starting with `a`**:

```
EU(take(a) | s) = E_{θ ~ P(θ|H)} [ max_{τ: τ₀=(s,a)} V(τ | θ) ]
```

With Thompson Sampling:
```
1. Sample θ̂ ~ P(θ | history)
2. Plan optimal trajectory τ* under θ̂ starting with action a
3. EU(take(a)) ≈ V(τ*)
```

### Expected Utility of Asking Questions (Trajectory-Level)

For question `q` to sensor `k`:

```
EU(ask(q,k) | s) = E_{answer}[max_τ V(τ) | answer] - cost_k
```

This is **trajectory-level VOI** minus query cost:

```
TVOI(q,k) = E_{answer}[max_τ V(τ) | answer] - max_τ V(τ) | current_beliefs

EU(ask(q,k)) = max_τ V(τ) | current + TVOI(q,k) - cost_k
```

### Computing Trajectory VOI

```julia
function trajectory_voi(state, question, sensor, model, horizon)
    # Current best trajectory
    θ_now = sample_dynamics(model)
    τ_now = mcts_plan(state, θ_now, horizon)
    V_now = trajectory_value(τ_now)
    
    # P(sensor says yes)
    prior = get_proposition_prior(question, state, model)
    p_yes = tpr(sensor) * prior + fpr(sensor) * (1 - prior)
    
    # Value if yes
    model_yes = copy_and_update_beliefs(model, question, true)
    θ_yes = sample_dynamics(model_yes)
    τ_yes = mcts_plan(state, θ_yes, horizon)
    V_yes = trajectory_value(τ_yes)
    
    # Value if no  
    model_no = copy_and_update_beliefs(model, question, false)
    θ_no = sample_dynamics(model_no)
    τ_no = mcts_plan(state, θ_no, horizon)
    V_no = trajectory_value(τ_no)
    
    # Expected value after asking
    V_after = p_yes * V_yes + (1 - p_yes) * V_no
    
    return V_after - V_now
end
```

### The Unified Decision

```julia
function unified_decide(agent)
    state = agent.current_state
    model = agent.model
    sensors = agent.sensors
    config = agent.config
    
    # 1. Plan trajectories via Thompson Sampling
    trajectories = []
    for _ in 1:config.n_samples
        θ = sample_dynamics(model)
        τ = mcts_plan(state, θ, config.horizon)
        push!(trajectories, τ)
    end
    
    best_trajectory = trajectories[argmax(τ -> τ.total_value, trajectories)]
    best_act_eu = best_trajectory.total_value
    
    # 2. Compute trajectory-VOI for questions
    best_ask = nothing
    best_ask_eu = -Inf
    
    for sensor in sensors
        for question in generate_questions(state, best_trajectory, model)
            tvoi = trajectory_voi(state, question, sensor, model, config.horizon)
            
            if tvoi > sensor.cost
                eu = best_act_eu + tvoi - sensor.cost
                if eu > best_ask_eu
                    best_ask_eu = eu
                    best_ask = (question, sensor)
                end
            end
        end
    end
    
    # 3. Consider more planning (diminishing returns)
    planning_voi = estimate_planning_improvement(trajectories)
    plan_eu = best_act_eu + planning_voi - config.compute_cost
    
    # 4. Consider abstraction refinement
    contradiction = check_contradiction(agent.abstractor)
    refine_eu = if !isnothing(contradiction)
        refinement_voi = estimate_refinement_value(contradiction, model)
        best_act_eu + refinement_voi - config.refine_cost
    else
        -Inf
    end
    
    # 5. Unified decision: pick best option
    options = [
        (:act, best_trajectory.first_action, best_act_eu),
        (:ask, best_ask, best_ask_eu),
        (:plan, config.extra_iterations, plan_eu),
        (:refine, contradiction, refine_eu)
    ]
    
    best = options[argmax(o -> o[3], options)]
    
    return Decision(best[1], best[2])
end
```

---

## IMPLEMENTATION SPECIFICATION

### Core Types

```julia
# Abstract interfaces
abstract type World end
abstract type WorldModel end
abstract type Sensor end
abstract type Planner end
abstract type StateAbstractor end

# Decision types
struct ActDecision
    action::Any
end

struct AskDecision
    question::String
    sensor::Sensor
end

const Decision = Union{ActDecision, AskDecision}
```

### World Interface

```julia
# Every world must implement:
reset!(world::World) → observation
step!(world::World, action) → (observation, reward, done, info)
actions(world::World, observation) → Vector{Action}
```

### WorldModel Interface

```julia
# Bayesian dynamics model must implement:
update!(model::WorldModel, s, a, r, s′)           # Posterior update
sample_dynamics(model::WorldModel) → SampledModel  # For Thompson Sampling
transition_dist(model::WorldModel, s, a) → Distribution
reward_dist(model::WorldModel, s, a) → Distribution
entropy(model::WorldModel) → Float64               # Uncertainty measure
```

### Sensor Interface

```julia
# Every sensor must implement:
query(sensor::Sensor, state, question) → response
tpr(sensor::Sensor) → Float64                      # True positive rate
fpr(sensor::Sensor) → Float64                      # False positive rate
update_reliability!(sensor::Sensor, predicted, actual)  # Learn from ground truth

# Provided by framework:
posterior(sensor::Sensor, prior, response) → Float64  # Bayes update
```

### Binary Sensor Implementation

```julia
mutable struct BinarySensor <: Sensor
    name::String
    
    # TPR ~ Beta(tp_α, tp_β)
    tp_α::Float64
    tp_β::Float64
    
    # FPR ~ Beta(fp_α, fp_β)  
    fp_α::Float64
    fp_β::Float64
    
    # Query function
    query_fn::Function
    
    # Cost of querying this sensor
    cost::Float64
end

tpr(s::BinarySensor) = s.tp_α / (s.tp_α + s.tp_β)
fpr(s::BinarySensor) = s.fp_α / (s.fp_α + s.fp_β)

function posterior(sensor::BinarySensor, prior::Float64, said_yes::Bool)
    t, f = tpr(sensor), fpr(sensor)
    if said_yes
        return t * prior / (t * prior + f * (1 - prior))
    else
        return (1-t) * prior / ((1-t) * prior + (1-f) * (1 - prior))
    end
end

function update_reliability!(sensor::BinarySensor, said_yes::Bool, actually_true::Bool)
    if actually_true
        said_yes ? (sensor.tp_α += 1) : (sensor.tp_β += 1)
    else
        said_yes ? (sensor.fp_α += 1) : (sensor.fp_β += 1)
    end
end
```

### VOI Calculation

```julia
function compute_voi(
    state,
    question::String,
    sensor::BinarySensor,
    action_eus::Dict,
    model::WorldModel;
    proposition_prior::Float64 = 0.5
)
    t, f = tpr(sensor), fpr(sensor)
    
    # Current best
    current_best_eu = maximum(values(action_eus))
    
    # P(sensor says yes)
    p_yes = t * proposition_prior + f * (1 - proposition_prior)
    p_no = 1 - p_yes
    
    # Posteriors
    post_if_yes = posterior(sensor, proposition_prior, true)
    post_if_no = posterior(sensor, proposition_prior, false)
    
    # Recompute action EUs under each posterior
    # (This depends on what the question is about)
    # For "does action A help?", update belief about A's effectiveness
    
    eus_if_yes = update_action_beliefs(action_eus, question, post_if_yes)
    eus_if_no = update_action_beliefs(action_eus, question, post_if_no)
    
    best_if_yes = maximum(values(eus_if_yes))
    best_if_no = maximum(values(eus_if_no))
    
    # Expected best after asking
    expected_best_after = p_yes * best_if_yes + p_no * best_if_no
    
    # VOI = improvement
    return expected_best_after - current_best_eu
end
```

### Unified Agent Loop

```julia
function agent_step!(agent::BayesianAgent)
    state = agent.current_state
    game_actions = actions(agent.world, agent.observation)
    
    # Generate candidate questions (about actions, goals, state)
    questions = generate_questions(state, game_actions, agent.config)
    
    # Make unified decision
    decision = decide(
        state, 
        game_actions, 
        questions, 
        agent.sensors,
        agent.model,
        agent.config
    )
    
    if decision isa AskDecision
        # Query sensor
        response = query(decision.sensor, state, decision.question)
        
        # Update beliefs based on answer
        update_beliefs_from_answer!(agent, decision.question, response)
        
        # Recurse (might ask more questions or finally act)
        return agent_step!(agent)
        
    else  # ActDecision
        # Execute game action
        obs, reward, done, info = step!(agent.world, decision.action)
        
        # Get new abstract state
        new_state = abstract_state(agent.abstractor, obs)
        
        # Update world model
        update!(agent.model, state, decision.action, reward, new_state)
        
        # Update sensor reliability if we can determine ground truth
        update_sensor_ground_truth!(agent, state, decision.action, reward)
        
        # Update agent state
        agent.observation = obs
        agent.current_state = new_state
        
        return obs, reward, done
    end
end
```

---

## QUESTION GENERATION FOR IF GAMES

For Interactive Fiction, generate questions about:

### Action-Focused Questions
```julia
function action_questions(state, actions)
    return [
        "Will '$action' help make progress toward winning?" 
        for action in actions
    ]
end
```

### Goal-Focused Questions
```julia
function goal_questions(state)
    return [
        "What is the immediate goal in this situation?",
        "Is there something blocking progress?",
        "Have I missed examining something important?"
    ]
end
```

### State-Focused Questions  
```julia
function state_questions(state, actions)
    return [
        "Are there any items I should pick up here?",
        "Is this location dangerous?",
        "Have I been here before?"
    ]
end
```

### Question Selection

Don't ask all questions—use VOI to select:

```julia
function select_questions(state, actions, sensors, budget)
    all_questions = vcat(
        action_questions(state, actions),
        goal_questions(state),
        state_questions(state, actions)
    )
    
    # Score by estimated VOI (can use heuristics for speed)
    scored = [(q, estimate_voi(q, state, sensors)) for q in all_questions]
    sort!(scored, by=x->x[2], rev=true)
    
    # Return top questions within budget
    return [q for (q, _) in scored[1:min(budget, length(scored))]]
end
```

---

## LLM SENSOR IMPLEMENTATION

```julia
struct LLMSensor <: Sensor
    name::String
    client::OllamaClient  # Or any LLM client
    
    # Reliability parameters (learned)
    tp_α::Float64
    tp_β::Float64
    fp_α::Float64
    fp_β::Float64
    
    # Configuration
    model_name::String
    temperature::Float64
    cost::Float64  # Cost per query (in utility units)
    
    # Prompt template
    system_prompt::String
end

function query(sensor::LLMSensor, state, question::String)
    prompt = """
    $(sensor.system_prompt)
    
    Current situation:
    $(format_state_for_llm(state))
    
    Question: $question
    
    Answer with only 'yes' or 'no'.
    """
    
    response = generate(sensor.client, sensor.model_name, prompt;
                       temperature=sensor.temperature)
    
    return parse_yes_no(response)
end

function format_state_for_llm(state)
    """
    Location: $(state.location)
    Observation: $(state.text)
    Inventory: $(state.inventory)
    Score: $(state.score)
    Recent actions: $(join(state.recent_actions, ", "))
    """
end

function parse_yes_no(response::String)
    r = lowercase(strip(response))
    if startswith(r, "yes")
        return true
    elseif startswith(r, "no")
        return false
    else
        # Ambiguous - try harder
        return occursin("yes", r) && !occursin("no", r)
    end
end
```

---

## GROUND TRUTH FOR SENSOR LEARNING

The agent learns sensor reliability by comparing predictions to outcomes.

**When do we get ground truth?**

1. **Action helped**: `reward > 0` after taking the action
2. **Action didn't help**: `reward ≤ 0` AND no state progress

```julia
function update_sensor_ground_truth!(agent, state, action, reward)
    # Find any questions we asked about this action
    for (question, sensor, answer) in agent.recent_queries
        if question_is_about_action(question, action)
            # Determine ground truth
            actually_helped = reward > 0
            
            # Update sensor reliability
            update_reliability!(sensor, answer, actually_helped)
        end
    end
end
```

**Important**: Only use `reward > 0` as ground truth, not "state changed". Many state changes are neutral or negative.

---

## STATE ABSTRACTION (Level 2)

**The Problem**: Raw state representations cause loops. "Wearing trousers" and "not wearing trousers" have different hashes but may be strategically identical.

**The Solution**: Learn equivalence classes via **bisimulation**.

### Bisimulation Definition

Two states $s_1, s_2$ are **bisimilar** iff:

$$\forall a \in \mathcal{A}: \quad P(r \mid s_1, a) = P(r \mid s_2, a) \quad \land \quad P(\phi(s') \mid s_1, a) = P(\phi(s') \mid s_2, a)$$

In plain English: same rewards and same transition distributions (to equivalence classes) for all actions.

### Behavioural Signature

Each state has a **signature** = its observed (action → outcome) map:

```julia
struct BehaviouralSignature
    # action → (mean_reward, var_reward, next_state_counts)
    outcomes::Dict{Any, Tuple{Float64, Float64, Dict{Int, Int}}}
end
```

### Implementation

```julia
mutable struct BisimulationAbstractor <: StateAbstractor
    # concrete observation → abstract state ID
    obs_to_abstract::Dict{Any, Int}
    
    # abstract ID → set of concrete observations
    abstract_to_obs::Dict{Int, Set{Any}}
    
    # concrete observation → behavioural signature
    signatures::Dict{Any, BehaviouralSignature}
    
    # For generating new abstract IDs
    next_id::Int
    
    # Similarity threshold for merging
    threshold::Float64
end

function abstract_state(abs::BisimulationAbstractor, obs)
    # If we've seen this observation, return its abstract state
    if haskey(abs.obs_to_abstract, obs)
        return abs.obs_to_abstract[obs]
    end
    
    # Check if signature matches any existing class
    if haskey(abs.signatures, obs)
        for (id, obs_set) in abs.abstract_to_obs
            rep = first(obs_set)
            if haskey(abs.signatures, rep)
                if signatures_match(abs.signatures[obs], abs.signatures[rep], abs.threshold)
                    # Merge into existing class
                    abs.obs_to_abstract[obs] = id
                    push!(obs_set, obs)
                    return id
                end
            end
        end
    end
    
    # Create new equivalence class
    id = abs.next_id
    abs.next_id += 1
    abs.obs_to_abstract[obs] = id
    abs.abstract_to_obs[id] = Set([obs])
    return id
end

function record_transition!(abs::BisimulationAbstractor, s_obs, a, r, s_next_obs)
    # Update signature for s_obs
    if !haskey(abs.signatures, s_obs)
        abs.signatures[s_obs] = BehaviouralSignature(Dict())
    end
    
    sig = abs.signatures[s_obs]
    s_next_abstract = abstract_state(abs, s_next_obs)
    
    # Update outcome statistics for this action
    update_signature!(sig, a, r, s_next_abstract)
end

function check_contradiction(abs::BisimulationAbstractor)
    # Check each equivalence class for internal contradictions
    for (id, obs_set) in abs.abstract_to_obs
        if length(obs_set) > 1
            obs_list = collect(obs_set)
            for i in 1:length(obs_list)-1
                for j in i+1:length(obs_list)
                    o1, o2 = obs_list[i], obs_list[j]
                    if haskey(abs.signatures, o1) && haskey(abs.signatures, o2)
                        conflict = find_conflict(abs.signatures[o1], abs.signatures[o2])
                        if !isnothing(conflict)
                            return (abstract_id=id, obs1=o1, obs2=o2, conflict=conflict)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function refine!(abs::BisimulationAbstractor, contradiction)
    # Split the equivalence class
    old_id = contradiction.abstract_id
    obs_to_move = contradiction.obs2
    
    # Create new class for the conflicting observation
    new_id = abs.next_id
    abs.next_id += 1
    
    abs.obs_to_abstract[obs_to_move] = new_id
    delete!(abs.abstract_to_obs[old_id], obs_to_move)
    abs.abstract_to_obs[new_id] = Set([obs_to_move])
    
    # Re-check other observations in old class
    reclassify!(abs, old_id, new_id)
end
```

This solves the "trousers loop" problem: states that differ only in irrelevant details will have the same signature and be merged.

---

## META-LEARNING (Level 3)

**The Problem**: What priors should we use for a new game/world?

**The Solution**: Learn priors from experience across tasks via **hierarchical Bayesian modelling**.

### What We Meta-Learn

1. **Dynamics sparsity** $\alpha_0$: How concentrated are transition distributions?
2. **Sensor reliability priors**: What TPR/FPR do sensors typically have?
3. **Action success rate**: What fraction of actions typically help?
4. **Abstraction granularity**: How aggressive should state merging be?

### Hierarchical Model for Sensor Reliability

```
Per-task model:
    TPR_task ~ Beta(α_tp, β_tp)
    FPR_task ~ Beta(α_fp, β_fp)

Meta-level:
    (α_tp, β_tp) ~ some prior, updated across tasks
```

### Implementation

```julia
mutable struct MetaPriors
    # Dynamics concentration
    α_dynamics::Float64      # Gamma posterior: shape
    β_dynamics::Float64      # Gamma posterior: rate
    
    # Sensor TPR prior params
    sensor_tpr_α::Float64
    sensor_tpr_β::Float64
    
    # Sensor FPR prior params  
    sensor_fpr_α::Float64
    sensor_fpr_β::Float64
    
    # Action success rate
    action_success_α::Float64  # Beta posterior
    action_success_β::Float64
    
    # Statistics for updates
    task_count::Int
    tpr_observations::Vector{Float64}
    fpr_observations::Vector{Float64}
end

function update_meta!(meta::MetaPriors, task_result::TaskSummary)
    meta.task_count += 1
    
    # Update sensor priors from observed final TPR/FPR
    for sensor in task_result.sensors
        push!(meta.tpr_observations, tpr(sensor))
        push!(meta.fpr_observations, fpr(sensor))
    end
    
    # Empirical Bayes: fit Beta to observed TPRs
    if length(meta.tpr_observations) >= 5
        meta.sensor_tpr_α, meta.sensor_tpr_β = fit_beta(meta.tpr_observations)
        meta.sensor_fpr_α, meta.sensor_fpr_β = fit_beta(meta.fpr_observations)
    end
    
    # Update action success prior
    success_rate = task_result.helpful_actions / task_result.total_actions
    meta.action_success_α += success_rate * 10  # Pseudo-count scaling
    meta.action_success_β += (1 - success_rate) * 10
end

function get_sensor_priors(meta::MetaPriors)
    return (
        tpr = (meta.sensor_tpr_α, meta.sensor_tpr_β),
        fpr = (meta.sensor_fpr_α, meta.sensor_fpr_β)
    )
end

function get_action_success_prior(meta::MetaPriors)
    return meta.action_success_α / (meta.action_success_α + meta.action_success_β)
end
```

### When to Use Meta-Priors

```julia
function create_agent_for_new_task(meta::MetaPriors, world::World)
    # Use meta-learned priors instead of defaults
    priors = get_sensor_priors(meta)
    
    sensor = LLMSensor(
        "llm",
        tp_prior = priors.tpr,
        fp_prior = priors.fpr,
        ...
    )
    
    model = TabularWorldModel(
        transition_prior = sample_concentration(meta),
        action_success_prior = get_action_success_prior(meta),
        ...
    )
    
    return BayesianAgent(world, model, sensors=[sensor], ...)
end
```

### Benefits of Meta-Learning

1. **Faster learning on new tasks**: Good priors → fewer samples needed
2. **Better calibrated uncertainty**: Priors match actual distributions
3. **Transfer across domains**: If IF games have similar structure, exploit it

---

## FORBIDDEN PATTERNS

### ❌ DO NOT: Add exploration bonuses
```julia
# WRONG
eu = belief * reward + exploration_bonus  
```
```julia
# RIGHT  
eu = belief * reward  # Exploration emerges from Thompson Sampling
```

### ❌ DO NOT: Separate asking from acting
```julia
# WRONG
if should_ask():
    ask_question()
else:
    take_action()
```
```julia
# RIGHT
decision = argmax([EU(ask(q)) for q in questions] ∪ [EU(take(a)) for a in actions])
```

### ❌ DO NOT: Let LLM choose actions
```julia
# WRONG
action = llm.choose_best_action(state, actions)
```
```julia
# RIGHT
beliefs = update_beliefs(beliefs, llm.answer_question(state, question))
action = argmax_eu(beliefs, actions)
```

### ❌ DO NOT: Add loop detection
```julia
# WRONG
if action in recent_actions:
    action = random_action()
```
```julia
# RIGHT
# If looping, fix the state abstraction or prior beliefs
```

### ❌ DO NOT: Use state change as ground truth
```julia
# WRONG
helped = (new_state != old_state)
```
```julia
# RIGHT
helped = (reward > 0)
```

### ❌ DO NOT: Use 0.5 prior for action success
```julia
# WRONG
prior_helps = 0.5  # Unrealistic
```
```julia
# RIGHT
prior_helps = 1.0 / n_actions  # Most actions don't help
```

---

## CONFIGURATION

```julia
Base.@kwdef struct AgentConfig
    # Trajectory Planning
    horizon::Int = 20                 # Planning horizon (steps)
    n_trajectory_samples::Int = 10    # Thompson samples per decision
    mcts_iterations::Int = 100        # MCTS iterations per sample
    discount::Float64 = 0.99
    
    # Unified Decision Costs
    ask_cost::Float64 = 0.01          # Cost per sensor query
    compute_cost::Float64 = 0.001     # Cost per extra planning iteration
    refine_cost::Float64 = 0.01       # Cost of abstraction refinement
    
    # Question Generation
    max_questions_per_step::Int = 5   # Limit recursion
    question_budget::Int = 10         # Questions to evaluate per decision
    
    # Priors (can be overridden by meta-learning)
    action_prior_success::Float64 = 0.05  # P(random action helps) ≈ 1/20
    sensor_prior_tpr::Tuple = (2.0, 1.0)  # Beta prior for TPR
    sensor_prior_fpr::Tuple = (1.0, 2.0)  # Beta prior for FPR
    dynamics_prior_α::Float64 = 0.1       # Dirichlet concentration
    reward_prior_mean::Float64 = 0.0
    reward_prior_var::Float64 = 1.0
    
    # State Abstraction
    use_bisimulation::Bool = true
    signature_similarity_threshold::Float64 = 0.95
    
    # Meta-Learning
    meta_learning_enabled::Bool = true
    meta_update_frequency::Int = 10   # Episodes between meta-updates
    min_tasks_for_meta::Int = 3       # Tasks before using meta-priors
end
```

---

## FILE STRUCTURE

```
BayesianAgents/
├── Project.toml
├── README.md
├── CLAUDE.md                    # This file (for Claude Code)
├── UNIFIED_SPEC.md              # Complete mathematical specification
│
├── src/
│   ├── BayesianAgents.jl        # Main module, exports
│   │
│   ├── core/
│   │   ├── types.jl             # Abstract types, Decision types
│   │   ├── beliefs.jl           # AgentBeliefs hierarchy
│   │   ├── agent.jl             # BayesianAgent struct and loop
│   │   ├── decision.jl          # Unified decision function
│   │   └── config.jl            # AgentConfig
│   │
│   ├── models/
│   │   ├── tabular.jl           # TabularWorldModel (Dirichlet-Categorical)
│   │   ├── sampled.jl           # SampledDynamics for Thompson
│   │   └── trajectory.jl        # Trajectory representation
│   │
│   ├── sensors/
│   │   ├── binary.jl            # BinarySensor with Beta posteriors
│   │   ├── llm.jl               # LLMSensor for language models
│   │   └── voi.jl               # Trajectory-level VOI calculations
│   │
│   ├── planning/
│   │   ├── thompson_mcts.jl     # MCTS with Thompson Sampling
│   │   ├── trajectory.jl        # Trajectory planning utilities
│   │   └── myopic.jl            # One-step lookahead (baseline)
│   │
│   ├── abstraction/
│   │   ├── identity.jl          # No abstraction (baseline)
│   │   ├── bisimulation.jl      # Behavioural equivalence classes
│   │   └── signature.jl         # Behavioural signatures
│   │
│   ├── meta/
│   │   ├── priors.jl            # MetaPriors struct
│   │   ├── update.jl            # Meta-learning updates
│   │   └── transfer.jl          # Cross-task transfer
│   │
│   └── worlds/
│       ├── interface.jl         # World abstract type
│       ├── gridworld.jl         # Testing environment
│       └── jericho.jl           # Interactive Fiction
│
├── test/
│   ├── runtests.jl
│   ├── test_trajectory_voi.jl   # Trajectory-level VOI
│   ├── test_sensor.jl           # Sensor learning
│   ├── test_bisimulation.jl     # State abstraction
│   ├── test_unified_decision.jl # All decision types
│   ├── test_meta_learning.jl    # Meta-prior updates
│   └── test_integration.jl
│
└── examples/
    ├── gridworld.jl
    ├── interactive_fiction.jl
    └── meta_learning_demo.jl
```

---

## TESTING CHECKLIST

A correct implementation should:

### Core Functionality
1. ✅ **Trajectory planning works**: Multi-step plans, not myopic
2. ✅ **Thompson Sampling explores**: Different posterior samples → different trajectories
3. ✅ **Unified decision**: Act, ask, plan, refine all compared by EU
4. ✅ **TVOI computed correctly**: Trajectory-level, not single-step

### Sensor System
5. ✅ **Ask when TVOI > cost**: Principled, not heuristic
6. ✅ **Stop asking when beliefs converge**: VOI → 0 as uncertainty decreases
7. ✅ **Learn sensor reliability**: TPR/FPR converge to true values
8. ✅ **Ground truth = reward > 0**: Not state change

### State Abstraction
9. ✅ **No loops on equivalent states**: Bisimulation merges them
10. ✅ **Contradictions trigger refinement**: Splits when outcomes differ
11. ✅ **Signatures computed correctly**: Match theory

### Learning
12. ✅ **Score improves over episodes**: Later > earlier
13. ✅ **Asking decreases over time**: As beliefs become certain

### Meta-Learning (if enabled)
14. ✅ **Priors improve across tasks**: Better calibration
15. ✅ **Faster learning on new tasks**: Fewer samples needed
16. ✅ **Prior updates correct**: Empirical Bayes on task summaries

---

## DEBUGGING GUIDE

### Sensor Issues

**Problem: Agent always asks, never acts**
- Check: Is `ask_cost` too low? Is TVOI being computed correctly?
- Fix: Ensure TVOI compares *trajectory* values, not immediate rewards

**Problem: Agent never asks**
- Check: Is `ask_cost` too high? Are sensor priors reasonable?
- Fix: Start with `ask_cost = 0.01`, TPR prior (2,1), FPR prior (1,2)

**Problem: Sensor reliability doesn't improve**
- Check: Is ground truth being determined? Are updates happening?
- Fix: Only use `reward > 0` as ground truth; log update calls

### State Abstraction Issues

**Problem: Agent loops on equivalent states**
- Check: Is bisimulation enabled? Are signatures being recorded?
- Fix: Enable `use_bisimulation = true`, verify signature computation
- Debug: Print signatures for looping states—they should differ

**Problem: Too much merging (different states collapsed)**
- Check: Is `signature_similarity_threshold` too low?
- Fix: Increase threshold (e.g., 0.95 → 0.99)

**Problem: Not enough merging (still loops)**
- Check: Are enough transitions recorded? Is threshold too high?
- Fix: Need sufficient data before signatures are reliable

### Trajectory Planning Issues

**Problem: Agent acts myopically despite horizon > 1**
- Check: Is MCTS actually running? Is horizon passed correctly?
- Fix: Verify MCTS iterations > 0, horizon > 0
- Debug: Print trajectory lengths—should be > 1

**Problem: Poor trajectory planning**
- Check: Is horizon sufficient? Are MCTS iterations enough?
- Fix: Increase horizon, iterations; check reward signal
- Debug: Print sampled trajectories and their values

**Problem: Same trajectory every time (no exploration)**
- Check: Is Thompson Sampling working? Different posterior samples?
- Fix: Verify `sample_dynamics()` actually samples, not point estimates

### Meta-Learning Issues

**Problem: Meta-learning not improving performance**
- Check: Enough tasks completed? Are prior updates implemented?
- Fix: Verify hierarchical updates; need sufficient task diversity

**Problem: Priors too confident too fast**
- Check: Is pseudo-count scaling appropriate?
- Fix: Reduce scaling factor in meta-updates

**Problem: Priors not updating at all**
- Check: Is `meta_learning_enabled = true`? Is `update_meta!` called?
- Fix: Verify task summaries are being passed to meta-learner

### General Performance Issues

**Problem: Agent very slow**
- Profile: Usually MCTS is the bottleneck
- Fix: Reduce `mcts_iterations`, `n_trajectory_samples`, or `horizon`
- Consider: Caching compiled trajectories

**Problem: Poor asymptotic performance**
- Check: Is world model learning? Is abstraction correct?
- Fix: Verify Bayesian updates happening; check for bugs in posterior

---

## IMPLEMENTATION ORDER

### Phase 1: Core Infrastructure
1. **Core types** (`types.jl`) — Decision types, abstract interfaces
2. **Binary sensor** (`binary.jl`) — Beta posteriors, Bayes updates
3. **Tabular world model** (`tabular.jl`) — Dirichlet-Categorical
4. **GridWorld** (`gridworld.jl`) — Testing environment

### Phase 2: Single-Step Agent (Baseline)
5. **Myopic VOI** (`voi.jl`) — Single-step value of information
6. **Simple decision** (`decision.jl`) — Act vs Ask comparison
7. **Agent loop** (`agent.jl`) — Basic step/update cycle
8. **Tests**: Sensor learning, VOI correctness

### Phase 3: Trajectory Planning
9. **Trajectory representation** (`trajectory.jl`)
10. **Thompson MCTS** (`thompson_mcts.jl`) — Sample, plan, execute
11. **Trajectory VOI** — Extend VOI to trajectory-level
12. **Tests**: Verify multi-step planning beats myopic

### Phase 4: State Abstraction
13. **Signatures** (`signature.jl`) — Behavioural fingerprints
14. **Bisimulation** (`bisimulation.jl`) — Equivalence classes
15. **Contradiction detection and refinement**
16. **Tests**: Verify no loops on equivalent states

### Phase 5: Unified Decision
17. **Full unified_decide()** — All four decision types
18. **Planning VOI** — Value of more computation
19. **Refinement VOI** — Value of splitting abstractions
20. **Tests**: Verify correct option selection

### Phase 6: Meta-Learning
21. **MetaPriors struct** (`priors.jl`)
22. **Cross-task updates** (`update.jl`)
23. **Prior injection for new tasks**
24. **Tests**: Verify priors improve across tasks

### Phase 7: Real Worlds
25. **LLM sensor** (`llm.jl`) — Ollama integration
26. **Jericho world** (`jericho.jl`) — IF games
27. **Integration tests** — Full agent on real domains

### Phase 8: Polish
28. **Documentation** — Complete API docs
29. **Examples** — Working demos
30. **Performance** — Profile and optimise

---

## KEY EQUATIONS SUMMARY

**Trajectory Value:**
```
V(τ) = Σₜ γᵗ r(sₜ, aₜ)
```

**Expected Utility of Acting (Trajectory-Level):**
```
EU(take(a)) = E_{θ~P(θ|H)} [ max_{τ: τ₀=a} V(τ | θ) ]
```

**Expected Utility of Asking (Trajectory-Level):**
```
EU(ask(q,k)) = E_{answer}[max_τ V(τ) | answer] - cost_k
```

**Trajectory Value of Information:**
```
TVOI(q,k) = E_{answer}[max_τ V(τ) | answer] - max_τ V(τ) | current
```

**Unified Decision Rule:**
```
decision = argmax_{d ∈ {act, ask, plan, refine}} EU(d)
```

**Bayesian Sensor Update:**
```
P(true | yes) = TPR · P(true) / [TPR · P(true) + FPR · P(false)]
```

**Sensor Reliability Learning:**
```
TPR ~ Beta(α_tp, β_tp)  updated by: yes∧true → α_tp++, no∧true → β_tp++
FPR ~ Beta(α_fp, β_fp)  updated by: yes∧false → α_fp++, no∧false → β_fp++
```

**Bisimulation Condition:**
```
s₁ ~ s₂ ⟺ ∀a: P(r|s₁,a) = P(r|s₂,a) ∧ P(φ(s')|s₁,a) = P(φ(s')|s₂,a)
```

**Meta-Learning (Hierarchical Priors):**
```
α₀ ~ P(α₀ | previous_tasks)      # Dynamics concentration
TPR_prior ~ P(α_tp, β_tp | previous_sensors)
```
