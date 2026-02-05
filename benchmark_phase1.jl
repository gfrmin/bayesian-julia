"""
Phase 1 Benchmark: Baseline vs FactoredWorldModel

Compares TabularWorldModel (baseline) with FactoredWorldModel (Phase 1 MVBN).

Metrics tracked:
- Score: Game points earned
- Steps: Actions taken
- First Reward Step: When first non-zero reward received
- Loop Count: Repeated (state, action) pairs
- States Learned: Unique abstract states visited
- Transitions: Unique transitions learned
- Model Efficiency: States/transitions ratio

Run with:
  julia --project=. benchmark_phase1.jl [game_path] [--episodes N] [--steps N] [--verbose]
"""

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using BayesianAgents
using Random
using Printf
using Statistics

# ============================================================================
# STATE ABSTRACTION FOR BASELINE (TabularWorldModel compatibility)
# ============================================================================

"""Hash-based abstractor for TabularWorldModel (baseline)"""
struct HashStateAbstractor <: StateAbstractor end

function BayesianAgents.abstract_state(::HashStateAbstractor, obs)
    loc = hasproperty(obs, :location) ? obs.location : "unknown"
    inv = hasproperty(obs, :inventory) ? obs.inventory : ""
    inv_hash = string(hash(inv), base=16)
    inv_short = length(inv_hash) >= 8 ? inv_hash[1:8] : inv_hash
    if loc == "unknown"
        h = hasproperty(obs, :state_hash) ? obs.state_hash : hash(obs)
        return "hash_$(string(h, base=16))"
    end
    return "$(loc)|$(inv_short)"
end

BayesianAgents.record_transition!(::HashStateAbstractor, s, a, r, s′) = nothing
BayesianAgents.check_contradiction(::HashStateAbstractor) = nothing
BayesianAgents.refine!(::HashStateAbstractor, contradiction) = nothing

# ============================================================================
# BENCHMARK METRICS
# ============================================================================

"""Track metrics for an episode"""
struct EpisodeMetrics
    score::Int
    steps::Int
    first_reward_step::Union{Int, Nothing}
    loop_count::Int
    unique_states::Int
    unique_transitions::Int
    final_reward::Float64
end

"""Count loops: repeated (state, action) pairs"""
function count_loops(state_action_pairs::Vector)
    if length(state_action_pairs) < 2
        return 0
    end
    seen = Set()
    loops = 0
    for pair in state_action_pairs
        if pair in seen
            loops += 1
        else
            push!(seen, pair)
        end
    end
    return loops
end

"""Run a single episode and collect metrics"""
function run_episode_with_metrics(
    agent::BayesianAgent,
    max_steps::Int
)
    reset!(agent)

    state_action_pairs = Vector()
    first_reward_step = nothing

    for step in 1:max_steps
        action, obs, reward, done = act!(agent)

        # Track state-action pair for loop detection
        current_state = agent.current_abstract_state
        push!(state_action_pairs, (current_state, action))

        # Track first non-zero reward
        if reward > 0 && isnothing(first_reward_step)
            first_reward_step = step
        end

        done && break
    end

    # Compute metrics
    score = agent.world.current_score
    steps = agent.step_count
    loop_count = count_loops(state_action_pairs)

    # Get model statistics
    if isa(agent.model, TabularWorldModel)
        unique_states = length(agent.model.known_states)
        unique_transitions = length(agent.model.transition_counts)
    elseif isa(agent.model, FactoredWorldModel)
        unique_states = length(agent.model.known_locations) * length(agent.model.known_objects)
        unique_transitions = length(agent.model.transitions)
    else
        unique_states = 0
        unique_transitions = 0
    end

    return EpisodeMetrics(
        score,
        steps,
        first_reward_step,
        loop_count,
        unique_states,
        unique_transitions,
        agent.total_reward
    )
end

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

function run_benchmark(;
    game_path::String,
    n_episodes::Int = 10,
    max_steps::Int = 150,
    mcts_iterations::Int = 60,
    mcts_depth::Int = 8,
    verbose::Bool = true
)
    println("=" ^ 80)
    println("PHASE 1 BENCHMARK: Baseline vs FactoredWorldModel")
    println("=" ^ 80)
    println("Game:     $(basename(game_path))")
    println("Episodes: $n_episodes × 2 (baseline + phase1)")
    println("Steps:    $max_steps per episode")
    println("MCTS:     $mcts_iterations iterations, depth $mcts_depth")
    println()

    # ========================================================================
    # BASELINE: TabularWorldModel + Hash Abstractor
    # ========================================================================
    println("=" ^ 80)
    println("BASELINE: TabularWorldModel + Hash State Abstraction")
    println("=" ^ 80)
    println()

    baseline_metrics = EpisodeMetrics[]

    for episode in 1:n_episodes
        # Create fresh baseline agent
        world_baseline = JerichoWorld(game_path; max_steps=max_steps)
        # Feature extractor for TabularWorldModel: abstract_state -> features
        feature_extractor = function(abstract_state)
            # abstract_state is a string like "kitchen|a3f2b1c8"
            parts = split(string(abstract_state), "|")
            return [(:location, parts[1])]
        end

        model_baseline = TabularWorldModel(
            transition_prior = 0.1,
            reward_prior_mean = 0.0,
            reward_prior_variance = 1.0,
            feature_extractor = feature_extractor
        )
        planner = ThompsonMCTS(iterations=mcts_iterations, depth=mcts_depth, discount=0.99)
        abstractor = HashStateAbstractor()

        agent = BayesianAgent(world_baseline, model_baseline, planner, abstractor)
        metrics = run_episode_with_metrics(agent, max_steps)
        push!(baseline_metrics, metrics)

        if verbose
            println(@sprintf("  Episode %2d: score=%3d steps=%3d first_reward=%s loops=%2d states=%2d",
                episode, metrics.score, metrics.steps,
                isnothing(metrics.first_reward_step) ? "N/A" : "$(metrics.first_reward_step)",
                metrics.loop_count, metrics.unique_states))
        end
    end

    println()
    baseline_score_mean = mean(m.score for m in baseline_metrics)
    baseline_score_std = std(m.score for m in baseline_metrics)
    baseline_steps_mean = mean(m.steps for m in baseline_metrics)
    baseline_loops_mean = mean(m.loop_count for m in baseline_metrics)
    baseline_states_mean = mean(m.unique_states for m in baseline_metrics)

    baseline_first_rewards = [m.first_reward_step for m in baseline_metrics if !isnothing(m.first_reward_step)]
    baseline_first_reward_mean = isempty(baseline_first_rewards) ? NaN : mean(baseline_first_rewards)

    println("Baseline Summary:")
    println(@sprintf("  Score:            %.1f ± %.1f (min=%d, max=%d)",
        baseline_score_mean, baseline_score_std,
        minimum(m.score for m in baseline_metrics),
        maximum(m.score for m in baseline_metrics)))
    println(@sprintf("  Steps/Episode:    %.1f", baseline_steps_mean))
    println(@sprintf("  Loops:            %.1f", baseline_loops_mean))
    println(@sprintf("  States Learned:   %.1f", baseline_states_mean))
    println(@sprintf("  First Reward:     %.1f steps", baseline_first_reward_mean))
    println()

    # ========================================================================
    # PHASE 1: FactoredWorldModel + MinimalStateAbstractor
    # ========================================================================
    println("=" ^ 80)
    println("PHASE 1: FactoredWorldModel + MinimalState Abstraction")
    println("=" ^ 80)
    println()

    phase1_metrics = EpisodeMetrics[]

    for episode in 1:n_episodes
        # Create fresh phase1 agent
        world_phase1 = JerichoWorld(game_path; max_steps=max_steps)
        model_phase1 = FactoredWorldModel(0.1)
        planner = ThompsonMCTS(iterations=mcts_iterations, depth=mcts_depth, discount=0.99)
        abstractor = MinimalStateAbstractor()

        agent = BayesianAgent(world_phase1, model_phase1, planner, abstractor)
        metrics = run_episode_with_metrics(agent, max_steps)
        push!(phase1_metrics, metrics)

        if verbose
            println(@sprintf("  Episode %2d: score=%3d steps=%3d first_reward=%s loops=%2d states=%2d",
                episode, metrics.score, metrics.steps,
                isnothing(metrics.first_reward_step) ? "N/A" : "$(metrics.first_reward_step)",
                metrics.loop_count, metrics.unique_states))
        end
    end

    println()
    phase1_score_mean = mean(m.score for m in phase1_metrics)
    phase1_score_std = std(m.score for m in phase1_metrics)
    phase1_steps_mean = mean(m.steps for m in phase1_metrics)
    phase1_loops_mean = mean(m.loop_count for m in phase1_metrics)
    phase1_states_mean = mean(m.unique_states for m in phase1_metrics)

    phase1_first_rewards = [m.first_reward_step for m in phase1_metrics if !isnothing(m.first_reward_step)]
    phase1_first_reward_mean = isempty(phase1_first_rewards) ? NaN : mean(phase1_first_rewards)

    println("Phase 1 Summary:")
    println(@sprintf("  Score:            %.1f ± %.1f (min=%d, max=%d)",
        phase1_score_mean, phase1_score_std,
        minimum(m.score for m in phase1_metrics),
        maximum(m.score for m in phase1_metrics)))
    println(@sprintf("  Steps/Episode:    %.1f", phase1_steps_mean))
    println(@sprintf("  Loops:            %.1f", phase1_loops_mean))
    println(@sprintf("  States Learned:   %.1f", phase1_states_mean))
    println(@sprintf("  First Reward:     %.1f steps", phase1_first_reward_mean))
    println()

    # ========================================================================
    # COMPARISON
    # ========================================================================
    println("=" ^ 80)
    println("COMPARISON (Phase 1 vs Baseline)")
    println("=" ^ 80)
    println()

    score_diff = phase1_score_mean - baseline_score_mean
    score_pct = (score_diff / max(baseline_score_mean, 1.0)) * 100
    loops_reduction = baseline_loops_mean - phase1_loops_mean
    loops_pct = (loops_reduction / max(baseline_loops_mean, 1.0)) * 100
    states_increase = phase1_states_mean - baseline_states_mean
    states_pct = (states_increase / max(baseline_states_mean, 1.0)) * 100
    first_reward_improvement = baseline_first_reward_mean - phase1_first_reward_mean

    println("Score:")
    println(@sprintf("  Baseline: %.1f ± %.1f", baseline_score_mean, baseline_score_std))
    println(@sprintf("  Phase 1:  %.1f ± %.1f", phase1_score_mean, phase1_score_std))
    println(@sprintf("  Change:   %.1f (%.1f%%) %s", score_diff, score_pct,
        score_diff > 0 ? "↑ IMPROVEMENT" : (score_diff < 0 ? "↓ REGRESSION" : "= NO CHANGE")))
    println()

    println("Loops (lower is better):")
    println(@sprintf("  Baseline: %.1f", baseline_loops_mean))
    println(@sprintf("  Phase 1:  %.1f", phase1_loops_mean))
    println(@sprintf("  Change:   %.1f (%.1f%% reduction) %s", loops_reduction, loops_pct,
        loops_reduction > 0 ? "↑ BETTER" : (loops_reduction < 0 ? "↓ WORSE" : "= SAME")))
    println()

    println("State Space (higher is better for discrimination):")
    println(@sprintf("  Baseline: %.1f unique states", baseline_states_mean))
    println(@sprintf("  Phase 1:  %.1f unique states", phase1_states_mean))
    println(@sprintf("  Change:   %.1f (%.1f%% increase)", states_increase, states_pct))
    println()

    if !isnan(first_reward_improvement) && !isnan(baseline_first_reward_mean) && !isnan(phase1_first_reward_mean)
        println("First Reward (steps, lower is better):")
        println(@sprintf("  Baseline: %.1f", baseline_first_reward_mean))
        println(@sprintf("  Phase 1:  %.1f", phase1_first_reward_mean))
        println(@sprintf("  Change:   %.1f steps %s", first_reward_improvement,
            first_reward_improvement > 0 ? "↑ FASTER" : (first_reward_improvement < 0 ? "↓ SLOWER" : "= SAME")))
        println()
    else
        println("First Reward:")
        println("  (Not applicable: FactoredWorldModel reward learning not yet implemented)")
        println()
    end

    # ========================================================================
    # STATISTICAL SIGNIFICANCE
    # ========================================================================
    println("Statistical Analysis:")

    # Paired t-test conceptually (simple version for visibility)
    baseline_scores = [m.score for m in baseline_metrics]
    phase1_scores = [m.score for m in phase1_metrics]

    t_score = (phase1_score_mean - baseline_score_mean) / sqrt((phase1_score_std^2 + baseline_score_std^2) / n_episodes)

    if abs(score_diff) > 0.1 * max(baseline_score_std, phase1_score_std)
        significance = "Potentially significant difference"
    else
        significance = "Within noise margin"
    end

    println(@sprintf("  Score difference: %.1f (t≈%.2f)", score_diff, t_score))
    println("  Assessment:       $significance")
    println()

    println("=" ^ 80)
    println("BENCHMARK COMPLETE")
    println("=" ^ 80)
    println()

    return (
        baseline = baseline_metrics,
        phase1 = phase1_metrics,
        comparison = (
            score_diff,
            score_pct,
            loops_reduction,
            loops_pct,
            states_increase,
            states_pct
        )
    )
end

# ============================================================================
# CLI
# ============================================================================

function main()
    game_path = nothing
    n_episodes = 5
    max_steps = 150
    verbose = true

    args = ARGS
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--episodes" && i < length(args)
            i += 1
            n_episodes = parse(Int, args[i])
        elseif arg == "--steps" && i < length(args)
            i += 1
            max_steps = parse(Int, args[i])
        elseif arg == "--quiet"
            verbose = false
        elseif !startswith(arg, "-") && isnothing(game_path)
            game_path = arg
        end
        i += 1
    end

    if isnothing(game_path)
        game_path = "/home/g/Sync/git/bayesian-agents/bayesian-if-agent/games/enchanter.z3"
    end

    if !isfile(game_path)
        println("ERROR: Game file not found: $game_path")
        exit(1)
    end

    run_benchmark(
        game_path = game_path,
        n_episodes = n_episodes,
        max_steps = max_steps,
        verbose = verbose
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
