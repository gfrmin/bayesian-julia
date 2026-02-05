"""
    Bayesian Agent Example: Interactive Fiction via Jericho

This example demonstrates the **5-stage Bayesian framework** on IF games:
1. **Stage 1: MVBN** ✓ - MinimalState (location, inventory) with FactoredWorldModel (ACTIVE)
2. **Stage 2: Variable Discovery** - Auto-discover state variables from text (coming soon)
3. **Stage 3: Structure Learning** - Learn causal dependencies (Bayesian networks) (coming soon)
4. **Stage 4: Action Schemas** - Generalize across action instances (coming soon)
5. **Stage 5: Goal Planning** - Extract objectives, plan toward goals (coming soon)

Stage 1 implements factored state representation with Dirichlet-Categorical dynamics learning.
All learning via Bayesian inference. No heuristics.

Requirements:
    pip install jericho
    ollama serve && ollama pull llama3.1

Run with:
    julia --project=. examples/jericho_agent.jl path/to/enchanter.z3 --llm --model llama3.1 --episodes 3 --steps 150
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Random
using Printf

include("../src/BayesianAgents.jl")
using .BayesianAgents

# ============================================================================
# JERICHO STATE ABSTRACTOR
# ============================================================================

"""
Maps JerichoWorld NamedTuple observations to compact state keys.

The raw observation contains text, score, steps, location, inventory, and
state_hash. Without abstraction the TabularWorldModel sees every observation
as unique (text and steps change each step). This abstractor reduces the
observation to "location|inv_hash" so the model can learn transitions.

TODO: Use MinimalState for FactoredWorldModel integration once planner is updated.
"""
struct JerichoStateAbstractor <: StateAbstractor end

function BayesianAgents.abstract_state(::JerichoStateAbstractor, obs)
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

BayesianAgents.record_transition!(::JerichoStateAbstractor, s, a, r, s′) = nothing
BayesianAgents.check_contradiction(::JerichoStateAbstractor) = nothing
BayesianAgents.refine!(::JerichoStateAbstractor, contradiction) = nothing

# ============================================================================
# JERICHO FEATURE EXTRACTOR (for factored reward model)
# ============================================================================

"""
Extract feature keys from a Jericho abstract state (e.g. "Kitchen|a3f2b1c8").

Returns features like [(:location, "Kitchen")] so the factored reward model
can generalise across states sharing the same location.
"""
function jericho_features(abstract_state)
    parts = split(string(abstract_state), "|")
    features = Any[(:location, parts[1])]
    return features
end

# ============================================================================
# OLLAMA CLIENT (pure Julia, stdlib Downloads + JSON.jl)
# ============================================================================

using Downloads
using JSON

"""
    make_ollama_client(; base_url, model, temperature, timeout)

Create a named tuple with a `query` function that calls the local Ollama
HTTP API. Compatible with the LLMSensor `llm_client.query(prompt)` contract.
"""
function make_ollama_client(;
    base_url::String = "http://localhost:11434",
    model::String = "llama3.2",
    temperature::Float64 = 0.1,
    timeout::Int = 30
)
    query_fn = function(prompt::String)
        url = "$base_url/api/generate"
        body = JSON.json(Dict(
            "model" => model,
            "prompt" => prompt,
            "temperature" => temperature,
            "stream" => false
        ))
        io = IOBuffer()
        try
            Downloads.request(url;
                method = "POST",
                headers = ["Content-Type" => "application/json"],
                input = IOBuffer(body),
                output = io,
                timeout = timeout)
            data = JSON.parse(String(take!(io)))
            return get(data, "response", "")
        catch e
            @warn "Ollama query failed" exception=e
            return ""
        end
    end
    return (query = query_fn,)
end

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

function run_jericho_experiment(;
    game_path::String,
    n_episodes::Int = 5,
    max_steps::Int = 50,
    use_llm::Bool = false,
    force_llm::Bool = false,
    ollama_model::String = "llama3.2",
    mcts_iterations::Int = 60,
    mcts_depth::Int = 8,
    verbose::Bool = true,
    enable_stage2::Bool = false,
    enable_stage3::Bool = false,
    enable_stage4::Bool = false,
    enable_stage5::Bool = false
)
    println("=" ^ 60)
    println("BAYESIAN IF AGENT")
    println("=" ^ 60)
    println("Game:     $(basename(game_path))")
    println("Episodes: $n_episodes")
    println("Steps:    $max_steps per episode")
    println("LLM:      $(use_llm ? ollama_model : "disabled")")
    println("MCTS:     $(mcts_iterations) iterations, depth $(mcts_depth)")
    println()

    # Create world
    world = JerichoWorld(game_path; max_steps=max_steps)
    println("Max score: $(world.max_score)")
    println()

    # Stage 1: MVBN with factored state representation
    # Learns P(location' | location, action) and P(obj' | obj, action)
    model = FactoredWorldModel(0.1)  # Dirichlet concentration prior

    # Create planner
    planner = ThompsonMCTS(
        iterations = mcts_iterations,
        depth = mcts_depth,
        discount = 0.99,
        ucb_c = 2.0
    )

    # Stage 1: Factored state extraction (location, inventory)
    abstractor = MinimalStateAbstractor()

    # Create sensors
    sensors = Sensor[]
    if use_llm
        println("Connecting to Ollama ($ollama_model)...")
        client = make_ollama_client(model=ollama_model, timeout=60)
        # Test connection
        test_response = client.query("Say 'ok'.")
        if isempty(test_response)
            if force_llm
                error("Ollama not responding and --force-llm specified. Start Ollama with: ollama serve")
            else
                @warn "Ollama not responding — continuing without LLM sensor"
            end
        else
            println("Ollama connected: $(strip(test_response))")
            # Skeptical prior: uniform Beta(1,1) for both TPR and FPR.
            sensor = LLMSensor("llm", client;
                prompt_template = "{question}",
                tp_prior = (2.0, 1.0),
                fp_prior = (1.0, 2.0))
            push!(sensors, sensor)
        end
        println()
    end

    # Create agent
    agent = BayesianAgent(
        world, model, planner, abstractor;
        sensors = sensors,
        config = AgentConfig(
            planning_depth = mcts_depth,
            mcts_iterations = mcts_iterations,
            sensor_cost = 0.01,
            max_queries_per_step = 3,
            enable_variable_discovery = enable_stage2,
            variable_discovery_frequency = 20,
            variable_bic_threshold = 0.0,
            enable_structure_learning = enable_stage3,
            structure_learning_frequency = 50,
            enable_action_schemas = enable_stage4,
            schema_discovery_frequency = 100,
            enable_goal_planning = enable_stage5
        )
    )

    # Run episodes - Agent created ONCE, model accumulates knowledge across episodes
    episode_rewards = Float64[]
    episode_scores = Int[]

    for episode in 1:n_episodes
        println("-" ^ 60)
        println("Episode $episode / $n_episodes")
        println("-" ^ 60)
        flush(stdout)

        # Reset world state and episode tracking, but NOT the model
        agent.current_observation = reset!(world)
        agent.current_abstract_state = abstract_state(agent.abstractor, agent.current_observation)
        agent.step_count = 0
        agent.total_reward = 0.0
        agent.trajectory = []
        # NOTE: agent.model is NOT reset - it accumulates knowledge across episodes!

        # Manual step loop for per-step output
        for step in 1:max_steps
            action, obs, reward, done = act!(agent)
            r_str = reward != 0 ? " → reward $(reward)" : ""
            sq = !isempty(agent.sensors) ? " [queries: $(agent.sensors[1].n_queries)]" : ""
            if verbose
                print(@sprintf("  %3d. %-30s%s%s\n", step, action, r_str, sq))
                flush(stdout)
            end
            done && break
        end

        push!(episode_rewards, agent.total_reward)
        push!(episode_scores, world.current_score)

        println()
        println(@sprintf("  Score:  %d / %d", world.current_score, world.max_score))
        println(@sprintf("  Steps:  %d", agent.step_count))
        println(@sprintf("  Reward: %.1f", agent.total_reward))

        # Model-agnostic state/transition reporting
        if isa(model, TabularWorldModel)
            println(@sprintf("  States learned: %d", length(model.known_states)))
            println(@sprintf("  Transitions:    %d", length(model.transition_counts)))
        elseif isa(model, FactoredWorldModel)
            num_states = length(model.known_locations) * length(model.known_objects)
            println(@sprintf("  State space:    %d", num_states))
            println(@sprintf("  Transitions:    %d", length(model.transitions)))
        end
        println()
        flush(stdout)
    end

    # Summary
    println("=" ^ 60)
    println("RESULTS")
    println("=" ^ 60)
    println(@sprintf("Episodes:         %d", n_episodes))
    println(@sprintf("Average reward:   %.2f", _mean(episode_rewards)))
    println(@sprintf("Best score:       %d / %d", maximum(episode_scores), world.max_score))

    # Model-agnostic reporting
    if isa(model, TabularWorldModel)
        println(@sprintf("States learned:   %d", length(model.known_states)))
        println(@sprintf("Transitions:      %d", length(model.transition_counts)))
    elseif isa(model, FactoredWorldModel)
        num_states = length(model.known_locations) * length(model.known_objects)
        println(@sprintf("State space:      %d", num_states))
        println(@sprintf("Transitions:      %d", length(model.transitions)))
    end

    if !isempty(sensors)
        for sensor in sensors
            println()
            println("Sensor: $(sensor.name)")
            println(@sprintf("  Queries: %d", sensor.n_queries))
            println(@sprintf("  TPR:     %.3f", tpr(sensor)))
            println(@sprintf("  FPR:     %.3f", fpr(sensor)))
        end
    end

    # Learning curve
    println()
    println("Score by episode:")
    for (i, sc) in enumerate(episode_scores)
        bar = "█" ^ max(0, sc)
        println(@sprintf("  Episode %2d: %s %d", i, bar, sc))
    end

    return (rewards = episode_rewards, scores = episode_scores, agent = agent)
end

# ============================================================================
# HELPERS
# ============================================================================

_mean(x) = isempty(x) ? 0.0 : sum(x) / length(x)

# ============================================================================
# CLI
# ============================================================================

function parse_args(args)
    game_path = nothing
    n_episodes = 5
    max_steps = 50
    use_llm = false
    force_llm = false
    ollama_model = "llama3.2"
    verbose = true
    mcts_iterations = 60
    mcts_depth = 8
    enable_stage2 = false
    enable_stage3 = false
    enable_stage4 = false
    enable_stage5 = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--episodes" && i < length(args)
            i += 1
            n_episodes = parse(Int, args[i])
        elseif arg == "--steps" && i < length(args)
            i += 1
            max_steps = parse(Int, args[i])
        elseif arg == "--llm"
            use_llm = true
        elseif arg == "--force-llm"
            use_llm = true
            force_llm = true
        elseif arg == "--model" && i < length(args)
            i += 1
            ollama_model = args[i]
        elseif arg == "--mcts-iter" && i < length(args)
            i += 1
            mcts_iterations = parse(Int, args[i])
        elseif arg == "--mcts-depth" && i < length(args)
            i += 1
            mcts_depth = parse(Int, args[i])
        elseif arg == "--quiet"
            verbose = false
        elseif arg == "--verbose"
            verbose = true
        elseif arg == "--debug"
            ENV["JULIA_DEBUG"] = "Main.BayesianAgents"
        elseif arg == "--stage2"
            enable_stage2 = true
        elseif arg == "--stage3"
            enable_stage3 = true
        elseif arg == "--stage4"
            enable_stage4 = true
        elseif arg == "--stage5"
            enable_stage5 = true
        elseif arg == "--all-stages"
            enable_stage2 = true
            enable_stage3 = true
            enable_stage4 = true
            enable_stage5 = true
        elseif !startswith(arg, "-") && isnothing(game_path)
            game_path = arg
        else
            println("Unknown argument: $arg")
        end
        i += 1
    end

    if isnothing(game_path)
        # Default: look for games in sibling project
        candidates = [
            joinpath(@__DIR__, "..", "..", "bayesian-if-agent", "games", "905.z5"),
            joinpath(@__DIR__, "..", "..", "bayesian-if-agent", "games", "pentari.z5"),
            joinpath(@__DIR__, "..", "..", "bayesian-if-agent", "games", "zork1.z5"),
        ]
        for c in candidates
            if isfile(c)
                game_path = c
                break
            end
        end
        if isnothing(game_path)
            error("""No game file specified. Usage:
  julia --project=. examples/jericho_agent.jl path/to/game.z5 [--episodes N] [--steps N] [--llm] [--model NAME] [--quiet]""")
        end
    end

    if !isfile(game_path)
        error("Game file not found: $game_path")
    end

    return (
        game_path = game_path,
        n_episodes = n_episodes,
        max_steps = max_steps,
        use_llm = use_llm,
        force_llm = force_llm,
        ollama_model = ollama_model,
        verbose = verbose,
        mcts_iterations = mcts_iterations,
        mcts_depth = mcts_depth,
        enable_stage2 = enable_stage2,
        enable_stage3 = enable_stage3,
        enable_stage4 = enable_stage4,
        enable_stage5 = enable_stage5
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_args(ARGS)
    run_jericho_experiment(;
        game_path = opts.game_path,
        n_episodes = opts.n_episodes,
        max_steps = opts.max_steps,
        use_llm = opts.use_llm,
        force_llm = opts.force_llm,
        ollama_model = opts.ollama_model,
        verbose = opts.verbose,
        mcts_iterations = opts.mcts_iterations,
        mcts_depth = opts.mcts_depth,
        enable_stage2 = opts.enable_stage2,
        enable_stage3 = opts.enable_stage3,
        enable_stage4 = opts.enable_stage4,
        enable_stage5 = opts.enable_stage5
    )
end
