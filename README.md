# Bayesian Agents

A Julia framework for **Bayes-Adaptive POMDPs** — agents that maintain uncertainty over both world state and world dynamics, planning in a way that naturally balances exploration and exploitation.

## Philosophy

**All behaviour derives from expected utility maximisation. No hacks.**

- Uncertainty is first-class: beliefs are distributions, not point estimates
- The agent *learns* the world: dynamics, observations, rewards are uncertain
- Sensors are fallible: LLMs, heuristics, oracles all have learned reliability
- Simplicity is cost, not prior: we prefer simple models because they're cheaper to use

## Quick Start

```julia
using BayesianAgents

# Create a world
world = GridWorld(width=10, height=10)

# Create a Bayesian world model
model = TabularWorldModel(transition_prior=0.1)

# Create a planner (Thompson MCTS)
planner = ThompsonMCTS(iterations=100, depth=10)

# Create the agent
agent = BayesianAgent(world, model, planner, IdentityAbstractor())

# Run episodes
for episode in 1:100
    reward = run_episode!(agent)
    println("Episode $episode: reward = $reward")
end
```

## Architecture

```
World Interface → State Abstractor → World Model → Planner → Decision
      ↑                                    ↑
      │                                    │
   Jericho                            Sensors
   GridWorld                          (LLM, etc.)
   Gymnasium
   Real Data
```

## Key Components

### World Model (`TabularWorldModel`)
- Dirichlet-Categorical for transitions
- Normal-Gamma for rewards
- Supports Thompson Sampling via posterior sampling

### Planner (`ThompsonMCTS`)
- Monte Carlo Tree Search with Thompson Sampling
- Samples a world model, plans optimally under it
- No exploration bonus needed — exploration emerges naturally

### State Abstractor (`BisimulationAbstractor`)
- Learns equivalence classes based on behavioural signatures
- Solves the "functionally equivalent states" problem
- Automatically refines when contradictions detected

### Sensors (`BinarySensor`, `LLMSensor`)
- Yes/no queries with learned reliability (TPR/FPR)
- Bayesian belief updates from sensor answers
- Value of Information calculations for when to query

## Connecting to Different Worlds

### Interactive Fiction (Jericho)

```julia
using PyCall
world = JerichoWorld("games/zork1.z5")
# ... rest is the same
```

### OpenAI Gym (via PyCall)

```julia
# See worlds/gymnasium_world.jl for adapter
```

### Real Data Streams

Implement the `World` interface:
```julia
struct MyDataWorld <: World
    data_source::Any
end

reset!(w::MyDataWorld) = fetch_initial_state(w.data_source)
step!(w::MyDataWorld, action) = apply_and_observe(w.data_source, action)
actions(w::MyDataWorld, obs) = available_actions(obs)
```

## Design Principles

1. **Expected utility maximisation** — every decision is `argmax E[U]`
2. **No arbitrary thresholds** — parameters have principled interpretations
3. **Bayesian throughout** — beliefs, updates, planning all probabilistic
4. **Modular** — swap planners, models, abstractors independently
5. **Testable** — every component can be unit tested

## Project Structure

```
bayesian_agents/
├── src/
│   ├── BayesianAgents.jl      # Main module
│   ├── models/
│   │   ├── tabular_world_model.jl
│   │   └── binary_sensor.jl
│   ├── planners/
│   │   └── thompson_mcts.jl
│   └── abstractors/
│       ├── identity_abstractor.jl
│       └── bisimulation_abstractor.jl
├── worlds/
│   ├── grid_world.jl
│   └── jericho_world.jl
├── examples/
│   └── gridworld_example.jl
├── docs/
│   └── SPECIFICATION.md
└── Project.toml
```

## References

- Ross, Chaib-draa, Pineau (2008). "Bayes-Adaptive POMDPs"
- Silver & Veness (2010). "Monte-Carlo Planning in Large POMDPs"
- Katt et al. (2018). "Bayesian Reinforcement Learning in Factored POMDPs"
- Egorov et al. (2017). "POMDPs.jl: A Framework for Sequential Decision Making"

## License

MIT
