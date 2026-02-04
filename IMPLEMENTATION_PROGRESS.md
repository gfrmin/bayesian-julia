# Bayesian Agents: Structured State Modeling Implementation Progress

## Project Overview

This is a 5-stage implementation of rigorous Bayesian state modeling for the Enchanter text adventure game agent. The goal is to replace opaque hash-based states with explicit factored representations that enable principled Bayesian inference and structured learning.

**Reference Document**: `CLAUDE.md` (comprehensive mathematical specification)
**Implementation Plan**: See end of `CLAUDE.md`

## Current Status: Stage 1 Foundations Complete ✓

### What Was Built

#### Core Mathematical Components (1000 lines)

**1. Dirichlet-Categorical Conjugate Pair** (`src/probability/cpd.jl`)
- Mathematical foundation: Dirichlet(α) prior + Categorical likelihood → Dirichlet(α+counts) posterior
- Used throughout for Bayesian updates
- Methods: `update!`, `predict`, `entropy`, `expected_entropy`, `mode`, `loglikelihood`
- **Key insight**: Enables efficient online learning without recomputing integrals

**2. Factored State Representation** (`src/state/`)
- `MinimalState`: (location, inventory) — replaces opaque hash keys
- `StateBelief`: maintains P(location|history) and P(object ∈ inventory|history) independently
- Each state variable tracked with its own Dirichlet-Categorical posterior
- **Key insight**: Factorization allows reasoning about state variables independently

**3. Factored World Model** (`src/models/factored_world_model.jl`)
- Learns action-conditional CPDs: P(location'|location,a), P(obj ∈ inventory'|obj ∈ inventory,a)
- Dirichlet-Categorical conjugacy enables safe online updates
- SampledFactoredDynamics: sample one concrete MDP for Thompson Sampling
- **Key insight**: Much smaller parameter space than tabular model

**4. Bayesian Inference** (`src/inference/bayesian_update.jl`)
- LLM likelihood model: P(observation_text|state_variable=value)
- Likelihood weighting for state belief updates
- **Key insight**: LLM provides evidence, not decisions

**5. Thompson Sampling MCTS** (`src/planning/factored_mcts.jl`)
- Sample dynamics from posterior: θ ~ P(θ|history)
- Sample initial state: s₀ ~ P(s|observation)
- Plan in sampled MDP via UCB-based MCTS
- **Key insight**: Automatic exploration/exploitation tradeoff

### Architecture Decisions

| Component | Design Decision | Rationale |
|-----------|-----------------|-----------|
| **State Factorization** | (location, inventory) only | MVP: minimal but sufficient for Enchanter |
| **CPD Prior** | Dirichlet(α=0.1) | Weak prior, fast learning from data |
| **Actions** | Per-action CPDs | Efficiency: learn separately per action |
| **Thompson Sampling** | Sample entire dynamics | Correct uncertainty quantification |
| **LLM Integration** | Likelihood model | Principled Bayesian combination |

### Key Innovations

1. **No Exploration Bonuses**: Uncertainty in dynamics naturally drives exploration
2. **Factored Updates**: Variables updated independently, enabling scale
3. **Online Learning**: Dirichlet conjugacy means no retraining needed
4. **Null Detection**: Automatic via expected information gain

## Next Stages (Planning)

### Stage 2: Variable Discovery
- **Goal**: Automatically discover new state variables from observations
- **Method**: LLM extracts candidate variables (door_state, lamp_lit, etc.)
- **Mechanism**: BIC model selection to decide which variables improve fit
- **Expected improvement**: 15-30 variables discovered vs 2 hardcoded

### Stage 3: Structure Learning
- **Goal**: Learn causal dependencies between variables
- **Method**: Bayesian network structure learning via BDe scores
- **Mechanism**: Greedy search over edge sets, per-action structure
- **Expected improvement**: Actions generalize better, fewer parameters needed

### Stage 4: Action Schemas
- **Goal**: Generalize actions across instances (take X for any X)
- **Method**: Cluster similar actions, extract lifted representation
- **Mechanism**: Shared parameters across object instances
- **Expected improvement**: Zero-shot transfer to new objects

### Stage 5: Goal-Directed Planning
- **Goal**: Strategic long-horizon behavior
- **Method**: Extract goals from game text, bias planning toward goals
- **Mechanism**: VOI-gated LLM queries, goal-biased MCTS rollouts
- **Expected improvement**: Achieves game objectives, score >100

## Testing & Validation

### Unit Tests Needed

```julia
# test/test_cpd.jl
- Dirichlet conjugacy holds (posterior = Dirichlet(α + counts))
- Predictive distribution correct
- Entropy computation
- Log-likelihood for model comparison

# test/test_state_belief.jl
- Factored beliefs maintained correctly
- Entropy decreases with more data
- Sampling produces valid MinimalState

# test/test_factored_world_model.jl
- Action-specific CPDs learned independently
- Correct posterior updates
- Thompson Sampling samples valid dynamics

# test/test_bayesian_update.jl
- Likelihood weighting correct
- Belief updates reasonable
- LLM calibration tracked
```

### Integration Tests

```julia
# examples/test_factored_agent.jl
- Create FactoredWorldModel agent
- Run 5 episodes on Enchanter
- Verify: module loads, agent runs, some progress made
```

### Benchmark Comparisons

Need to establish baseline metrics for later stage comparison:
- **Score**: Mean, std, min, max over 20 episodes
- **Loops**: Count of repeated (state, action) pairs
- **First Reward**: Steps until first nonzero reward
- **States Discovered**: Unique MinimalState values visited
- **LLM Queries**: Number and quality of LLM calls

## Implementation Checklist

### Stage 1 Completion
- [x] DirichletCategorical conjugate pair
- [x] MinimalState factored representation
- [x] StateBelief factored belief tracking
- [x] FactoredWorldModel online learning
- [x] Bayesian inference infrastructure
- [x] FactoredMCTS Thompson Sampling
- [x] Module loads without errors
- [ ] Unit tests pass (needs: test/ directory setup)
- [ ] Integration test passes (needs: JerichoStateAbstractor update)
- [ ] Baseline benchmarks established (needs: harness)

### Stage 2 Planning
- [ ] Variable discovery from LLM extraction
- [ ] BIC model selection
- [ ] Dynamic state expansion
- [ ] Test on Enchanter: score >35, loops <15

### Stage 3 Planning
- [ ] Bayesian network structure learning
- [ ] BDe score computation
- [ ] Greedy structure search
- [ ] Per-action learned graphs
- [ ] Test: learned structures match LLM priors

### Stage 4 Planning
- [ ] Action schema clustering
- [ ] Lifted dynamics representation
- [ ] Precondition learning
- [ ] Zero-shot transfer test
- [ ] Test: >50% success on unseen objects

### Stage 5 Planning
- [ ] Goal extraction from game text
- [ ] Goal-biased MCTS rollout
- [ ] VOI gating for LLM queries
- [ ] Integration of all components
- [ ] Test: score >100, loops <5

## How to Continue

### Immediate (Next Session)

1. **Write Unit Tests**
   - Create `test/runtests.jl` structure
   - Test DirichletCategorical conjugacy
   - Test MinimalState equality/hashing
   - Run `julia --project=. -e 'using Pkg; Pkg.test()'`

2. **Update Jericho Adapter**
   - Modify `examples/jericho_agent.jl` to use MinimalState
   - Create new StateAbstractor that returns MinimalState
   - Test on one episode of Enchanter

3. **Establish Baseline**
   - Run 20 episodes with current (legacy) agent
   - Record: score, loops, first_reward, states discovered
   - Save as `benchmark_baseline.json`

### Medium Term (Stages 2-5)

Each stage should:
1. Follow the mathematical specification in `CLAUDE.md`
2. Write unit tests as components are implemented
3. Compare to previous stage via benchmarks
4. Document any deviations from plan

## Mathematical Rigor

**Non-Negotiables**:
- All updates must be proper Bayesian inference (posterior from prior + likelihood)
- No heuristics: all decisions from expected utility maximization
- No exploration bonuses: uncertainty naturally drives exploration
- No loop detection: fix model instead

**Validation**:
- Calibration: P(event|predicted) ≈ P(predicted)
- Convergence: posteriors should sharpen with more data
- Generalization: schema transfer should work

## References

**Primary**: `CLAUDE.md` — Complete mathematical specification including:
- Hierarchical uncertainty levels (state, dynamics, structure, meta)
- Dirichlet-Categorical conjugacy derivation
- Trajectory-level VOI computation
- Bisimulation-based state abstraction
- BDe score for structure learning
- Lifted action schemas
- Goal-directed planning with intrinsic motivation

**Key Files**:
- Implementation: `src/probability/`, `src/state/`, `src/models/`, `src/inference/`, `src/planning/`
- Example: `examples/jericho_agent.jl`
- Tests: `test/` (to be created)
- Benchmarks: `benchmark_baseline.json` (to be created)

## Estimated Effort

| Stage | Lines of Code | Time | Difficulty |
|-------|---------------|------|-----------|
| 0 (Baseline) | 200 | 2h | Low |
| 1 (MVBN) | 1000 | 4h | Medium |
| 2 (Discovery) | 400 | 2h | Medium |
| 3 (Structure) | 600 | 3h | High |
| 4 (Schemas) | 500 | 3h | High |
| 5 (Planning) | 400 | 2h | Medium |
| **Total** | **3100** | **16h** | — |

This is a research-grade Bayesian RL implementation. Quality > quantity.
