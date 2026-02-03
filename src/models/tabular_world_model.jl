"""
    TabularWorldModel

A Bayesian world model for discrete state-action spaces using:
- Dirichlet-Categorical for transition probabilities
- Normal-Gamma for reward distributions

Supports Thompson Sampling via sampling from posteriors.
"""

using Distributions

"""
    TabularWorldModel

Maintains Bayesian beliefs over transition and reward dynamics.
"""
mutable struct TabularWorldModel <: WorldModel
    # Transition model: (state, action) → Dirichlet over next states
    # Stored as counts: transition_counts[(s,a)][s'] = count
    transition_counts::Dict{Tuple{Any,Any}, Dict{Any, Float64}}
    
    # Reward model: (state, action) → Normal-Gamma posterior
    # Stored as sufficient statistics
    reward_stats::Dict{Tuple{Any,Any}, NamedTuple{(:n, :mean, :sum_sq), Tuple{Int, Float64, Float64}}}
    
    # Prior hyperparameters
    transition_prior::Float64  # Dirichlet concentration (pseudocount per state)
    reward_prior_mean::Float64
    reward_prior_variance::Float64
    
    # Known states (discovered during interaction)
    known_states::Set{Any}
end

"""
    TabularWorldModel(; transition_prior=0.1, reward_prior_mean=0.0, reward_prior_variance=1.0)

Create a new tabular world model with specified priors.
"""
function TabularWorldModel(;
    transition_prior::Float64 = 0.1,
    reward_prior_mean::Float64 = 0.0,
    reward_prior_variance::Float64 = 1.0
)
    return TabularWorldModel(
        Dict{Tuple{Any,Any}, Dict{Any, Float64}}(),
        Dict{Tuple{Any,Any}, NamedTuple{(:n, :mean, :sum_sq), Tuple{Int, Float64, Float64}}}(),
        transition_prior,
        reward_prior_mean,
        reward_prior_variance,
        Set{Any}()
    )
end

"""
    update!(model::TabularWorldModel, s, a, r, s′)

Update the model with an observed transition.
"""
function update!(model::TabularWorldModel, s, a, r, s′)
    key = (s, a)
    
    # Update transition counts
    if !haskey(model.transition_counts, key)
        model.transition_counts[key] = Dict{Any, Float64}()
    end
    model.transition_counts[key][s′] = get(model.transition_counts[key], s′, model.transition_prior) + 1.0
    
    # Update reward statistics (online mean and variance)
    if !haskey(model.reward_stats, key)
        model.reward_stats[key] = (n=0, mean=model.reward_prior_mean, sum_sq=0.0)
    end
    
    stats = model.reward_stats[key]
    n_new = stats.n + 1
    delta = r - stats.mean
    mean_new = stats.mean + delta / n_new
    delta2 = r - mean_new
    sum_sq_new = stats.sum_sq + delta * delta2
    
    model.reward_stats[key] = (n=n_new, mean=mean_new, sum_sq=sum_sq_new)
    
    # Track known states
    push!(model.known_states, s)
    push!(model.known_states, s′)
    
    return nothing
end

"""
    transition_dist(model::TabularWorldModel, s, a) → Dict{state, probability}

Return the posterior predictive distribution over next states.
"""
function transition_dist(model::TabularWorldModel, s, a)
    key = (s, a)
    
    if !haskey(model.transition_counts, key)
        # No observations — return uniform over known states
        n_states = max(1, length(model.known_states))
        return Dict(state => 1.0 / n_states for state in model.known_states)
    end
    
    counts = model.transition_counts[key]
    total = sum(values(counts))
    
    return Dict(state => count / total for (state, count) in counts)
end

"""
    reward_dist(model::TabularWorldModel, s, a) → Normal

Return the posterior predictive distribution over rewards.
"""
function reward_dist(model::TabularWorldModel, s, a)
    key = (s, a)
    
    if !haskey(model.reward_stats, key) || model.reward_stats[key].n == 0
        # No observations — return prior
        return Normal(model.reward_prior_mean, sqrt(model.reward_prior_variance))
    end
    
    stats = model.reward_stats[key]
    
    if stats.n == 1
        # Single observation — use prior variance
        return Normal(stats.mean, sqrt(model.reward_prior_variance))
    end
    
    # Posterior predictive (approximation)
    variance = stats.sum_sq / (stats.n - 1)
    std = sqrt(max(variance, 1e-6))
    
    return Normal(stats.mean, std)
end

"""
    sample_dynamics(model::TabularWorldModel) → SampledDynamics

Sample a concrete dynamics model from the posterior for Thompson Sampling.
Returns a callable object that gives deterministic transitions and rewards.
"""
function sample_dynamics(model::TabularWorldModel)
    # Sample transition probabilities from Dirichlet posteriors
    sampled_transitions = Dict{Tuple{Any,Any}, Any}()
    
    for (key, counts) in model.transition_counts
        states = collect(keys(counts))
        alphas = [counts[s] for s in states]
        
        # Sample from Dirichlet
        probs = rand(Dirichlet(alphas))
        
        # Store as categorical distribution
        sampled_transitions[key] = (states=states, probs=probs)
    end
    
    # Sample reward means from Normal posteriors
    sampled_rewards = Dict{Tuple{Any,Any}, Float64}()
    
    for (key, stats) in model.reward_stats
        dist = reward_dist(model, key[1], key[2])
        sampled_rewards[key] = rand(dist)
    end
    
    return SampledDynamics(
        sampled_transitions,
        sampled_rewards,
        model.known_states,
        model.reward_prior_mean
    )
end

"""
    SampledDynamics

A sampled world model for use in planning.
"""
struct SampledDynamics
    transitions::Dict{Tuple{Any,Any}, Any}
    rewards::Dict{Tuple{Any,Any}, Float64}
    known_states::Set{Any}
    default_reward::Float64
end

"""
    sample_next_state(dynamics::SampledDynamics, s, a) → s′

Sample a next state from the sampled dynamics.
"""
function sample_next_state(dynamics::SampledDynamics, s, a)
    key = (s, a)
    
    if !haskey(dynamics.transitions, key)
        # Unknown transition — return random known state or same state
        if isempty(dynamics.known_states)
            return s
        end
        return rand(collect(dynamics.known_states))
    end
    
    trans = dynamics.transitions[key]
    idx = rand(Categorical(trans.probs))
    return trans.states[idx]
end

"""
    get_reward(dynamics::SampledDynamics, s, a) → Float64

Get the sampled reward for a state-action pair.
"""
function get_reward(dynamics::SampledDynamics, s, a)
    key = (s, a)
    return get(dynamics.rewards, key, dynamics.default_reward)
end

"""
    entropy(model::TabularWorldModel) → Float64

Return the entropy of the model posterior (sum over all state-action pairs).
"""
function entropy(model::TabularWorldModel)
    total_entropy = 0.0
    
    for (key, counts) in model.transition_counts
        total = sum(values(counts))
        for count in values(counts)
            p = count / total
            if p > 0
                total_entropy -= p * log(p)
            end
        end
    end
    
    return total_entropy
end

"""
    information_gain(model::TabularWorldModel, s, a, s′) → Float64

Compute the information gain from observing a transition.
This is used for intrinsic motivation.
"""
function information_gain(model::TabularWorldModel, s, a, s′)
    key = (s, a)
    
    if !haskey(model.transition_counts, key)
        # First observation of this state-action pair — high information gain
        return 1.0
    end
    
    counts = model.transition_counts[key]
    total = sum(values(counts))
    
    # Entropy before
    entropy_before = 0.0
    for count in values(counts)
        p = count / total
        if p > 0
            entropy_before -= p * log(p)
        end
    end
    
    # Simulate update and compute entropy after
    new_counts = copy(counts)
    new_counts[s′] = get(new_counts, s′, model.transition_prior) + 1.0
    new_total = total + 1.0
    
    entropy_after = 0.0
    for count in values(new_counts)
        p = count / new_total
        if p > 0
            entropy_after -= p * log(p)
        end
    end
    
    return entropy_before - entropy_after
end
