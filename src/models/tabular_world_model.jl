"""
    TabularWorldModel

A Bayesian world model for discrete state-action spaces using:
- Dirichlet-Categorical for transition probabilities
- Normal-Gamma conjugate prior for reward distributions

Supports Thompson Sampling via sampling from posteriors.
"""

using Distributions

"""
    NormalGammaPosterior

Conjugate prior for Normal observations with unknown mean and variance.

    μ | σ² ~ Normal(μ₀, σ²/κ)
    σ²     ~ InvGamma(α, β)

Posterior predictive is a t-distribution:
    r ~ t_{2α}(μ, β(κ+1)/(ακ))
"""
struct NormalGammaPosterior
    κ::Float64   # pseudo-observations for mean (controls mean certainty)
    μ::Float64   # posterior mean location
    α::Float64   # shape: half-observations for variance
    β::Float64   # rate: scaled sum of squared deviations
end

"""
    TabularWorldModel

Maintains Bayesian beliefs over transition and reward dynamics.

Optionally maintains feature-level reward posteriors alongside the tabular model.
When a `feature_extractor` is provided, reward predictions combine tabular and
feature posteriors via precision-weighted averaging — features with more observations
(higher κ) contribute more. This enables generalisation: seeing reward for "eat" in
Kitchen informs "eat" in Pantry via shared (:action_type, :interact) features.
"""
mutable struct TabularWorldModel <: WorldModel
    # Transition model: (state, action) → Dirichlet over next states
    # Stored as counts: transition_counts[(s,a)][s'] = count
    transition_counts::Dict{Tuple{Any,Any}, Dict{Any, Float64}}

    # Reward model: (state, action) → Normal-Gamma posterior
    reward_posterior::Dict{Tuple{Any,Any}, NormalGammaPosterior}

    # Feature-level reward posteriors: (feature_key, action) → NormalGammaPosterior
    # Enables generalisation across states sharing features (location, action type, etc.)
    feature_reward_posterior::Dict{Tuple{Any,Any}, NormalGammaPosterior}

    # Feature extractor: abstract_state → Vector of feature keys
    # nothing means pure tabular (existing behaviour)
    feature_extractor::Union{Nothing, Function}

    # Prior hyperparameters
    transition_prior::Float64  # Dirichlet concentration (pseudocount per state)
    reward_prior::NormalGammaPosterior  # Prior for unseen (s,a) pairs

    # Known states (discovered during interaction)
    known_states::Set{Any}
end

"""
    TabularWorldModel(; transition_prior=0.1, reward_prior_mean=0.0, reward_prior_variance=1.0, feature_extractor=nothing)

Create a new tabular world model with specified priors.

The reward prior is a Normal-Gamma with:
- κ₀ = 1.0 (one pseudo-observation worth of mean certainty)
- μ₀ = reward_prior_mean
- α₀ = 1.0 (weakly informative variance prior)
- β₀ = reward_prior_variance (prior scale for variance)

When `feature_extractor` is provided (a function: abstract_state → Vector of feature keys),
the model also maintains feature-level reward posteriors and combines them with tabular
posteriors via precision-weighted averaging in reward_dist and sample_dynamics.
"""
function TabularWorldModel(;
    transition_prior::Float64 = 0.1,
    reward_prior_mean::Float64 = 0.0,
    reward_prior_variance::Float64 = 1.0,
    feature_extractor::Union{Nothing, Function} = nothing
)
    prior = NormalGammaPosterior(1.0, reward_prior_mean, 1.0, reward_prior_variance)
    return TabularWorldModel(
        Dict{Tuple{Any,Any}, Dict{Any, Float64}}(),
        Dict{Tuple{Any,Any}, NormalGammaPosterior}(),
        Dict{Tuple{Any,Any}, NormalGammaPosterior}(),
        feature_extractor,
        transition_prior,
        prior,
        Set{Any}()
    )
end

"""
    action_features(action) → Vector

Extract feature keys from an action string for the factored reward model.
Returns a vector of (feature_type, feature_value) tuples.
"""
function action_features(action)
    a = lowercase(string(action))
    if startswith(a, "go ") || a in ["north","south","east","west","up","down","ne","nw","se","sw","n","s","e","w"]
        return [(:action_type, :movement)]
    elseif startswith(a, "take ") || startswith(a, "get ") || startswith(a, "pick ")
        return [(:action_type, :take)]
    elseif startswith(a, "drop ") || startswith(a, "put ")
        return [(:action_type, :drop)]
    elseif startswith(a, "look") || startswith(a, "examine") || startswith(a, "x ") || a == "x" || startswith(a, "read ")
        return [(:action_type, :examine)]
    elseif startswith(a, "open ") || startswith(a, "close ") || startswith(a, "unlock ")
        return [(:action_type, :manipulate)]
    else
        return [(:action_type, :interact)]
    end
end

"""
    update!(model::TabularWorldModel, s, a, r, s′)

Update the model with an observed transition.

Reward posterior uses Normal-Gamma conjugate update:
    κₙ = κ₀ + 1
    μₙ = (κ₀μ₀ + r) / κₙ
    αₙ = α₀ + 0.5
    βₙ = β₀ + κ₀(r - μ₀)² / (2κₙ)
"""
function update!(model::TabularWorldModel, s, a, r, s′)
    key = (s, a)

    # Update transition counts
    if !haskey(model.transition_counts, key)
        model.transition_counts[key] = Dict{Any, Float64}()
    end
    model.transition_counts[key][s′] = get(model.transition_counts[key], s′, model.transition_prior) + 1.0

    # Update reward posterior (Normal-Gamma conjugate update)
    p = get(model.reward_posterior, key, model.reward_prior)
    κₙ = p.κ + 1.0
    μₙ = (p.κ * p.μ + r) / κₙ
    αₙ = p.α + 0.5
    βₙ = p.β + p.κ * (r - p.μ)^2 / (2.0 * κₙ)
    model.reward_posterior[key] = NormalGammaPosterior(κₙ, μₙ, αₙ, βₙ)

    # Feature-level reward updates
    if !isnothing(model.feature_extractor)
        for fkey in model.feature_extractor(s)
            feature_key = (fkey, a)
            fp = get(model.feature_reward_posterior, feature_key, model.reward_prior)
            fκₙ = fp.κ + 1.0
            fμₙ = (fp.κ * fp.μ + r) / fκₙ
            fαₙ = fp.α + 0.5
            fβₙ = fp.β + fp.κ * (r - fp.μ)^2 / (2.0 * fκₙ)
            model.feature_reward_posterior[feature_key] = NormalGammaPosterior(fκₙ, fμₙ, fαₙ, fβₙ)
        end
        for afkey in action_features(a)
            feature_key = (afkey, :any)
            fp = get(model.feature_reward_posterior, feature_key, model.reward_prior)
            fκₙ = fp.κ + 1.0
            fμₙ = (fp.κ * fp.μ + r) / fκₙ
            fαₙ = fp.α + 0.5
            fβₙ = fp.β + fp.κ * (r - fp.μ)^2 / (2.0 * fκₙ)
            model.feature_reward_posterior[feature_key] = NormalGammaPosterior(fκₙ, fμₙ, fαₙ, fβₙ)
        end
    end

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
    collect_posteriors(model::TabularWorldModel, s, a) → Vector{NormalGammaPosterior}

Collect all relevant posteriors for a (state, action) pair: tabular + features.
"""
function collect_posteriors(model::TabularWorldModel, s, a)
    tab = get(model.reward_posterior, (s, a), model.reward_prior)
    if isnothing(model.feature_extractor)
        return [tab]
    end

    posteriors = NormalGammaPosterior[tab]
    for fkey in model.feature_extractor(s)
        fk = (fkey, a)
        if haskey(model.feature_reward_posterior, fk)
            push!(posteriors, model.feature_reward_posterior[fk])
        end
    end
    for afkey in action_features(a)
        fk = (afkey, :any)
        if haskey(model.feature_reward_posterior, fk)
            push!(posteriors, model.feature_reward_posterior[fk])
        end
    end
    return posteriors
end

"""
    combine_posteriors(posteriors::Vector{NormalGammaPosterior}) → NormalGammaPosterior

Precision-weighted combination of Normal-Gamma posteriors.

κ acts as precision (more observations → higher κ → more weight).
    μ_combined = Σ(κᵢ·μᵢ) / Σ(κᵢ)
    α_combined = mean(αᵢ)
    β_combined = mean(βᵢ)
"""
function combine_posteriors(posteriors::Vector{NormalGammaPosterior})
    κ_total = sum(p.κ for p in posteriors)
    μ_combined = sum(p.κ * p.μ for p in posteriors) / κ_total
    α_combined = sum(p.α for p in posteriors) / length(posteriors)
    β_combined = sum(p.β for p in posteriors) / length(posteriors)
    return NormalGammaPosterior(κ_total, μ_combined, α_combined, β_combined)
end

"""
    reward_dist(model::TabularWorldModel, s, a) → LocationScale{TDist}

Return the posterior predictive distribution over rewards.

The posterior predictive of a Normal-Gamma model is a (scaled, shifted) t-distribution:
    r ~ t_{2α}(μ, √(β(κ+1)/(ακ)))

When a feature_extractor is present, combines tabular and feature-level posteriors
via precision-weighted averaging before computing the predictive distribution.
"""
function reward_dist(model::TabularWorldModel, s, a)
    posteriors = collect_posteriors(model, s, a)
    p = length(posteriors) == 1 ? posteriors[1] : combine_posteriors(posteriors)

    df = 2.0 * p.α
    scale = sqrt(p.β * (p.κ + 1.0) / (p.α * p.κ))

    return LocationScale(p.μ, scale, TDist(df))
end

"""
    sample_dynamics(model::TabularWorldModel) → SampledDynamics

Sample a concrete dynamics model from the posterior for Thompson Sampling.

For each observed (s,a), samples:
- Transition probs from Dirichlet posterior
- Reward from Normal-Gamma posterior: σ² ~ InvGamma(α,β), then r ~ Normal(μ, σ²/κ)

For unobserved (s,a), rewards are lazily sampled from the prior (wide distribution).
"""
function sample_dynamics(model::TabularWorldModel)
    # Sample transition probabilities from Dirichlet posteriors
    sampled_transitions = Dict{Tuple{Any,Any}, Any}()

    for (key, counts) in model.transition_counts
        states = collect(keys(counts))
        alphas = [counts[s] for s in states]
        probs = rand(Dirichlet(alphas))
        sampled_transitions[key] = (states=states, probs=probs)
    end

    # Sample rewards from Normal-Gamma posteriors (combined with features if available)
    sampled_rewards = Dict{Tuple{Any,Any}, Float64}()

    for (key, _) in model.reward_posterior
        s, a = key
        posteriors = collect_posteriors(model, s, a)
        p = length(posteriors) == 1 ? posteriors[1] : combine_posteriors(posteriors)
        # Sample σ² ~ InvGamma(α, β), then μ ~ Normal(μ_post, σ²/κ)
        σ² = rand(InverseGamma(p.α, p.β))
        r = rand(Normal(p.μ, sqrt(σ² / p.κ)))
        sampled_rewards[key] = r
    end

    return SampledDynamics(
        sampled_transitions,
        sampled_rewards,
        model.known_states,
        model.reward_prior
    )
end

"""
    SampledDynamics

A sampled world model for use in planning.

Lazily samples from the prior for unknown (s,a) pairs — different across
Thompson samples (exploration) but consistent within one sample (coherent planning).
"""
mutable struct SampledDynamics
    transitions::Dict{Tuple{Any,Any}, Any}
    rewards::Dict{Tuple{Any,Any}, Float64}
    known_states::Set{Any}
    reward_prior::NormalGammaPosterior
end

"""
    sample_next_state(dynamics::SampledDynamics, s, a) → s′

Sample a next state from the sampled dynamics.

For unknown (s,a), samples once (50% self-loop, 50% random known state) and caches
the result for consistency within this Thompson sample.
"""
function sample_next_state(dynamics::SampledDynamics, s, a)
    key = (s, a)

    if !haskey(dynamics.transitions, key)
        # Unknown transition — sample and cache for consistency
        next = if isempty(dynamics.known_states) || rand() < 0.5
            s  # Self-loop
        else
            rand(collect(dynamics.known_states))
        end
        # Cache as a degenerate categorical so future lookups are consistent
        dynamics.transitions[key] = (states=[next], probs=[1.0])
        return next
    end

    trans = dynamics.transitions[key]
    idx = rand(Categorical(trans.probs))
    return trans.states[idx]
end

"""
    get_reward(dynamics::SampledDynamics, s, a) → Float64

Get the sampled reward for a state-action pair.

For unknown (s,a), lazily samples from the Normal-Gamma prior and caches
the result. This ensures different Thompson samples see different rewards for
untried actions (exploration) while each sample is internally consistent (coherent planning).
"""
function get_reward(dynamics::SampledDynamics, s, a)
    key = (s, a)
    if haskey(dynamics.rewards, key)
        return dynamics.rewards[key]
    end
    # Sample from Normal-Gamma prior and cache for consistency
    p = dynamics.reward_prior
    σ² = rand(InverseGamma(p.α, p.β))
    r = rand(Normal(p.μ, sqrt(σ² / p.κ)))
    dynamics.rewards[key] = r
    return r
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
