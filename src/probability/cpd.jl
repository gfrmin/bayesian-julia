"""
    ConditionalProbabilityDistribution (CPD)

Represents P(V | parents) for a discrete variable V.
Uses Dirichlet-Categorical conjugate pair for efficient Bayesian updates.

Mathematical foundations:
- Prior: θ ~ Dirichlet(α)
- Likelihood: observations ~ Categorical(θ)
- Posterior: θ ~ Dirichlet(α + counts)
- Predictive: P(V=v | data) = (α_v + count_v) / (Σα + n)
"""

"""
    DirichletCategorical

Dirichlet-Categorical conjugate pair: maintains Dirichlet posterior over
categorical parameters θ = [θ₁, θ₂, ..., θₖ] where Σθᵢ = 1.

Fields:
- alpha::Vector{Float64}       # Dirichlet concentration parameters α
- counts::Vector{Int}          # Empirical counts from observations
- domain::Vector              # Possible values for this variable
"""
mutable struct DirichletCategorical
    domain::Vector              # Possible values: [v₁, v₂, ..., vₖ]
    alpha::Vector{Float64}      # Prior: Dirichlet(α)
    counts::Vector{Int}         # Observed counts: count[i] = #{times V=domain[i]}

    function DirichletCategorical(domain::Vector, alpha::Vector{Float64})
        @assert length(domain) == length(alpha) "domain and alpha must have same length"
        @assert all(alpha .> 0) "alpha must be positive (Dirichlet support)"

        counts = zeros(Int, length(domain))
        new(domain, alpha, counts)
    end
end

"""
    DirichletCategorical(domain, alpha_scalar)

Constructor: uniform Dirichlet prior with concentration alpha_scalar.
"""
function DirichletCategorical(domain::Vector, alpha_scalar::Float64)
    alpha = fill(alpha_scalar, length(domain))
    DirichletCategorical(domain, alpha)
end

"""
    update!(cpd::DirichletCategorical, observation)

Update posterior with an observation.
Finds value in domain, increments corresponding count.
"""
function update!(cpd::DirichletCategorical, observation)
    idx = findfirst(v -> v == observation, cpd.domain)
    if isnothing(idx)
        @warn "Observation $observation not in domain $(cpd.domain)"
        return
    end
    cpd.counts[idx] += 1
end

"""
    predict(cpd::DirichletCategorical) → Vector{Float64}

Compute posterior predictive distribution P(V=v | data).
Returns normalized probabilities for each value in domain.

Formula: P(V=vᵢ | data) = (α_i + count_i) / (Σα + Σcount)
"""
function predict(cpd::DirichletCategorical)::Vector{Float64}
    posterior_alpha = cpd.alpha + cpd.counts
    return posterior_alpha / sum(posterior_alpha)
end

"""
    sample(cpd::DirichletCategorical) → value

Sample from posterior predictive distribution.
1. Sample θ ~ Dirichlet(α + counts)
2. Sample V ~ Categorical(θ)
"""
function Random.rand(cpd::DirichletCategorical)
    theta = rand(Distributions.Dirichlet(cpd.alpha + cpd.counts))
    value_idx = rand(Distributions.Categorical(theta))
    return cpd.domain[value_idx]
end

"""
    loglikelihood(cpd::DirichletCategorical, observations::Vector) → Float64

Compute log marginal likelihood log P(data | prior).
Used for model comparison (BIC scoring).

Formula: log P(data | V) = log ∫ P(data | θ) P(θ | α) dθ
                         = Σᵢ [log Γ(αᵢ + countᵢ) - log Γ(αᵢ)]
                           + log Γ(Σα) - log Γ(Σα + n)
"""
function loglikelihood(cpd::DirichletCategorical)::Float64
    alpha_sum = sum(cpd.alpha)
    count_sum = sum(cpd.counts)

    # Normalization constant
    log_z = loggamma(alpha_sum) - loggamma(alpha_sum + count_sum)

    # Per-value term
    log_ll = 0.0
    for i in eachindex(cpd.domain)
        log_ll += loggamma(cpd.alpha[i] + cpd.counts[i]) - loggamma(cpd.alpha[i])
    end

    return log_z + log_ll
end

"""
    entropy(cpd::DirichletCategorical) → Float64

Compute Shannon entropy of posterior predictive distribution.
H[V | data] = -Σᵢ P(V=vᵢ) log P(V=vᵢ)

High entropy → uncertain predictions. Drives exploration in planning.
"""
function entropy(cpd::DirichletCategorical)::Float64
    probs = predict(cpd)
    return -sum(probs .* log.(probs .+ 1e-10))
end

"""
    expected_entropy(cpd::DirichletCategorical) → Float64

Expected entropy of θ over the posterior distribution.
E[H[θ]] = E[-Σᵢ θᵢ log θᵢ]

Used to compute information gain and exploration bonus.
"""
function expected_entropy(cpd::DirichletCategorical)::Float64
    alpha = cpd.alpha + cpd.counts
    alpha_sum = sum(alpha)

    # Dirichlet expected entropy
    h = loggamma(alpha_sum) - sum(loggamma.(alpha))
    h += sum((alpha_sum .- alpha) .* digamma.(alpha))
    h /= alpha_sum

    return h
end

"""
    mode(cpd::DirichletCategorical) → value

Return the mode (maximum a posteriori estimate) of posterior.
"""
function mode(cpd::DirichletCategorical)
    posterior_alpha = cpd.alpha + cpd.counts
    max_idx = argmax(posterior_alpha)
    return cpd.domain[max_idx]
end

"""
    copy(cpd::DirichletCategorical) → DirichletCategorical

Create a deep copy of the CPD.
"""
function Base.copy(cpd::DirichletCategorical)
    new_cpd = DirichletCategorical(copy(cpd.domain), copy(cpd.alpha))
    new_cpd.counts = copy(cpd.counts)
    return new_cpd
end

"""
    reset!(cpd::DirichletCategorical)

Reset to prior (clear all observations).
"""
function reset!(cpd::DirichletCategorical)
    fill!(cpd.counts, 0)
end

export DirichletCategorical, update!, predict, entropy, expected_entropy, mode
