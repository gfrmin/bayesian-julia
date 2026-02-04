"""
    StateBelief (Stage 1: MVBN)

Maintains factored belief distribution over state variables:
- P(location | history)     : DirichletCategorical over observed locations
- P(inventory | history)    : Set of DirichletCategorical, one per object

Mathematical basis:
    P(s | history) = P(location | history) × P(inventory | history)
                   = P(location | history) × ∏_obj P(obj ∈ inventory | history)

Supports:
1. Bayesian updates from observations
2. Sampling for Thompson Sampling (sample s ~ P(s | history))
3. Entropy computation (for exploration)
4. Online learning of state space
"""

mutable struct StateBelief
    # Location belief: location → (prior_α, observed_counts)
    location_belief::DirichletCategorical

    # Inventory belief: object_name → DirichletCategorical(domain={in, out})
    # Each object has binary indicator: true = in inventory, false = not in inventory
    inventory_beliefs::Dict{String, DirichletCategorical}

    # History of observed states (for discovering new variables)
    history::Vector{MinimalState}

    # Known objects discovered so far
    known_objects::Set{String}

    function StateBelief(initial_locations::Vector{String})
        location_belief = DirichletCategorical(initial_locations, 0.1)
        inventory_beliefs = Dict{String, DirichletCategorical}()
        history = MinimalState[]
        known_objects = Set{String}()

        new(location_belief, inventory_beliefs, history, known_objects)
    end
end

"""
    StateBelief() → StateBelief

Create a fresh belief with common IF game locations.
"""
function StateBelief()
    # Start with common locations; will be extended as discovered
    common_locations = ["Kitchen", "Forest", "Room", "House", "Ground", "outside"]
    return StateBelief(common_locations)
end

"""
    add_object!(belief::StateBelief, object_name::String)

Register a new object to track in inventory.
Creates a binary inventory belief for the object.
"""
function add_object!(belief::StateBelief, object_name::String)
    if !haskey(belief.inventory_beliefs, object_name)
        # Binary: object is in inventory or not
        domain = [true, false]
        alpha = 0.5  # Weak prior: equally likely in or out
        belief.inventory_beliefs[object_name] = DirichletCategorical(domain, alpha)
        push!(belief.known_objects, object_name)
    end
end

"""
    update_from_state!(belief::StateBelief, state::MinimalState)

Update beliefs given an observed state.
"""
function update_from_state!(belief::StateBelief, state::MinimalState)
    # Record location
    update!(belief.location_belief, state.location)

    # Record inventory for each known object
    for obj in belief.known_objects
        in_inventory = obj ∈ state.inventory
        update!(belief.inventory_beliefs[obj], in_inventory)
    end

    # Register any new objects observed
    for obj in state.inventory
        add_object!(belief, obj)
    end

    # Store in history for later analysis
    push!(belief.history, state)
end

"""
    sample_state(belief::StateBelief) → MinimalState

Sample a state from the posterior belief distribution.
Uses Thompson Sampling: sample each variable from posterior, combine.
"""
function sample_state(belief::StateBelief)::MinimalState
    # Sample location
    location = rand(belief.location_belief)

    # Sample inventory: for each object, independently sample in/out
    inventory = Set{String}()
    for obj in belief.known_objects
        in_inventory = rand(belief.inventory_beliefs[obj])
        if in_inventory
            push!(inventory, obj)
        end
    end

    return MinimalState(location, inventory)
end

"""
    predict_state(belief::StateBelief) → MinimalState

Return the mode (MAP estimate) of the posterior belief.
"""
function predict_state(belief::StateBelief)::MinimalState
    location = mode(belief.location_belief)

    inventory = Set{String}()
    for obj in belief.known_objects
        in_inventory = mode(belief.inventory_beliefs[obj])
        if in_inventory
            push!(inventory, obj)
        end
    end

    return MinimalState(location, inventory)
end

"""
    entropy(belief::StateBelief) → Float64

Compute Shannon entropy of posterior belief.
H[s] = H[location] + Σ_obj H[obj ∈ inventory]
"""
function entropy(belief::StateBelief)::Float64
    h = entropy(belief.location_belief)

    for obj in belief.known_objects
        h += entropy(belief.inventory_beliefs[obj])
    end

    return h
end

"""
    posterior_prob(belief::StateBelief, state::MinimalState) → Float64

Compute P(state | history) under factored model.
P(s) = P(location) × ∏_obj P(obj ∈ inventory)
"""
function posterior_prob(belief::StateBelief, state::MinimalState)::Float64
    probs = predict(belief.location_belief)
    location_idx = findfirst(l -> l == state.location, belief.location_belief.domain)
    if isnothing(location_idx)
        return 0.0
    end
    p = probs[location_idx]

    for obj in belief.known_objects
        in_inventory = obj ∈ state.inventory
        obj_probs = predict(belief.inventory_beliefs[obj])
        # domain = [true, false], find idx for whether obj is in inventory
        idx = findfirst(v -> v == in_inventory, belief.inventory_beliefs[obj].domain)
        p *= obj_probs[idx]
    end

    return p
end

"""
    loglikelihood(belief::StateBelief) → Float64

Sum of log marginal likelihoods for all belief variables.
Used for model comparison (e.g., variable discovery, structure learning).
"""
function loglikelihood(belief::StateBelief)::Float64
    ll = loglikelihood(belief.location_belief)
    for obj in belief.known_objects
        ll += loglikelihood(belief.inventory_beliefs[obj])
    end
    return ll
end

"""
    reset!(belief::StateBelief)

Clear all observations, reset to prior.
"""
function reset!(belief::StateBelief)
    reset!(belief.location_belief)
    for obj in belief.known_objects
        reset!(belief.inventory_beliefs[obj])
    end
    empty!(belief.history)
end

"""
    copy(belief::StateBelief) → StateBelief

Deep copy of belief state.
"""
function Base.copy(belief::StateBelief)
    new_belief = StateBelief(copy(belief.location_belief.domain))
    new_belief.location_belief = copy(belief.location_belief)
    new_belief.inventory_beliefs = Dict(k => copy(v) for (k, v) in belief.inventory_beliefs)
    new_belief.history = copy(belief.history)
    new_belief.known_objects = copy(belief.known_objects)
    return new_belief
end

export StateBelief, add_object!, update_from_state!, sample_state, predict_state, entropy, posterior_prob, loglikelihood
