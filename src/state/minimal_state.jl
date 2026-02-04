"""
    MinimalState (Stage 1: MVBN)

Factored state representation for Enchanter:
- location: string identifier (e.g., "Kitchen", "Forest")
- inventory: Set of strings (items agent is carrying)

Replaces opaque hash-based state IDs with explicit factorization.
Enables reasoning about state variables independently.

Mathematical representation:
    s = (location, inventory)
    P(s | history) = P(location | history) × P(inventory | history)
"""

struct MinimalState
    location::String
    inventory::Set{String}

    function MinimalState(location::String, inventory::Set{String}=Set{String}())
        return new(location, inventory)
    end
end

"""
    MinimalState(location_str, inventory_str)

Convenience constructor from string representations.
inventory_str format: "book,lantern,key" or empty string "".
"""
function MinimalState(location::String, inventory_str::String)
    if isempty(inventory_str)
        inv = Set{String}()
    else
        inv = Set(strip.(split(inventory_str, ",")))
    end
    return MinimalState(location, inv)
end

"""
    extract_minimal_state(observation) → MinimalState

Extract MinimalState from Jericho observation (NamedTuple).
"""
function extract_minimal_state(obs)::MinimalState
    location = hasproperty(obs, :location) ? obs.location : "unknown"

    inventory_set = Set{String}()
    if hasproperty(obs, :inventory) && obs.inventory isa String
        if !isempty(obs.inventory)
            inv_items = strip.(split(obs.inventory, ","))
            inventory_set = Set(inv_items)
        end
    elseif hasproperty(obs, :inventory) && obs.inventory isa Vector
        inventory_set = Set(obs.inventory)
    end

    return MinimalState(location, inventory_set)
end

"""
    Base.:(==)(s1::MinimalState, s2::MinimalState) → Bool

Equality comparison for states.
"""
function Base.:(==)(s1::MinimalState, s2::MinimalState)::Bool
    return s1.location == s2.location && s1.inventory == s2.inventory
end

"""
    Base.hash(s::MinimalState) → UInt

Hash for use in dictionaries/sets.
"""
function Base.hash(s::MinimalState)::UInt
    return hash((s.location, sort(collect(s.inventory))))
end

"""
    Base.show(io::IO, s::MinimalState)

Pretty printing.
"""
function Base.show(io::IO, s::MinimalState)
    inv_str = isempty(s.inventory) ? "∅" : join(sort(collect(s.inventory)), ",")
    print(io, "MinimalState($(s.location), {$inv_str})")
end

export MinimalState, extract_minimal_state
