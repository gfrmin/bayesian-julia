"""
    hidden_variable_inference.jl (Stage 2: Hidden Variable Inference)

Bayesian inference for hidden state variables that are not directly observed.

Instead of assuming all state is visible (location, inventory), we maintain beliefs
over hidden variables:
- spells_known: Set of magic abilities learned
- object_states: State of objects (lit, locked, activated, etc.)
- knowledge_gained: Facts discovered

These are inferred from observation text via LLM likelihood queries.

Mathematical basis:
    P(hidden_var | observation_text) ∝ P(text | hidden_var) × P(hidden_var)

where P(text | hidden_var) is queried from the LLM sensor.

This solves the "look nitfol" problem:
- Episode 1: agent doesn't know detect_magic
  State: (library, {}, spells={})
  "look nitfol" gives no reward
- Episode 2: agent learned detect_magic
  State: (library, {}, spells={detect_magic})  ← Different state!
  "look nitfol" gives reward (hidden variable changed state space)
"""

"""
    infer_hidden_variables!(state::MinimalState, observation_text::String, sensor::LLMSensor)

Update hidden variables in state based on observation text via Bayesian inference.

Uses LLM as a sensor to query likelihood P(text | hidden_var_value).
"""
function infer_hidden_variables!(
    state::MinimalState,
    observation_text::String,
    sensor::LLMSensor
)
    # Query candidates for hidden variables
    candidates = [
        ("spell_learned", "Did the agent learn a spell or magical ability?"),
        ("object_activated", "Did an object get activated or change state (e.g., light up, unlock)?"),
        ("knowledge_gained", "Did the agent gain new knowledge or discover something?"),
    ]

    for (var_type, question) in candidates
        try
            # Query LLM: what is the likelihood this text indicates the hidden variable?
            response = query(sensor, state, question)

            if response isa Bool && response
                # Positive evidence for this hidden variable
                if var_type == "spell_learned"
                    # Extract spell name from text if possible
                    spell_name = extract_spell_name(observation_text)
                    if !isempty(spell_name)
                        push!(state.spells_known, spell_name)
                    end
                elseif var_type == "object_activated"
                    # Extract object state changes
                    updates = extract_object_states(observation_text)
                    merge!(state.object_states, updates)
                elseif var_type == "knowledge_gained"
                    # Extract facts discovered
                    facts = extract_knowledge(observation_text)
                    union!(state.knowledge_gained, facts)
                end
            end
        catch e
            # LLM query may fail; continue gracefully
            @debug "Hidden variable inference failed" var_type=var_type error=e
        end
    end

    return state
end

"""
    extract_spell_name(text::String) → String

Attempt to extract spell name from observation text.
E.g., "You feel the power of detect magic" → "detect_magic"
"""
function extract_spell_name(text::String)::String
    # Simple heuristic: look for common spell keywords
    spell_keywords = [
        ("detect magic" => "detect_magic"),
        ("detect_magic" => "detect_magic"),
        ("magic missile" => "magic_missile"),
        ("fireball" => "fireball"),
        ("light" => "light"),
        ("invisibility" => "invisibility"),
        ("teleport" => "teleport"),
        ("summon" => "summon"),
    ]

    text_lower = lowercase(text)
    for (keyword, spell_name) in spell_keywords
        if contains(text_lower, keyword)
            return spell_name
        end
    end

    return ""
end

"""
    extract_object_states(text::String) → Dict{String, String}

Extract object state changes from observation text.
E.g., "The lantern glows brightly" → Dict("lantern" => "lit")
"""
function extract_object_states(text::String)::Dict{String, String}
    states = Dict{String, String}()

    text_lower = lowercase(text)

    # Light states
    if contains(text_lower, "lantern") && (contains(text_lower, "lit") || contains(text_lower, "glow") || contains(text_lower, "bright"))
        states["lantern"] = "lit"
    end
    if contains(text_lower, "torch") && (contains(text_lower, "lit") || contains(text_lower, "glow") || contains(text_lower, "bright"))
        states["torch"] = "lit"
    end

    # Lock states
    if contains(text_lower, "door") && (contains(text_lower, "unlock") || contains(text_lower, "open"))
        states["door"] = "unlocked"
    end
    if contains(text_lower, "chest") && (contains(text_lower, "unlock") || contains(text_lower, "open"))
        states["chest"] = "unlocked"
    end

    # Activation states
    if contains(text_lower, "portal") && contains(text_lower, "open")
        states["portal"] = "open"
    end
    if contains(text_lower, "bridge") && contains(text_lower, "appear")
        states["bridge"] = "visible"
    end

    return states
end

"""
    extract_knowledge(text::String) → Set{String}

Extract discovered facts from observation text.
"""
function extract_knowledge(text::String)::Set{String}
    knowledge = Set{String}()

    text_lower = lowercase(text)

    # Common discoveries in IF games
    if contains(text_lower, "secret") || contains(text_lower, "hidden")
        push!(knowledge, "secret_location")
    end
    if contains(text_lower, "passage")
        push!(knowledge, "passage_discovered")
    end
    if contains(text_lower, "spell")
        push!(knowledge, "spell_exists")
    end
    if contains(text_lower, "ancient")
        push!(knowledge, "ancient_artifact")
    end
    if contains(text_lower, "magic")
        push!(knowledge, "magic_exists")
    end

    return knowledge
end

export infer_hidden_variables!, extract_spell_name, extract_object_states, extract_knowledge
