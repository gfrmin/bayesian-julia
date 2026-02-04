"""
    BinarySensor

A yes/no sensor with learned reliability (TPR and FPR).
Uses Beta posteriors for both true positive rate and false positive rate.
"""

using Distributions

"""
    BinarySensor

A sensor that provides binary (yes/no) answers with learnable reliability.
"""
mutable struct BinarySensor <: Sensor
    name::String
    
    # TPR: P(yes | true) ~ Beta(tp_α, tp_β)
    tp_α::Float64
    tp_β::Float64
    
    # FPR: P(yes | false) ~ Beta(fp_α, fp_β)
    fp_α::Float64
    fp_β::Float64
    
    # Query function: (state, question) → Bool
    query_fn::Function
    
    # Statistics
    n_queries::Int
    n_correct::Int
end

"""
    BinarySensor(name, query_fn; tp_prior=(2,1), fp_prior=(1,2))

Create a binary sensor with specified query function and priors.

The default priors encode a weak belief that the sensor is somewhat reliable:
- TPR prior: Beta(2,1) → mean 0.67, believes sensor usually detects true positives
- FPR prior: Beta(1,2) → mean 0.33, believes sensor usually avoids false positives
"""
function BinarySensor(
    name::String,
    query_fn::Function;
    tp_prior::Tuple{Float64,Float64} = (2.0, 1.0),
    fp_prior::Tuple{Float64,Float64} = (1.0, 2.0)
)
    return BinarySensor(
        name,
        tp_prior[1], tp_prior[2],
        fp_prior[1], fp_prior[2],
        query_fn,
        0, 0
    )
end

"""
    query(sensor::BinarySensor, state, question; action_history=nothing) → Bool

Query the sensor and return its answer.
"""
function query(sensor::BinarySensor, state, question; action_history=nothing)
    sensor.n_queries += 1
    return sensor.query_fn(state, question)
end

"""
    tpr(sensor::BinarySensor) → Float64

Return the expected true positive rate: E[P(yes | true)].
"""
function tpr(sensor::BinarySensor)
    return sensor.tp_α / (sensor.tp_α + sensor.tp_β)
end

"""
    fpr(sensor::BinarySensor) → Float64

Return the expected false positive rate: E[P(yes | false)].
"""
function fpr(sensor::BinarySensor)
    return sensor.fp_α / (sensor.fp_α + sensor.fp_β)
end

"""
    tpr_dist(sensor::BinarySensor) → Beta

Return the full posterior distribution over TPR.
"""
function tpr_dist(sensor::BinarySensor)
    return Beta(sensor.tp_α, sensor.tp_β)
end

"""
    fpr_dist(sensor::BinarySensor) → Beta

Return the full posterior distribution over FPR.
"""
function fpr_dist(sensor::BinarySensor)
    return Beta(sensor.fp_α, sensor.fp_β)
end

"""
    update_reliability!(sensor::BinarySensor, said_yes::Bool, actual::Bool)

Update the sensor's reliability estimates from ground truth.

Arguments:
- said_yes: What the sensor predicted
- actual: The ground truth
"""
function update_reliability!(sensor::BinarySensor, said_yes::Bool, actual::Bool)
    if actual
        # Ground truth was positive
        if said_yes
            sensor.tp_α += 1  # True positive
            sensor.n_correct += 1
        else
            sensor.tp_β += 1  # False negative
        end
    else
        # Ground truth was negative
        if said_yes
            sensor.fp_α += 1  # False positive
        else
            sensor.fp_β += 1  # True negative
            sensor.n_correct += 1
        end
    end
end

"""
    posterior(sensor::BinarySensor, prior::Float64, answer::Bool) → Float64

Compute the posterior probability that the proposition is true,
given the prior and the sensor's answer.

Uses Bayes' rule: P(true | answer) = P(answer | true) P(true) / P(answer)
"""
function posterior(sensor::BinarySensor, prior::Float64, answer::Bool)
    t = tpr(sensor)
    f = fpr(sensor)
    
    if answer  # Sensor said "yes"
        numerator = t * prior
        denominator = t * prior + f * (1 - prior)
    else  # Sensor said "no"
        numerator = (1 - t) * prior
        denominator = (1 - t) * prior + (1 - f) * (1 - prior)
    end
    
    return denominator > 0 ? numerator / denominator : prior
end

"""
    accuracy(sensor::BinarySensor) → Float64

Return the empirical accuracy of the sensor.
"""
function accuracy(sensor::BinarySensor)
    return sensor.n_queries > 0 ? sensor.n_correct / sensor.n_queries : 0.5
end

"""
    reliability_summary(sensor::BinarySensor) → String

Return a human-readable summary of the sensor's reliability.
"""
function reliability_summary(sensor::BinarySensor)
    return """
    Sensor: $(sensor.name)
      TPR: $(round(tpr(sensor), digits=3)) [Beta($(sensor.tp_α), $(sensor.tp_β))]
      FPR: $(round(fpr(sensor), digits=3)) [Beta($(sensor.fp_α), $(sensor.fp_β))]
      Queries: $(sensor.n_queries)
      Accuracy: $(round(accuracy(sensor), digits=3))
    """
end

# ============================================================================
# LLM SENSOR (specialised for language model queries)
# ============================================================================

"""
    LLMSensor

A binary sensor backed by a large language model.
Wraps an LLM client and converts questions to yes/no queries.
"""
mutable struct LLMSensor <: Sensor
    name::String
    
    # Reliability tracking (same as BinarySensor)
    tp_α::Float64
    tp_β::Float64
    fp_α::Float64
    fp_β::Float64
    
    # LLM interface
    llm_client::Any  # Duck-typed: must have query(client, prompt) → String
    prompt_template::String
    
    # Statistics
    n_queries::Int
    n_correct::Int
end

"""
    LLMSensor(name, llm_client; prompt_template="...", tp_prior=(2,1), fp_prior=(1,2))

Create an LLM-backed binary sensor.
"""
function LLMSensor(
    name::String,
    llm_client;
    prompt_template::String = "Answer with only 'yes' or 'no': {question}",
    tp_prior::Tuple{Float64,Float64} = (2.0, 1.0),
    fp_prior::Tuple{Float64,Float64} = (1.0, 2.0)
)
    return LLMSensor(
        name,
        tp_prior[1], tp_prior[2],
        fp_prior[1], fp_prior[2],
        llm_client,
        prompt_template,
        0, 0
    )
end

"""
    format_observation_for_llm(obs) → String

Format an observation for use as LLM context.
Handles NamedTuples with :text/:location/:inventory/:score fields (e.g. Jericho),
and falls back to string() for other types.
"""
function format_observation_for_llm(obs)
    if obs isa NamedTuple
        parts = String[]
        hasproperty(obs, :location) && push!(parts, "Location: $(obs.location)")
        hasproperty(obs, :text) && push!(parts, "Observation: $(obs.text)")
        hasproperty(obs, :inventory) && push!(parts, "Inventory: $(obs.inventory)")
        hasproperty(obs, :score) && push!(parts, "Score: $(obs.score)")
        if !isempty(parts)
            return join(parts, "\n")
        end
    end
    return string(obs)
end

"""
    query(sensor::LLMSensor, state, question::String; action_history=nothing) → Bool

Query the LLM and parse the response as yes/no.

When `action_history` is provided (a vector of recent action strings), it is
included in the prompt so the LLM can avoid recommending already-tried actions.
"""
function query(sensor::LLMSensor, state, question::String; action_history=nothing)
    sensor.n_queries += 1

    # Format prompt
    prompt = replace(sensor.prompt_template, "{question}" => question)

    # Add state context if available, formatted for LLM readability
    if !isnothing(state)
        context = format_observation_for_llm(state)
        prompt = "Context:\n$context\n\n$prompt"
    end

    # Add action history if available
    if !isnothing(action_history) && !isempty(action_history)
        history_str = join(string.(action_history), ", ")
        prompt = "$prompt\n\nRecent actions tried: $history_str"
    end

    # Query LLM
    response = lowercase(strip(sensor.llm_client.query(prompt)))

    # Parse response
    if startswith(response, "yes")
        return true
    elseif startswith(response, "no")
        return false
    else
        # Ambiguous — try harder
        if occursin("yes", response) && !occursin("no", response)
            return true
        elseif occursin("no", response) && !occursin("yes", response)
            return false
        else
            # Truly ambiguous — default to no (conservative)
            return false
        end
    end
end

# Inherit reliability methods from BinarySensor
tpr(s::LLMSensor) = s.tp_α / (s.tp_α + s.tp_β)
fpr(s::LLMSensor) = s.fp_α / (s.fp_α + s.fp_β)

function update_reliability!(sensor::LLMSensor, said_yes::Bool, actual::Bool)
    if actual
        if said_yes
            sensor.tp_α += 1
            sensor.n_correct += 1
        else
            sensor.tp_β += 1
        end
    else
        if said_yes
            sensor.fp_α += 1
        else
            sensor.fp_β += 1
            sensor.n_correct += 1
        end
    end
end

# ============================================================================
# RANKING QUERY (ask LLM to pick best action from a list)
# ============================================================================

"""
    query_ranking(sensor::LLMSensor, observation, actions) → Union{action, Nothing}

Ask the LLM to select the most promising action from the full action list.

One ranking query replaces N binary queries, giving more information per LLM call.
Returns the matched action, or nothing if the response couldn't be parsed.
"""
function query_ranking(sensor::LLMSensor, observation, actions)
    sensor.n_queries += 1

    action_list = join(string.(actions), "\n")
    context = format_observation_for_llm(observation)

    prompt = """Context:
$context

Which of these actions is most likely to make progress toward winning?
$action_list

Reply with ONLY the action text, nothing else."""

    response = lowercase(strip(sensor.llm_client.query(prompt)))

    # Match response to an action (exact match first, then substring)
    for a in actions
        if lowercase(string(a)) == response
            return a
        end
    end
    for a in actions
        if occursin(lowercase(string(a)), response) || occursin(response, lowercase(string(a)))
            return a
        end
    end

    return nothing
end

"""
    update_beliefs_from_ranking!(sensor, actions, selected, action_beliefs) → Dict

Apply Bayesian update from a ranking query result.

The selected action gets a positive observation (posterior with answer=true).
All non-selected actions get a negative observation (posterior with answer=false).
Uses the existing posterior() function — same math, just applied to all actions
from a single LLM call.
"""
function update_beliefs_from_ranking!(sensor, actions, selected, action_beliefs::Dict)
    for a in actions
        prior = get(action_beliefs, a, 1.0 / length(actions))
        if a == selected
            action_beliefs[a] = posterior(sensor, prior, true)
        else
            action_beliefs[a] = posterior(sensor, prior, false)
        end
    end
    return action_beliefs
end

# ============================================================================
# NULL OUTCOME DETECTION
# ============================================================================

"""
    is_null_outcome(obs_before, obs_after) → Bool

Detect whether an action produced no change — same command + same observation
text means the transition is S→S with reward 0.

This is a world model fact: P(obs_unchanged | action_helpful) ≈ 0. Used to
provide negative ground truth to sensors without waiting for sparse rewards.
"""
function is_null_outcome(obs_before, obs_after)
    text_before = if obs_before isa NamedTuple && hasproperty(obs_before, :text)
        obs_before.text
    else
        string(obs_before)
    end
    text_after = if obs_after isa NamedTuple && hasproperty(obs_after, :text)
        obs_after.text
    else
        string(obs_after)
    end
    return strip(text_before) == strip(text_after)
end
