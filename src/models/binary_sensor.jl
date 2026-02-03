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
    query(sensor::BinarySensor, state, question) → Bool

Query the sensor and return its answer.
"""
function query(sensor::BinarySensor, state, question)
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
    query(sensor::LLMSensor, state, question::String) → Bool

Query the LLM and parse the response as yes/no.
"""
function query(sensor::LLMSensor, state, question::String)
    sensor.n_queries += 1
    
    # Format prompt
    prompt = replace(sensor.prompt_template, "{question}" => question)
    
    # Add state context if available
    if !isnothing(state)
        prompt = "Context: $state\n\n$prompt"
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
