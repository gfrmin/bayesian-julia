"""
    ThompsonMCTS

Monte Carlo Tree Search with Thompson Sampling for Bayes-Adaptive planning.

The key insight: instead of UCB for exploration, we sample a world model from
the posterior and plan optimally under that sample. This naturally balances
exploration and exploitation.
"""

using Random

"""
    MCTSNode

A node in the MCTS search tree.
"""
mutable struct MCTSNode
    state::Any
    parent::Union{Nothing, MCTSNode}
    parent_action::Any
    children::Dict{Any, MCTSNode}  # action → child
    
    # Statistics
    visit_count::Int
    value_sum::Float64
    
    # Is this a terminal state?
    is_terminal::Bool
end

"""
    MCTSNode(state; parent=nothing, parent_action=nothing, is_terminal=false)

Create a new MCTS node.
"""
function MCTSNode(
    state;
    parent::Union{Nothing, MCTSNode} = nothing,
    parent_action = nothing,
    is_terminal::Bool = false
)
    return MCTSNode(state, parent, parent_action, Dict{Any, MCTSNode}(), 0, 0.0, is_terminal)
end

"""
    value(node::MCTSNode) → Float64

Return the mean value of this node.
"""
function value(node::MCTSNode)
    return node.visit_count > 0 ? node.value_sum / node.visit_count : 0.0
end

"""
    is_expanded(node::MCTSNode, actions) → Bool

Check if the node has been fully expanded.
"""
function is_expanded(node::MCTSNode, actions)
    return length(node.children) == length(actions)
end

"""
    ThompsonMCTS

Thompson Sampling MCTS planner.
"""
struct ThompsonMCTS <: Planner
    iterations::Int
    depth::Int
    discount::Float64
    ucb_c::Float64  # For tree policy within a single Thompson sample
    
    # Action prior function: (state, actions) → Dict{action, prior}
    # Used to bias exploration toward promising actions
    action_prior::Union{Nothing, Function}
end

"""
    ThompsonMCTS(; iterations=100, depth=10, discount=0.99, ucb_c=2.0, action_prior=nothing)

Create a Thompson MCTS planner.
"""
function ThompsonMCTS(;
    iterations::Int = 100,
    depth::Int = 10,
    discount::Float64 = 0.99,
    ucb_c::Float64 = 2.0,
    action_prior::Union{Nothing, Function} = nothing
)
    return ThompsonMCTS(iterations, depth, discount, ucb_c, action_prior)
end

"""
    plan(planner::ThompsonMCTS, state, model::WorldModel, actions) → action

Plan using Thompson Sampling MCTS and return the best action.
"""
function plan(planner::ThompsonMCTS, state, model::WorldModel, actions)
    if isempty(actions)
        error("No actions available")
    end
    
    if length(actions) == 1
        return first(actions)
    end
    
    # Create root node
    root = MCTSNode(state)
    
    # Get action priors if available
    priors = if !isnothing(planner.action_prior)
        planner.action_prior(state, actions)
    else
        Dict(a => 1.0 / length(actions) for a in actions)
    end
    
    # Run MCTS iterations
    for _ in 1:planner.iterations
        # Thompson sample: draw a world model from the posterior
        sampled_dynamics = sample_dynamics(model)
        
        # Run one simulation with the sampled dynamics
        simulate!(planner, root, sampled_dynamics, actions, priors, planner.depth)
    end
    
    # Select best action by visit count (robust to noise)
    best_action = nothing
    best_visits = -1
    
    for (action, child) in root.children
        if child.visit_count > best_visits
            best_visits = child.visit_count
            best_action = action
        end
    end
    
    return best_action
end

"""
    simulate!(planner, node, dynamics, actions, priors, depth) → value

Run one simulation from a node using the sampled dynamics.
"""
function simulate!(
    planner::ThompsonMCTS,
    node::MCTSNode,
    dynamics,
    actions,
    priors::Dict,
    depth::Int
)
    if depth == 0 || node.is_terminal
        return 0.0
    end
    
    # Selection: if fully expanded, use UCB to select child
    if is_expanded(node, actions)
        action, child = select_ucb(planner, node, actions)
        value = get_reward(dynamics, node.state, action) +
                planner.discount * simulate!(planner, child, dynamics, actions, priors, depth - 1)
    else
        # Expansion: add a new child
        action = select_unexpanded(node, actions, priors)
        next_state = sample_next_state(dynamics, node.state, action)
        child = MCTSNode(next_state; parent=node, parent_action=action)
        node.children[action] = child
        
        # Rollout from new child
        value = get_reward(dynamics, node.state, action) +
                planner.discount * rollout(planner, child, dynamics, actions, depth - 1)
    end
    
    # Backpropagation
    node.visit_count += 1
    node.value_sum += value
    
    return value
end

"""
    select_ucb(planner, node, actions) → (action, child)

Select a child using UCB (within a single Thompson sample).
"""
function select_ucb(planner::ThompsonMCTS, node::MCTSNode, actions)
    best_action = nothing
    best_child = nothing
    best_ucb = -Inf
    
    log_parent = log(node.visit_count + 1)
    
    for (action, child) in node.children
        if child.visit_count == 0
            # Unvisited child — select immediately
            return action, child
        end
        
        # UCB formula
        exploitation = child.value_sum / child.visit_count
        exploration = planner.ucb_c * sqrt(log_parent / child.visit_count)
        ucb = exploitation + exploration
        
        if ucb > best_ucb
            best_ucb = ucb
            best_action = action
            best_child = child
        end
    end
    
    return best_action, best_child
end

"""
    select_unexpanded(node, actions, priors) → action

Select an unexpanded action, biased by priors.
"""
function select_unexpanded(node::MCTSNode, actions, priors::Dict)
    unexpanded = [a for a in actions if !haskey(node.children, a)]
    
    if isempty(unexpanded)
        # Shouldn't happen, but fallback
        return rand(actions)
    end
    
    # Sample according to priors
    weights = [get(priors, a, 1.0) for a in unexpanded]
    total = sum(weights)
    weights ./= total
    
    r = rand()
    cumsum = 0.0
    for (i, w) in enumerate(weights)
        cumsum += w
        if r <= cumsum
            return unexpanded[i]
        end
    end
    
    return unexpanded[end]
end

"""
    rollout(planner, node, dynamics, actions, depth) → value

Random rollout from a node to estimate value.
"""
function rollout(
    planner::ThompsonMCTS,
    node::MCTSNode,
    dynamics,
    actions,
    depth::Int
)
    if depth == 0 || node.is_terminal
        return 0.0
    end
    
    state = node.state
    total_value = 0.0
    discount = 1.0
    
    for _ in 1:depth
        # Random action
        action = rand(actions)
        
        # Get reward and next state
        reward = get_reward(dynamics, state, action)
        state = sample_next_state(dynamics, state, action)
        
        total_value += discount * reward
        discount *= planner.discount
    end
    
    return total_value
end

"""
    get_action_values(root::MCTSNode) → Dict{action, (visits, mean_value)}

Extract action statistics from the root node.
"""
function get_action_values(root::MCTSNode)
    return Dict(
        action => (child.visit_count, value(child))
        for (action, child) in root.children
    )
end
