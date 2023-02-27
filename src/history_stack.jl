"""
    HistoryStack

History stack of input-output data used for parameter estimation.

# Fields
- `regressor` : vector of saved regressor values
- `target` : vector of saved target values
- `M::Int` : maximum number of stack elements
- `δ::Float64` : tolerance for adding data to stack
"""
mutable struct HistoryStack
    regressor
    target
    M::Int
    δ::Float64
end

# Main constructors
HistoryStack(M::Int, δ::Float64) = HistoryStack([], [], M, δ)
HistoryStack(M::Int) = HistoryStack(M, 0.01)

# Extend some base functionality
Base.isempty(H::HistoryStack) = isempty(H.regressor)
Base.length(H::HistoryStack) = isempty(H) ? 0 : length(H.regressor)
isfull(H::HistoryStack) = length(H) >= H.M
isnotfull(H::HistoryStack) = length(H) < H.M
isdifferent(H::HistoryStack, regressor) =  norm(regressor - H.regressor[end])^2/norm(regressor) >= H.δ

function Base.push!(H::HistoryStack, regressor, target)
    push!(H.regressor, regressor)
    push!(H.target, target)

    return H
end

function Base.insert!(H::HistoryStack, idx::Int, regressor, target)
    H.regressor[idx] = regressor
    H.target[idx] = target

    return H
end

function stacksum(H::HistoryStack)
    if isempty(H)
        return 0.0
    else
        if isa(H.regressor[1], Float64)
            return H.regressor*H.regressor'
        else
            return sum([x'x for x in H.regressor])
        end
    end
end

function regressor_target_sum(H::HistoryStack)
    return sum([F'y for (F, y) in zip(H.regressor, H.target)])
end

LinearAlgebra.eigmin(H::HistoryStack) = eigmin(stacksum(H))


# Main functionality
"""
    update!(H::HistoryStack, target, regressor)

Update history stack with new data using the algorithm from

G. Chowdhary and E. Johnson, "A singular value maximizing data recording algorithm for concurrent learning," in Proceedings of the American Control Conference, pp. 3547-3552, 2011.

This algorithm adds data to the stack until it is full, at which point it only adds data if it possible to increase the minimum eigenvalue of the stack.
"""
function update!(H::HistoryStack, regressor, target)
    if isempty(H)
        push!(H, regressor, target)
    else
        if isdifferent(H, regressor)
            if isnotfull(H)
                push!(H, regressor, target)
            else
                temp = deepcopy(H)
                λ0 = eigmin(temp)
                λ = []
                for j in 1:H.M
                    insert!(H, j, regressor, target)
                    push!(λ, eigmin(H))
                    H = deepcopy(temp)
                end
                λmax = maximum(λ)
                if λmax > λ0
                    insert!(H, argmax(λ), regressor, target)
                else
                    H = deepcopy(temp)
                end
            end
        end
    end

    return H
end

# Main gradient update law
function gradient_vector_field(H::HistoryStack, θ̂::Union{Float64,Vector{Float64}})
    if isempty(H)
        return length(θ̂) == 1 ? 0.0 : zeros(length(θ̂))
    else
        return sum([F'*(Y - F*θ̂) for (F, Y) in zip(H.regressor, H.target)])
    end
end

# Upper bound update law
upper_bound_update(H::HistoryStack, θ̃::Float64, γ::Float64) = isempty(H) ? 0.0 : -γ*eigmin(H)*θ̃
function upper_bound_update(H::HistoryStack, θ̃::Float64, Γ::Matrix{Float64})
    return isempty(H) ? 0.0 : -minimum(real(eigvals(Γ*stacksum(H))))*θ̃
end
function upper_bound_update(H::HistoryStack, θ̃::Vector{Float64}, Γ::Matrix{Float64})  
    return isempty(H) ? zeros(length(θ̂)) : -Γ*stacksum(H)*θ̃
end

# Get worst-case parameter estimation error
worst_case_error(𝚯::Vector{Float64}) = maximum(𝚯) - minimum(𝚯) 
worst_case_error(𝚯::Vector{Vector{Float64}}) = norm([maximum(θb) - minimum(θb) for θb in 𝚯])