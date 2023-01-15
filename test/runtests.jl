using Revise
using Test
using HistoryStacks

@testset "history stacks" begin
    n = 2
    p = 2
    M = 10
    δ = 0.01
    H = HistoryStack(M, δ)

    @test HistoryStacks.isempty(H) == true
    @test HistoryStacks.length(H) == 0
    @test HistoryStacks.isfull(H) == false
    @test HistoryStacks.isnotfull(H) == true
    
    update!(H, rand(n, p), rand(n))

    @test HistoryStacks.isempty(H) == false
    @test HistoryStacks.length(H) == 1
    @test HistoryStacks.isfull(H) == false
    @test HistoryStacks.isnotfull(H) == true

    λ = []
    for i in 1:30
        update!(H, rand(n, p), rand(n))
        push!(λ, HistoryStacks.eigmin(H))
    end
    
    @test HistoryStacks.isempty(H) == false
    @test HistoryStacks.length(H) == M
    @test HistoryStacks.isfull(H) == true
    @test HistoryStacks.isnotfull(H) == false 
    @test λ[end] >= λ[1]

    HistoryStacks.gradient_vector_field(H, rand(p))
    HistoryStacks.upper_bound_update(H, rand(), 1.0)
end