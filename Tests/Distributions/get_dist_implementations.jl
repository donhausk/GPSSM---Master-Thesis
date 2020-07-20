
# In this file we test the dist_xy mechanism

using GPSSM
using LinearAlgebra
@testset "get_dists" begin
    @apf_testset "get_distxy_and_args" begin

        for k = 1:10
            dimension = k
            A = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
            S = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
            b = rand(Normal(), dimension)


            function GPSSM.get_distxy_and_args(X::AbstractVector; A, b,S)
                return GPSSM.Zygote_MvNormal, (A*X .+ b, S)
            end
            noise = 0.25 #variance

            params = (A=A, b=b, S=S)

            vals = []
            tvals = []
            for i = 1:100
                v = rand(Normal(),dimension)
                v2 = rand(Normal(),dimension)
                dist, args = GPSSM.get_distxy_and_args(v; params...)
                push!(vals, logpdf(dist(args...),v2)) #our variant
                push!(tvals, logpdf(MvNormal(A*v+b,S),v2)) #the ground truth
            end

            @test sum(vals .≈ tvals) == length(vals)
        end


        for k = 1:10
            dimension = k
            A = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
            S = rand(Exponential(),dimension)
            b = rand(Normal(), dimension)


            function GPSSM.get_distxy_and_args(X::AbstractVector; A, b,S)
                return GPSSM.Zygote_MvNormal, (A*X .+ b, Diagonal(S))
            end
            noise = 0.25 #variance

            params = (A=A, b=b, S=S)

            vals = []
            tvals = []
            for i = 1:100
                v = rand(Normal(),dimension)
                v2 = rand(Normal(),dimension)
                dist, args = GPSSM.get_distxy_and_args(v; params...)
                push!(vals, logpdf(dist(args...),v2)) #our variant
                push!(tvals, logpdf(MvNormal(A*v+b,Diagonal(S)),v2)) #the ground truth
            end

            @test sum(vals .≈ tvals) == length(vals)
        end
    end
end
