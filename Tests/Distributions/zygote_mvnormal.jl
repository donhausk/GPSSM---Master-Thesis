
using GPSSM
using LinearAlgebra

@testset "Zygote_MvNormal" begin
    @apf_testset "logpdf" begin
        #Test general case
        dimension = 2
        A = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        S = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        b = rand(Normal(), dimension)

        vals = []
        tvals = []
        for i = 1:100
            v = rand(Normal(),dimension)
            v2 = rand(Normal(),dimension)
            push!(vals, logpdf(GPSSM.Zygote_MvNormal(A*v .+b, S),v2)) #our variant
            push!(tvals, logpdf(MvNormal(A*v .+b, S),v2)) #the ground truth
        end

        @test sum(vals .≈ tvals) == length(vals)


        #Test one dimensional

        dimension = 1
        A = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        S = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        b = rand(Normal(), dimension)

        vals = []
        tvals = []
        for i = 1:100
            v = rand(Normal(),dimension)
            v2 = rand(Normal(),dimension)
            push!(vals, logpdf(GPSSM.Zygote_MvNormal(A*v .+b, S),v2)) #our variant
            push!(tvals, logpdf(MvNormal(A*v .+b, S),v2)) #the ground truth
        end

        @test sum(vals .≈ tvals) == length(vals)

        #Test special inputs
        dimension = 5
        A = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        b = rand(Normal(), dimension)

        vals = []
        tvals = []
        for i = 1:100
            s = rand(Exponential())
            v = rand(Normal(),dimension)
            v2 = rand(Normal(),dimension)
            push!(vals, logpdf(GPSSM.Zygote_MvNormal(A*v .+b, s),v2)) #our variant
            push!(tvals, logpdf(MvNormal(A*v .+b, Diagonal(ones(dimension).*s)),v2)) #the ground truth
        end

        @test sum(vals .≈ tvals) == length(vals)

        dimension = 5
        A = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        b = rand(Normal(), dimension)

        vals = []
        tvals = []
        for i = 1:100
            s = rand(Exponential())
            v = rand(Normal(),dimension)
            v2 = rand(Normal(),dimension)
            push!(vals, logpdf(GPSSM.Zygote_MvNormal(A*v .+b, ones(dimension)*s),v2)) #our variant
            push!(tvals, logpdf(MvNormal(A*v .+b, Diagonal(ones(dimension).*s)),v2)) #the ground truth
        end

        @test sum(vals .≈ tvals) == length(vals)

    end

    @apf_testset "logpdfchol" begin
        #Test general case
        dimension = 2
        A = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        S = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        b = rand(Normal(), dimension)

        vals = []
        tvals = []
        for i = 1:100
            v = rand(Normal(),dimension)
            v2 = rand(Normal(),dimension)
            push!(vals, logpdf(GPSSM.Zygote_MvNormal(A*v .+b, cholesky(S).U),v2)) #our variant
            push!(tvals, logpdf(MvNormal(A*v .+b, S),v2)) #the ground truth
        end

        @test sum(vals .≈ tvals) == length(vals)


        #Test one dimensional

        dimension = 1
        A = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        S = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        b = rand(Normal(), dimension)

        vals = []
        tvals = []
        for i = 1:100
            v = rand(Normal(),dimension)
            v2 = rand(Normal(),dimension)
            push!(vals, logpdf(GPSSM.Zygote_MvNormal(A*v .+b, cholesky(S).U),v2)) #our variant
            push!(tvals, logpdf(MvNormal(A*v .+b, S),v2)) #the ground truth
        end

        @test sum(vals .≈ tvals) == length(vals)


    end

    @apf_testset "rand" begin

        dimension = 5
        A = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        b = rand(Normal(), dimension)

        for i = 1:3
            s = rand(Exponential())
            v = rand(Normal(),dimension)
            draws = rand(GPSSM.Zygote_MvNormal(A*v .+b, s),100000)#our variant
            @test mean(abs.(mean(draws, dims=2) .- mean(GPSSM.Zygote_MvNormal(A*v .+b, s)))./abs.(mean(GPSSM.Zygote_MvNormal(A*v .+b, s)))) < 0.1
            @test mean(diag(abs.(cov(draws, dims=2) .- cov(GPSSM.Zygote_MvNormal(A*v .+b, s)))./abs.(cov(draws, dims=2)))) < 0.1
        end

        dimension = 1
        A = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
        b = rand(Normal(), dimension)

        for i = 1:3
            s = rand(Exponential())
            v = rand(Normal(),dimension)
            draws = rand(GPSSM.Zygote_MvNormal(A*v .+b, s),100000)#our variant
            @test mean(abs.(mean(draws, dims=2) .- mean(GPSSM.Zygote_MvNormal(A*v .+b, s)))./abs.(mean(GPSSM.Zygote_MvNormal(A*v .+b, s)))) < 0.1
            @test mean(diag(abs.(cov(draws, dims=2) .- cov(GPSSM.Zygote_MvNormal(A*v .+b, s)))./abs.(cov(draws, dims=2)))) < 0.1
        end
    end

end
