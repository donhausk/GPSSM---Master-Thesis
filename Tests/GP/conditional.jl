


using GPSSM
using LinearAlgebra
@testset "Test_GP" begin
    @apf_testset "get_transition_mean full gp" begin

        # The direct formula from equaton Frigolas PHD thesis ( equation 3.15 page 30)

        function clean_f(
            xmeanst::AbstractVector,
            xmeans0t::AbstractVector,
            xout::AbstractVector,
            Ktt0t::AbstractMatrix,
            Chol::UpperTriangular
        )
            return xmeanst + Ktt0t * inv(Chol'*Chol) * ( xout - xmeans0t)
        end


        for i = 3:100
            t = i
            xmeanst = rand(Normal(),1) # 1 D
            xmeans0t = rand(Normal(),t-1) # 0:t-2 has length t-1
            xout = rand(Normal(), t-1) # 0:t-2 has again length t-1
            Kfull = rand(Wishart(t,Matrix{Float64}(I,t-1,t-1))) # K0:t-1,0:t-1
            Q = rand(Exponential())
            Chol = cholesky(Kfull + Diagonal(ones(t-1) .* Q)).U
            Ktt0t = Kfull[t-1:t-1, 1:t-1]

            mpredtrue = clean_f( xmeanst, xmeans0t, xout, Ktt0t, Chol)
            mpredgpssm = GPSSM.get_transition_mean( xmeanst, xmeans0t, xout, Ktt0t, Chol)
            @test mpredtrue ≈ mpredgpssm
        end

    end


    @apf_testset "get_transition_cov full gp" begin

        # The direct formula from equaton Frigolas PHD thesis ( equation 3.15 page 30)

        function clean_f(
            Ktt0t::AbstractMatrix,
            Kttq::AbstractMatrix,
            Chol::UpperTriangular
        )
            return Kttq - Ktt0t * inv(Chol'* Chol)* Ktt0t'
        end


        for i = 3:100
            t = i
            Kfull = rand(Wishart(t,Matrix{Float64}(I,t,t))) # K0:t,0:t
            Q = rand(Exponential())
            Chol = cholesky(Kfull[1:t-1,1:t-1] + Diagonal(ones(t-1) .* Q)).U
            Ktt0t = Kfull[t-1:t-1, 1:t-1]
            Kttq = Kfull[t:t,t:t] .+ Q

            mpredtrue = clean_f( Ktt0t, Kttq, Chol)
            mpredgpssm = GPSSM.get_transition_cov( Ktt0t, Kttq, Chol)
            @test mpredtrue ≈ mpredgpssm
        end
        @apf_testset "get_transition_cov full gp" begin

            # The direct formula from equaton Frigolas PHD thesis ( equation 3.15 page 30)

            function clean_f(
                Ktt0t::AbstractMatrix,
                Kttq::AbstractMatrix,
                Chol::UpperTriangular
            )
                return Kttq - Ktt0t * inv(Chol'* Chol)* Ktt0t'
            end


            for i = 3:100
                t = i
                Kfull = rand(Wishart(t,Matrix{Float64}(I,t,t))) # K0:t,0:t
                Q = rand(Exponential())
                Chol = cholesky(Kfull[1:t-1,1:t-1] + Diagonal(ones(t-1) .* Q)).U
                Ktt0t = Kfull[t-1:t-1, 1:t-1]
                Kttq = Kfull[t:t,t:t] .+ Q

                mpredtrue = clean_f( Ktt0t, Kttq, Chol)
                mpredgpssm = GPSSM.get_transition_cov( Ktt0t, Kttq, Chol)
                @test mpredtrue ≈ mpredgpssm
            end

    end
end
