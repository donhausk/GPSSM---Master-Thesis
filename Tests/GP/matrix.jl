using GPSSM
using LinearAlgebra
using Stheno
@testset "Test Gaussian Process Files" begin
    tol = 1.0e-3 # inverse are not very stable, we need a relatively high tolerance.
    abstol = 1.0e-3
    @apf_testset "K_CTCQ_Chol" begin

        for i = 3:50
            l = rand(5:10)
            l2 = rand( 11:20)
            Kzz = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            Kxz =rand(Uniform(), l2, l)
            Q = rand(Exponential())
            truef = cholesky(Symmetric(Kzz + Kxz'*  Kxz / Q)).U

            @test norm(GPSSM.K_CTCQ_Chol(cholesky(Kzz).U, Kxz, Q) .- truef)/norm(truef) < tol
        end

    end
    @apf_testset "K_ΣInv_K" begin

        for i = 3:50
            l = rand(5:10)
            K1 = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            K2 =rand(Wishart(l,Matrix{Float64}(I,l,l)))
            truef = cholesky(Symmetric(K2*inv(K1)*K2)).U
            @test norm(GPSSM.K_ΣInv_K(cholesky(K1).U,cholesky(K2).U) .- truef)/norm(truef) < abstol
        end

    end
    @apf_testset "A_Kinv_A" begin

        for i = 3:50
            l = rand(10:15)
            l2 = rand( 1:10)
            Kzz = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            Kxz =rand(Uniform(), l2, l)

            truef = Symmetric(Kxz * inv(Kzz)* Kxz')

            @test norm(GPSSM.A_Kinv_A(cholesky(Kzz).U, Kxz') .- truef)/norm(truef) < abstol
        end

    end
    @apf_testset "A_K_A" begin

        for i = 3:50
            l = rand(10:15)
            l2 = rand( 1:10)
            Kzz = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            Kxz =rand(Uniform(), l2, l)

            truef = Symmetric(Kxz * Kzz* Kxz')

            @test norm(GPSSM.A_K_A(cholesky(Kzz).U, Kxz) .- truef)/norm(truef) < abstol
        end

    end
    @apf_testset "Kinv_A" begin

        for i = 3:50
            l = rand(10:15)
            l2 = rand( 1:10)
            Kzz = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            Kxz =rand(Uniform(), l2, l)

            truef = inv(Kzz)* Kxz'

            @test norm(GPSSM.Kinv_A(cholesky(Kzz).U, Kxz') .- truef)/norm(truef)  < abstol
        end

    end
    @apf_testset "Ainv_K_Ainv" begin

        for i = 3:50
            l = rand(5:10)
            K1 = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            K2 =rand(Wishart(l,Matrix{Float64}(I,l,l)))
            truef = cholesky(Symmetric(inv(K2)*K1*inv(K2))).U
            @test norm(GPSSM.Ainv_K_Ainv(cholesky(K1).U,cholesky(K2).U) .- truef)/norm(truef) < abstol
        end
    end

    @apf_testset "CholInv" begin

        for i = 3:20
            l = rand(5:20)
            K1 = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            truef = Symmetric(inv(K1))
            @test norm(GPSSM.CholInv(cholesky(K1).U) .- truef)/norm(truef) < abstol
        end
    end

    @apf_testset "get_At" begin

        for i = 3:50
            l = rand(10:15)
            l2 = rand( 1:10)
            Kzz = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            Kxz =rand(Uniform(), l2, l)
            truef = Kxz * inv(Kzz)
            @test norm(GPSSM.get_At(cholesky(Kzz).U, Kxz) .- truef)/norm(truef) < abstol
        end
    end
    @apf_testset "get_Bt" begin

        for i = 3:50
            l = rand(8:12)
            l2 = rand( 1:10)
            Kxx = rand(Wishart(l2,Matrix{Float64}(I,l2,l2)))

            Kzz = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            Kxz =rand(Uniform(), l2, l)
            truef = Kxx - Kxz * inv(Kzz)* Kxz'
            @test norm(GPSSM.get_Bt(cholesky(Kzz).U, Kxz, Kxx) .- truef)/norm(truef) < abstol
        end
    end



    @apf_testset "QInv_B_A_Σ_A" begin

        for i = 3:50
            l = rand(8:10)
            l2 = rand( 1:5)
            Kxx = rand(Wishart(l2,Matrix{Float64}(I,l2,l2)))
            Kqq = rand(Wishart(l,Matrix{Float64}(I,l,l)))

            Kzz = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            Kxz =rand(Uniform(), l2, l)
            Q = rand(Exponential())
            truef = -0.5 *  tr(1/Q * ( Kxx -  Kxz * inv(Kzz)* Kxz' + Kxz *inv(Kzz)* Kqq * (Kxz *inv(Kzz))'))

            @test norm(GPSSM.QInv_B_A_Σ_A(cholesky(Kqq).U, cholesky(Kzz).U, Kxz, Kxx, Q) .- truef)/norm(truef) < abstol
        end
    end



    @apf_testset "KL" begin

        for i = 3:50
            l = rand(10:15)
            l2 = rand( 1:10)

            Knn = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            Kzz = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            mn = rand(Normal(), l)
            mz = rand(Normal(), l)



            truef = 0.5 *log(det(Kzz*inv(Knn))) + 0.5 * tr(inv(Kzz)*((mn - mz)*(mn - mz)' + Knn - Kzz))

            @test norm(GPSSM.KL(mn, mz, cholesky(Knn).U, cholesky(Kzz).U) .- truef)/norm(truef) < abstol
        end
    end



    @apf_testset "get_Σ_μ_VI" begin
        for i = 1:10
            Mu1 = rand(10,5)
            Mu2 = zeros(10,10,5)
            for k = 1:5
                Mu2[:,:,k] = -2 .*rand(Wishart(10,Matrix{Float64}(I,10,10)))
            end
            μuq, Σuq = GPSSM.get_Σ_μ_VI(Mu1, Mu2)

            for k = 1:5
                Sigmatrue = Symmetric(inv(-2 .* Mu2[:,:,k]))
                mutrue = Sigmatrue * Mu1[:,k]
                @test norm(μuq[k] .- mutrue) < tol
                @test norm(Σuq[k] .- cholesky(Sigmatrue).U) < tol
            end

        end
    end


    @apf_testset "elbo" begin


        for i = 3:50
            l = rand(10:15)
            l2 = rand( 16:20)


            Z= rand(1, l)
            X = rand(1, l2)

            Kzz = Stheno.pairwise(Stheno.Matern32(), Stheno.ColVecs(Z))
            Kxx = Stheno.pairwise(Stheno.Matern32(), Stheno.ColVecs(X))
            Kxz = Stheno.pairwise(Stheno.Matern32(), Stheno.ColVecs(X), Stheno.ColVecs(Z))
            Kqq = rand(Wishart(l,Matrix{Float64}(I,l,l)))
            mq = rand(Normal(), l)
            mz = rand(Normal(), l)
            mx = rand(Normal(), l2)
            y = rand(Normal(), l2)
            Q = rand(Exponential())
            IQ = Matrix{Float64}(I, l2,l2)


            e = GPSSM.elbo(mq, cholesky(Kqq).U ,cholesky(Kzz).U, diag(Kxx), Kxz,  Q, y, mx,  mz)

            Kyopt = IQ*Q + Kxz * inv(Kzz) * Kxz'


            Kopt = Kzz - Kxz' * inv(Kyopt) *Kxz
            mopt = mz +  Kxz' * inv(Kyopt) * (y- mx)
            KL = 0.5 *log(det(Kopt*inv(Kqq))) + 0.5 *tr(inv(Kopt)*((mq - mopt)*(mq - mopt)' +  Kqq - Kopt))

            elbooptimal = logpdf(GPSSM.Zygote_MvNormal(mx, Kyopt),y)- 0.5* tr(Kxx - Kxz * inv(Kzz)* Kxz')


            truef = elbooptimal - KL

            @test norm(e.- truef)/norm(truef) < abstol
        end
    end


    @apf_testset "elbo_otpimal" begin


        for i = 3:50
            l = rand(10:15)
            l2 = rand( 16:20)


            Z= rand(1, l)
            X = rand(1, l2)

            Kzz = Stheno.pairwise(Stheno.Matern32(), Stheno.ColVecs(Z))
            Kxx = Stheno.pairwise(Stheno.Matern32(), Stheno.ColVecs(X))
            Kxz = Stheno.pairwise(Stheno.Matern32(), Stheno.ColVecs(X), Stheno.ColVecs(Z))
            mz = rand(Normal(), l)
            mx = rand(Normal(), l2)
            y = rand(Normal(), l2)
            Q = rand(Exponential())
            IQ = Matrix{Float64}(I, l2,l2)

            e = GPSSM.elbo_optimal( diag(Kxx), Kxz, cholesky(Kzz).U,  Q, y, mx)
            Kyopt = Symmetric(IQ* Q + Kxz * inv(Kzz) * Kxz')

            elbooptimal = logpdf(GPSSM.Zygote_MvNormal(mx, Kyopt), y)- 0.5/Q* tr(Kxx - Kxz * inv(Kzz)* Kxz')


            truef = elbooptimal
            # Intense numerical error
            @test norm(e.- truef)/norm(truef) < abstol
        end
    end
end
