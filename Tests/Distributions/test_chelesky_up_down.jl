## The idea is to test if the colesky up and downdate is indeed correct...

using Random
using Distributions
using LinearAlgebra
using GPSSM
using Revise
#We follow the following idea...


@testset "Test Sequential Cholesky Updates" begin
    @apf_testset "cholesky_up_down_date!" begin
        # In this test we up and down date the samve vector, which should lead again to teh same matrix!

        function update_this_chol_up_down(column, vtt, mat, q, ndim)
            tr = permutedims(vtt,(2,1))
            GPSSM.cholesky_up_down_date!(mat.U, tr, q, column, ndim)
        end


        ntest = 10
        N = 50
        ndim =1 # we only need the update for one dimension


        successful = 0
        avnorm = 0
        avsqrdnorm = 0
        avrelsqrnorm = 0
        avrelnorm = 0


        q = rand(Wishart(ndim,cholesky(Array{Float64}(I,ndim,ndim))),1)[1]
        n = N
        @time Mats = rand(Wishart(n+ndim,cholesky(Array{Float64}(I,n+ndim,n+ndim))),ntest)
        # No, we subtract a random vector...
        for j = 1:ntest
            Mats[j] = Mats[j]+ kron(Array{Float64}(I,N+1,N+1),q)
        end

        for l = 1:ntest
            for j = 1:N
                vt = Mats[l][:,(j-1)*ndim+1:j*ndim]
                mat = cholesky(Mats[l])
                chltrue = mat.U

                vt[(j-1)*ndim+1:j*ndim,:] = vt[(j-1)*ndim+1:j*ndim,:]- q

                chlours= update_this_chol_up_down(j, vt, mat, q, ndim)
                matnrm = norm(chltrue)
                nr = norm(chltrue- chlours)
                avnorm = avnorm + nr
                avsqrdnorm = avsqrdnorm+ nr^2
                avrelnorm = avrelnorm + nr/matnrm
                avrelsqrnorm = avrelsqrnorm + nr^2/matnrm^2
                successful +=1


            end
        end

        @test successful/(N*ntest) == 1
        @test  avnorm/successful ≈ 0
        @test avsqrdnorm/successful  ≈ 0
        @test  avrelsqrnorm/successful  ≈ 0
        @test avrelsqrnorm/successful  ≈ 0
    end

    @apf_testset "update_chol!" begin
        function update_true_chol(vtt,vtt0,mat,q)
            m = [mat vtt0; transpose(vtt0) vtt+q]
            cholesky(m).U
        end
        function update_this_chol(vtt,vtt0,chol,q)
            cholcu = chol.U
            vttcu = vtt+q
            t = transpose(vtt0)
            GPSSM.update_chol!(cholcu,vttcu, t)
        end

        successful = 0.0
        avnorm = 0.0
        avsqrdnorm = 0.0
        avrelsqrnorm = 0.0
        avrelnorm = 0.0
        ntest = 5

        ndim = 1
        N = 50
        testruns = 1

        for ndim = 1:1+testruns-1
            q = rand(Wishart(ndim,cholesky(Array{Float64}(I,ndim,ndim))),1)[1]

            n = N*ndim
            Mats = rand(Wishart(n+ndim,cholesky(Array{Float64}(I,n+ndim,n+ndim))),ntest)
            # No, we subtract a random vector...
            for j = 1:ntest
                Mats[j] = Mats[j]+ kron(Array{Float64}(I,N+1,N+1),q)
            end

            for l = 1:ntest
                mat = Mats[l][1:n,1:n]
                vt = Mats[l][:,n+1:end]

                #try
                chltrue = update_true_chol(vt[end-ndim+1:end,:]-q,vt[1:end-ndim,:],mat,q)
                #finally
                    #try
                chlours= update_this_chol(vt[end-ndim+1:end,:]-q,vt[1:end-ndim,:],cholesky(mat),q)

                matnrm = norm(chltrue)
                nr = norm(chltrue- chlours)
                avnorm = avnorm + nr
                avsqrdnorm = avsqrdnorm+ nr^2
                avrelnorm = avrelnorm + nr/matnrm
                avrelsqrnorm = avrelsqrnorm + nr^2/matnrm^2
                successful +=1
            end
        end

        @test successful/(ntest) == 1
        @test avnorm/successful  < 1.e-6
        @test avsqrdnorm/successful  < 1.e-6
        @test  avrelsqrnorm/successful  < 1.e-6
        @test avrelsqrnorm/successful   < 1.e-6
    end

end
