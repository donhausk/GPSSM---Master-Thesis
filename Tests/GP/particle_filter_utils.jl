
using GPSSM
using LinearAlgebra
using Distributions
@testset "Particle Filter Utils" begin
    @apf_testset "get_logjointp" begin



        differences = []
        for i = 1:100
            dimension = 50
            S = rand(Wishart(dimension,Matrix{Float64}(I,dimension,dimension)))
            v = rand(Normal(),dimension)
            push!(differences, GPSSM.get_logjointp(v, cholesky(S).U) - logpdf(MvNormal(zeros(dimension), S), v))
        end
        @test sum([!(differences[i] < 1.0e-8 ) for i = 1:length(differences)]) == 0

    end
    @apf_testset "get_traj" begin

        Xs = zeros(2,100,5)
        for i = 1:5
            for k = 1:100
                for j = 1:2
                    Xs[j,k,i] = i
                end
            end
        end

        As = ones(Integer, 5,100)
        As[:,1] = 1:5

        Traj = GPSSM.get_traj(As, Xs, 1)
        @test sum(Traj[1,1,:] .!= 1:5 ) == 0
        As[:,1] = [1,1,1,1,2]
        Traj = GPSSM.get_traj(As, Xs, 1)

        @test sum(Traj[1,1,:] .!= [1,1,1,1,2] ) == 0
        As[:,2] = [2,5,1,2,2]
        Traj = GPSSM.get_traj(As, Xs, 2)
        @test sum(Traj[1,1,:] .!= [1,1,1,1,2 ] ) == 0
        @test sum(Traj[1,2,:] .!= [2,5,1,2,2] ) == 0
        As[:,3] = [2,1,4,4,5]
        Traj = GPSSM.get_traj(As, Xs, 3)
        @test sum(Traj[1,1,:] .!= [1,1,1,1,2] ) == 0
        @test sum(Traj[1,2,:] .!= [2,5,1,2,2] ) == 0
        @test sum(Traj[1,3,:] .!= [2,1,4,4,5] ) == 0


    end


    @apf_testset "set_as!" begin

        Xs = zeros(2,100,5)
        for i = 1:5
            for k = 1:100
                for j = 1:2
                    Xs[j,k,i] = i
                end
            end
        end
        As = ones(Integer, 5,100)
        As[:,1] = 1:5

        Traj = GPSSM.get_traj(As, Xs, 1)
        @test sum(Traj[1,1,:] .!= 1:5 ) == 0
        As[:,1] = [1,1,1,1,2]
        Traj = GPSSM.get_traj(As, Xs, 1)

        @test sum(Traj[1,1,:] .!= [1,1,1,1,2] ) == 0
        As[:,2] = [2,5,1,2,2]
        Traj = GPSSM.get_traj(As, Xs, 2)
        @test sum(Traj[1,1,:] .!= [1,1,1,1,2 ] ) == 0
        @test sum(Traj[1,2,:] .!= [2,5,1,2,2] ) == 0
        As[:,3] = [2,1,4,4,5]
        Traj = GPSSM.get_traj(As, Xs, 3)
        @test sum(Traj[1,1,:] .!= [1,1,1,1,2] ) == 0
        @test sum(Traj[1,2,:] .!= [2,5,1,2,2] ) == 0
        @test sum(Traj[1,3,:] .!= [2,1,4,4,5] ) == 0



    end
    @apf_testset "reorder_indicies!" begin
        idx = [1,1,1,2,3]
        v = [1,2,3,4,5][idx]
        mvs = GPSSM.reorder_indicies!(idx)
        @test sum(idx .!= [1,2,3,1,1]) == 0
        @test sum(v[mvs] .!= [1,2,3,4,5][idx]) == 0

        idx = [1,4,5,2,2]
        v = [1,2,3,4,5][idx]
        mvs = GPSSM.reorder_indicies!(idx)
        @test sum(idx .!= [1,2,2,4,5]) == 0
        @test sum(v[mvs] .!= [1,2,3,4,5][idx]) == 0

        idx = [4,4,4,5,2]
        v = [1,2,3,4,5][idx]
        mvs = GPSSM.reorder_indicies!(idx)
        @test sum(idx .!= [4,2,4,4,5]) == 0
        @test sum(v[mvs] .!= [1,2,3,4,5][idx]) == 0
    end

    @apf_testset "copy_cholesky!" begin
        lssize = (2,10,2)
        i = 2
        XT = Float64
        n_particles = 5
        for i = 1:10

            Chols = Matrix{UpperTriangular{XT, Array{XT,2}}}(undef,lssize[1], n_particles)
            CholsPGAS = Matrix{UpperTriangular{XT, Array{XT,2}}}(undef, lssize[1], n_particles)
            Ktt_save = Matrix{Matrix{XT}}(undef, lssize[1], n_particles)
            Means_save = Matrix{Vector{XT}}(undef, lssize[1], n_particles)
            # We will always reuse them
            for j = 1:n_particles
                for i =1:lssize[1]
                    Chols[i,j] = UpperTriangular{XT, Array{XT,2}}(rand( lssize[2],lssize[2]))
                    CholsPGAS[i,j] = UpperTriangular{XT, Array{XT,2}}(rand( lssize[2]+1,lssize[2]+1))
                    Ktt_save[i,j] = rand( 1, lssize[2]+1)
                    Means_save[i,j] = rand( lssize[2]+1)
                end
            end
            indx = rand(1:5,5)
            GPSSM.reorder_indicies!(indx)
            println(indx)
            idc = (rand(1:2), rand(1:5))
            idcpgas =(rand(1:2), rand(1:5))
            idcktt =(rand(1:2), rand(1:5))
            idcmean =(rand(1:2), rand(1:5))

            testc =deepcopy(Chols[idc[1], indx[idc[2]]])
            testcpgas = deepcopy(CholsPGAS[idcpgas[1], indx[idcpgas[2]]])
            testktt=deepcopy(Ktt_save[idcktt[1], indx[idcktt[2]]])
            testmean=deepcopy(Means_save[idcmean[1], indx[idcmean[2]]])
            Cholini = deepcopy(CholsPGAS)
            GPSSM.copy_cholesky!(CholsPGAS, Chols, Ktt_save, Means_save, indx, lssize[1], i+1)
            if i > 1
                @test  norm(testc[1:i,1:i] .- Chols[idc...][1:i,1:i]) < 1.0e-12 #some numerical error might exists...
            end

            @test  norm(testcpgas .- CholsPGAS[idcpgas...]) < 1.0e-12 #some numerical error might exists...
            @test  norm(testktt[1,1:i] .- Ktt_save[idcktt...][1,1:i]) <1.0e-12 #some numerical error might exists...
            @test  norm(testmean[1:i] .- Means_save[idcmean...][1:i]) <1.0e-12 #some numerical error might exists...

        end


    end

end
