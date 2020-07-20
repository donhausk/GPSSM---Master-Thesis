

# export JULIA_NUM_THREADS=4

push!(LOAD_PATH, "/home/kongi/GPSSM")
push!(LOAD_PATH, "/home/kongi/GPSSM/Test_Runner/")

using Distributed
using AbstractMCMC
using Distributions
using Stheno
using JLD2
using LinearAlgebra
using Plots
using StatsPlots
using Turing
using Flux
using Random
import Test_Runner: @run_experiments, @getelementm, ExpData, storagepath,
    datapath, Gaussian1D, kinkf, get_test_specifications, generate_data, Narendra_li, NarendraLiParameters
using Test_Runner
using Revise
using GPSSM
import GPSSM: GPSSMInstance
using Zygote
using Bijectors

struct LinearInstance <: GPSSMInstance end
function GPSSM.get_distxy_and_args(i::LinearInstance, X::Vector; A, b,S)
    return GPSSM.Zygote_MvNormal, (A*X+b, S)
end
GPSSM.get_distx0_and_args(i::LinearInstance) = (GPSSM.Zygote_MvNormal, ([0.0], 1.0))

function GPSSM.get_kernels(i::LinearInstance, args)

    (Stheno.scale(  Stheno.stretch(Stheno.Matern32(), args[:d1][:l]),args[:d1][:l2]),)
end
function GPSSM.get_means(i::LinearInstance, args)
    (Stheno.ZeroMean(),)
end
GPSSM.get_latent_type(i::LinearInstance) = Float64

get_space(val::Val{NT}) where NT = NT



@inline GPSSM.add_jitter(i::LinearInstance) = false

@testset "varinf test" begin
    @testset "generate_vi" begin
        ## Struct 1
        hp = GPSSM.GPSpec(
            1,
            NamedTuple(),
            [(l= 2.9, l2 =2.0)],
            typeof(NamedTuple())[NamedTuple()],
            0.1, #We start with a huge amount of noise....
            (A =ones(1,1),b = [0.0], S = ones(1,1));
            trainQ = "F",
            trainθxkernel = [(l = "TP", l2 = "TP")],
            trainθy = (A = "F",b = "F", S = "F"))
        varinf = GPSSM.generate_vi(hp)
        k_args = GPSSM.get_kernel_args(varinf, hp)
        mean_args = GPSSM.get_mean_args(varinf, hp)
        Q = GPSSM.get_Q(varinf)
        argsy = GPSSM.get_distθy_args(varinf, hp)
        argx0 = GPSSM.get_distθx0_args(varinf, hp)
        @test k_args == (d1 = (l= 2.9, l2 =2.0),)
        @test mean_args == (d1 = NamedTuple(),)
        @test argsy == (A =ones(1,1),b = [0.0], S = ones(1,1))
        @test argx0 == NamedTuple()
        @test Q == [0.1]
        @test varinf[GPSSM.TrainableSelector] == [2.9, 2.0]
        space = GPSSM._getspace(varinf, GPSSM.TrainableSelector)
        @test get_space(space) == ( :l_kernel_d1 , :l2_kernel_d1 )

        ## Struct 2
        hp = GPSSM.GPSpec(
            2, #  2-dimensional
            (q = ones(3,3), ),
            [(l= 2.9, l2 =2.0)],
            [(a = 0.1, ),(b = 0.2 , a = 0.3)], # Different HP for different dimensions
            0.1, #We start with a huge amount of noise....
            (A =ones(1,1),b = [0.0], S = ones(1,1));
            trainQ = "F",
            trainθxkernel = [(l = "TP", l2 = "TP"), (l = "F", l2 = "T")],
            trainθy = (A = "T",b = "F", S = "F"))
        varinf = GPSSM.generate_vi(hp)
        k_args = GPSSM.get_kernel_args(varinf, hp)
        mean_args = GPSSM.get_mean_args(varinf, hp)
        Q = GPSSM.get_Q(varinf)
        argsy = GPSSM.get_distθy_args(varinf, hp)
        argx0 = GPSSM.get_distθx0_args(varinf, hp)
        @test k_args == (d1 = (l= 2.9, l2 =2.0),d2 = (l= 2.9, l2 =2.0))
        @test mean_args == (d1 = (a = 0.1,), d2 = (b = 0.2, a=0.3))
        @test argsy == (A =ones(1,1),b = [0.0], S = ones(1,1))
        @test argx0 == (q = ones(3,3),) # We need also to check if it correctly reconstructs...
        @test Q == [0.1, 0.1] # Because we are two dimensional

        @test length(varinf[GPSSM.TrainableSelector]) == 7+9 # We train l, l2 -d1, l2 -d2, A and all of meanargs + all of argsx0
        space = GPSSM._getspace(varinf, GPSSM.TrainableSelector)
        @test get_space(space) == (:q_x0, :A_xy, :l_kernel_d1, :l2_kernel_d1, :a_mean_d1, :l2_kernel_d2, :b_mean_d2, :a_mean_d2)

        ## Struct 3
        hp = GPSSM.GPSpec(
            2, #  2-dimensional
            (q = ones(3,3),),
            [(l= Uniform(), l2 = MvLogNormal(2,1.0),)],
            [(a = Matrix(Diagonal(ones(2))),)], # Different HP for different dimensions
            [0.1, 0.3], #We start with a huge amount of noise....
            (A = Normal(),b = Exponential(), S = MvNormal(2, 1.0));
            trainQ = "F",
            trainθxkernel = [(l = "TP", l2 = "TP"), (l = "F", l2 = "T")],
            trainθy = (A = "T",b = "F", S = "F"))
        varinf = GPSSM.generate_vi(hp)
        k_args = GPSSM.get_kernel_args(varinf, hp)
        mean_args = GPSSM.get_mean_args(varinf, hp)
        Q = GPSSM.get_Q(varinf)
        argsy = GPSSM.get_distθy_args(varinf, hp)
        argx0 = GPSSM.get_distθx0_args(varinf, hp)
        @test k_args.d1.l > 0.0 && k_args.d1.l < 1.0
        @test sum(k_args.d1.l2 .> [0.0, 0.0]) == 2
        @test isposdef(mean_args.d1.a )
        @test size(mean_args.d1.a) == (2,2)
        @test isposdef(mean_args.d2.a )
        @test size(mean_args.d2.a) == (2,2)
        @test argsy.b > 0.0
        @test size(argsy.S) == (2,)
        @test Q == [0.1, 0.3] # Because we are two dimensional
    end
    @testset "link_vi" begin
        ## Struct 1
        hp = GPSSM.GPSpec(
            1,
            NamedTuple(),
            [(l= 3.0, l2 =2.0)],
            typeof(NamedTuple())[NamedTuple()],
            0.1, #We start with a huge amount of noise....
            (A =ones(1,1),b = [0.0], S = ones(1,1));
            trainQ = "F",
            trainθxkernel = [(l = "TP", l2 = "T")],
            trainθy = (A = "TP",b = "F", S = "F"))
        varinf = GPSSM.generate_vi(hp)
        space = GPSSM._getspace(varinf, GPSSM.TrainableSelector)
        GPSSM.link!(varinf,GPSSM.TrainableSelector)

        k_args = GPSSM.get_kernel_args(varinf, hp)
        argsy = GPSSM.get_distθy_args(varinf, hp)
        @test k_args == (d1 = (l= 3.0, l2 =2.0),)
        @test argsy == (A =(ones(1,1)),b = [0.0], S = ones(1,1))
        @test varinf.metadata.A_xy.vals[:] == [0.0]
        @test varinf.metadata.S_xy.vals[:] == [1.0]
        @test varinf.metadata.l_kernel_d1.vals[:] == [log(3.0)]
        @test varinf.metadata.l2_kernel_d1.vals[:] == [2.0]

        @test varinf[GPSSM.TrainableSelector] == [0.0, log(3.0), 2.0 ]
        space = GPSSM._getspace(varinf, GPSSM.TrainableSelector)

        new_varinfo = GPSSM.NewVarInfo(varinf, space, varinf[GPSSM.TrainableSelector])
        @test new_varinfo[GPSSM.TrainableSelector] == [0.0, log(3.0), 2.0 ]
        k_args = GPSSM.get_kernel_args(new_varinfo, hp)
        argsy = GPSSM.get_distθy_args(new_varinfo, hp)
        @test k_args == (d1 = (l= 3.0, l2 =2.0),)
        @test argsy == (A =(ones(1,1)),b = [0.0], S = ones(1,1))
        @test new_varinfo.metadata.A_xy.vals[:] == [0.0]
        @test new_varinfo.metadata.S_xy.vals[:] == [1.0]
        @test new_varinfo.metadata.l_kernel_d1.vals[:] == [log(3.0)]
        @test new_varinfo.metadata.l2_kernel_d1.vals[:] == [2.0]
        GPSSM.invlink!(new_varinfo,GPSSM.TrainableSelector)
        k_args = GPSSM.get_kernel_args(new_varinfo, hp)
        argsy = GPSSM.get_distθy_args(new_varinfo, hp)
        @test k_args == (d1 = (l= 3.0, l2 =2.0),)
        @test argsy == (A =(ones(1,1)),b = [0.0], S = ones(1,1))
        @test new_varinfo.metadata.A_xy.vals[:] == [1.0]
        @test new_varinfo.metadata.S_xy.vals[:] == [1.0]
        @test new_varinfo.metadata.l_kernel_d1.vals[:] == [3.0]
        @test new_varinfo.metadata.l2_kernel_d1.vals[:] == [2.0]


    end
    @testset "zygote" begin


        ## Struct 1
        hp = GPSSM.GPSpec(
            1,
            NamedTuple(),
            [(l= 2.9, l2 =2.0)],
            typeof(NamedTuple())[NamedTuple()],
            0.1, #We start with a huge amount of noise....
            (A =Wishart(5, Matrix(Diagonal(ones(5)))),b = zeros(5), S = ones(5,));
            trainQ = "F",
            trainθxkernel = [(l = "TP", l2 = "T")],
            trainθy = (A = "F",b = "F", S = "F"))
        varinf = GPSSM.generate_vi(hp)
        space = GPSSM._getspace(varinf, GPSSM.TrainableSelector)

        DynamicPPL.link!(varinf,GPSSM.TrainableSelector)

        space = GPSSM._getspace(varinf, GPSSM.TrainableSelector)
        θ = varinf[GPSSM.TrainableSelector]
        K = rand(10,10)

        function f(θ)
            new_varinfo = GPSSM.NewVarInfo(varinf, space, θ )
            k_args = GPSSM.get_kernel_args(new_varinfo, hp)
            kernel =  GPSSM.get_kernels(LinearInstance(),k_args)[1]
            return norm(GPSSM.Kxx(LinearInstance(), kernel, K))
        end
        # true function
        function f_true(θ)
            # We manually apply the bijector here...
            norm(GPSSM.Kxx(LinearInstance(), Stheno.scale(  Stheno.stretch(Stheno.Matern32(), exp(θ[1])),θ[2]), K))
        end
        # test kernel_args
        @test Zygote.pullback(f,θ)[2](1)[1] ==  Zygote.pullback(f_true,θ)[2](1)[1]


        ## Struct 2
        hp = GPSSM.GPSpec(
            1,
            NamedTuple(),
            [(l= 2.9, l2 =2.0)],
            typeof(NamedTuple())[NamedTuple()],
            0.1, #We start with a huge amount of noise....
            (A =Wishart(5, Matrix(Diagonal(ones(5)))),b = zeros(5), S = Diagonal(ones(5,)));
            trainQ = "F",
            trainθxkernel = [(l = "F", l2 = "F")],
            trainθy = (A = "TP",b = "F", S = "F"))

        varinf = GPSSM.generate_vi(hp)
        A = varinf.metadata.A_xy.vals[:]
        space = GPSSM._getspace(varinf, GPSSM.TrainableSelector)

        DynamicPPL.link!(varinf,GPSSM.TrainableSelector)

        space = GPSSM._getspace(varinf, GPSSM.TrainableSelector)
        θ = varinf[GPSSM.TrainableSelector]
        x = rand(5,)
        y = rand(5,)

        function f_ytrue(θ)
            mu = inv(Bijectors.PDBijector())(reshape(θ,(5,5))) *x
            logpdf(MvNormal(mu, 1.0), y)
        end

        function f_y(θ)
            new_varinfo = GPSSM.NewVarInfo(varinf, space, θ )
            argsy = GPSSM.get_distθy_args(new_varinfo, hp)
            dist,args =  GPSSM.get_distxy_and_args(LinearInstance(), x; argsy...)
            return logpdf(dist(args...), y)
        end

        # test kernel_args
        @test Zygote.pullback(f_y,θ)[2](1)[1] ==  Zygote.pullback(f_ytrue,θ)[2](1)[1]

        ## Struct 3
        hp = GPSSM.GPSpec(
            1,
            NamedTuple(),
            [(l= 2.9, l2 =2.0)],
            typeof(NamedTuple())[NamedTuple()],
            0.1, #We start with a huge amount of noise....
            (A =Wishart(5, Matrix(Diagonal(ones(5)))),b = zeros(5), S = Diagonal(ones(5,)));
            trainQ = "F",
            trainθxkernel = [(l = "F", l2 = "F")],
            trainθy = (A = "T",b = "F", S = "F"))

        varinf = GPSSM.generate_vi(hp)
        A = varinf.metadata.A_xy.vals[:]
        space = GPSSM._getspace(varinf, GPSSM.TrainableSelector)

        DynamicPPL.link!(varinf,GPSSM.TrainableSelector)

        space = GPSSM._getspace(varinf, GPSSM.TrainableSelector)
        θ = varinf[GPSSM.TrainableSelector]
        x = rand(5,)
        y = rand(5,)

        function f_ytrue(θ)
            mu = reshape(θ,(5,5))*x .* 1.0 # The .* 1.0 is for the type of mu!
            logpdf(MvNormal(mu, 1.0), y)
        end

        function f_y(θ)
            new_varinfo = GPSSM.NewVarInfo(varinf, space, θ )
            argsy = GPSSM.get_distθy_args(new_varinfo, hp)
            dist,args =  GPSSM.get_distxy_and_args(LinearInstance(), x; argsy...)
            return logpdf(dist(args...), y)
        end

        # test kernel_args
        @test Zygote.pullback(f_y,θ)[2](1)[1] ==  Zygote.pullback(f_ytrue,θ)[2](1)[1]

    end
end
