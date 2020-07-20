module Test_Runner
    using MacroTools
    using DifferentialEquations
    using ControlSystems
    using JLD2
    using LinearAlgebra
    using SymPy
    using Stheno



    abstract type ModelParameters end
    struct GaussianParameters <: ModelParameters end


    struct TestSetGen
        name::String
        f::Function
        f_obs::Function
        f_u::Function
        f_x0::Function
        latentdim::Int
        obsdim::Int
        cdim::Int
        hp::ModelParameters
    end

    function _clip(x, lim)
        return min.(max.(x, -lim), lim)
    end



    function Gaussian1D(f::Function, f_u::Function, f_x0::Function)
        f_opt(x, on, hp) = [x[1] + rand(Normal(0.0,on))]
        return TestSetGen("Gaussian_1D", f,f_opt,f_u,f_x0,1,1,length(f_u(0.0,1, GaussianParameters())), GaussianParameters() )
    end
    function Gaussian1D(f::Function)
        f_x0(hp) = rand(Normal(),1)
        f_u(x,i, hp) = zeros(0)
        return Gaussian1D(f,f_u,f_x0)
    end
    function Gaussian1D(f::Function, f_x0::Function)
        f_u(x,i, hp) = zeros(0)
        return Gaussian1D(f,f_u,f_x0)
    end

    reset_inputs!(T::TestSetGen) = reset_inputs!(T.hp)
    reset_inputs!(hp::ModelParameters) = hp
    include("run_experiments.jl")
    include("generate_data.jl")
    include("generate_from_ode.jl")


    include("Datasets/Kink.jl")
    include("Datasets/LinearGaussian.jl")
    #include("Datasets/Unicycle.jl")
    include("Datasets/NarendraLi.jl")

    include("Datasets/ControlDatasets.jl")

    dir = "/home/kongi/GPSSM/"

    storagepath = "/media/kongi/0347aef5-d154-4a37-a3a0-58fae158de2e/GPSSM/Experiments/"
    datapath = "/media/kongi/0347aef5-d154-4a37-a3a0-58fae158de2e/GPSSM/Data/"

    get_test_specifications() = (1000, 1, 1000, 3, 200)

    export  @run_experiments,
            @getelementm,
            ExpData,
            storagepath,
            datapath
end
