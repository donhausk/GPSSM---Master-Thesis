


"""
    get_kernels(instance::GPSSMInstance, θ_kernel)


This method must be imported and overloaded. This method takes the kernel
hyper parameters θ_kernel specified in the GP specification as inputs and returns
a tuple of Stheno kernels for every dimension.



Example:

```julia

function GPSSM.get_kernels(args)
    (Stheno.Scaled(args[1][:variance][1], Stheno.stretch(Stheno.Matern32(), args[1][:length])),)
end

function GPSSM.get_kernels(args)
    (Stheno.Scaled(args[1][:l6][1] ,Stheno.stretch(Stheno.Matern32(), [args[1][:l1]...,args[1][:l2]...,args[1][:l3]..., args[1][:l4]...,args[1][:l5]...])),
    Stheno.Scaled(args[2][:l6][1] ,Stheno.stretch(Stheno.Matern32(), [args[2][:l1]...,args[2][:l2]...,args[2][:l3]..., args[2][:l4]...,args[2][:l5]...])),
    Stheno.Scaled(args[3][:l6][1] ,Stheno.stretch(Stheno.Matern32(), [args[3][:l1]...,args[3][:l2]...,args[3][:l3]..., args[3][:l4]...,args[3][:l5]...])),
    Stheno.Scaled(args[4][:l6][1] ,Stheno.stretch(Stheno.Matern32(), [args[4][:l1]...,args[4][:l2]...,args[4][:l3]..., args[4][:l4]...,args[4][:l5]...])),)
end

```

"""

function get_kernels(instance::GPSSMInstance)
    error("This function must be spcified!")
end

"""
    get_means(θ_mean)


This method must be imported and overloaded. This method takes the mean
hyper parameters θ_mean specified in the GP specification as inputs and returns
a tuple of Stheno mean functions for every dimension.



Example:

```julia

function GPSSM.get_means(args)
    (Stheno.ZeroMean(),Stheno.ZeroMean(),Stheno.ZeroMean(),Stheno.ZeroMean())

end
```

"""



@inline function get_means(instance::GPSSMInstance)
    error("This function must be spcified!")
end


"""
 	get_distxy_and_args(X::Vector; θ_xyargs...)

This method must be imported and overwritten. This method takes the latent state X
and the hyper parameters θ_xyargs specified in the GP specification as inputs and returns
a distribution and a set of arguments for this distribution. For example, (Normal, (x, 0.1)).
This is a bit confusing but it is imporant because this way we have the maximal freedom while still being
able to take care for positive definiteness.


Example:

```julia

function GPSSM.get_distxy_and_args(X::Vector; A, b,S)
    return GPSSM.Zygote_MvNormal, (A*X .+ b, S)
end

function GPSSM.get_distxy_and_args(X::Vector; l)
    x1 =X[1]
    x2 = X[2]
    yt = x1/(1.0 +p4*sin(x2)+p5*sin(x1))
    return GPSSM.Zygote_MvNormal, ([yt], ones(1,1)*l)
end

```

"""

@inline function get_distxy_and_args(instance::GPSSMInstance, x::AbstractVector)
    error("This function must be specified!")
end


"""
 	get_distxy_and_args(X::Vector; θ_xyargs...)

Same as for get_distxy_and_args.


Example:

```julia

GPSSM.get_distx0_and_args() = (GPSSM.Zygote_MvNormal, ([0.0], 1.0))

```

"""

@inline function get_distx0_and_args(instance::GPSSMInstance)
    error("This function must be specified!")
end




# The type of the latent states
#This should also be changed...
@inline function get_latent_type()
    return Float64
end

# initialization of X
@inline function get_X_Init(instance::GPSSMInstance, sizes)
	X_init = rand(Uniform(),(sizes[1], sizes[2]+1, sizes[3])) # We include the first state...
end

# In case that the emission is gaussian, we need to extract the parameters A, b and S
@inline function get_gaussian_em(instance::GPSSMInstance, args_θy)
	return (A = args_θy[:A], b = args_θy[:b], S= args_θy[:S])
end

@inline add_jitter(instance::GPSSMInstance ) = true
@inline get_jitter_level(instance::GPSSMInstance ) = 1e-8








"""

    ARDInstance{N, K, M, R, EM}

## Fields
    - emission:                                 An ARDEmission
    - kernel:                                   The Stheno Kernel
    - mean:                                     The Stheno Mean function
    - Q_init:                                   The initial Q
    - Z_Ind::Union{Nothing,AbstractArray}       The inducing points if existing.
    - trainkernel::Bool                         Indicate if variable should be trained
    - trainmean::Bool                           Indicate if variable should be trained
    - traininit::Bool                           Indicate if variable should be trained
    - trainQ::Bool                              Indicate if variable should be trained
    - trainZ_Ind::Bool                          Whether to train the inducing points if existent.
    - init_scale                                The initial scale
    - init_var                                  The initial variance
This is an N-dimensional ARD-GP with a ARDEmission.

"""
struct  ARDInstance{N,NY ,NC, K, M , EM} <: GPSSMInstance where{K<:Stheno.Kernel, M<: Stheno.MeanFunction, R <:Real, EM<:GPSSMEmission}
    emission::EM
    kernel::K
    mean::M
    init_noise
    Q_init
    Z_Ind::Union{Nothing,AbstractArray}
    trainkernel::Bool
    trainmean::Bool
    traininit::Bool
    trainQ::Bool
    trainZ_Ind::Bool
    init_scale
    init_var
end


ARDInstance(emission::GPSSMEmission, dim::Integer = 1, dimy::Integer= 1, dimc::Integer = 0 ;
    kernel::Stheno.Kernel = Stheno.Matern32(),
    mean::Stheno.MeanFunction = Stheno.ZeroMean(),
    init_noise = 1.0, #can also be a distribution for the fully adapted case...
    Q_init = 1.0,
    Z_Ind::Union{Nothing,AbstractArray} = nothing,
    trainkernel::Bool = true,
    trainmean::Bool = true,
    traininit::Bool = false,
    trainQ::Bool= true,
    trainZ_Ind::Bool = true,
    init_scale = DEFAULT_VAR_SCALE_VALUES[:scale],
    init_var = DEFAULT_VAR_SCALE_VALUES[:var]
    ) = ARDInstance{dim,dimy, dimc, typeof(kernel), typeof(mean), typeof(emission)}(
        emission, kernel, mean, init_noise, Q_init, Z_Ind, trainkernel, trainmean, traininit, trainQ,trainZ_Ind, init_scale,init_var )


"""

Overwrite the get_kernels, get_means, get_distxy_and_args and get_distx0_and_args functions

"""
# Zygote trick
@nograd get_sym_ardinstance(i::Integer) = Symbol("d"*string(i))

@inline @generated function get_kernels(instance::ARDInstance{N, NY ,NC, K, M , EM}, args) where {N, NY ,NC, K <: Stheno.Kernel, M <: Stheno.MeanFunction, EM}
    exprs = []
    for i = 1:N
        push!(exprs, :(Stheno.scale(Stheno.stretch(instance.kernel, args[get_sym_ardinstance($i)][:scale]), args[get_sym_ardinstance($i)][:var])))
    end
    return :($(exprs...),)
end


@inline @generated function get_means(instance::ARDInstance{N, NY ,NC, K, M , EM}, args) where {N,NY ,NC, K <: Stheno.Kernel, M <: Stheno.ZeroMean, EM}
    exprs = []
    for i = 1:N
        push!(exprs, :(Stheno.ZeroMean()))
    end
    return :($(exprs...),)
end
@inline function GPSSM.get_distx0_and_args(instance::ARDInstance{N, NY ,NC, K, M , EM}, qx0 ::Real) where {N,NY ,NC, K, M, EM }
     (GPSSM.Zygote_MvNormal, (zeros(N), qx0))
end
@inline function GPSSM.get_distx0_and_args(instance::ARDInstance{N, NY ,NC,K, M , EM}, qx0::AbstractVector) where {N,NY ,NC, K, M, EM }
     (GPSSM.Zygote_MvNormal, (zeros(N), Diagonal(qx0)))
end
@inline function GPSSM.get_distx0_and_args(instance::ARDInstance{N, NY ,NC, K, M , EM}, qx0::AbstractMatrix) where {N,NY ,NC, K, M, EM}
     (GPSSM.Zygote_MvNormal, (zeros(N), qx0))
end







"""
     LinearGaussianEmission{AT,BT, ST}

## Fields
     - A:             A*xt +b
     - b:             A*xt +b
     - S:             Variance of the emission
     - trainA:        Indicate if variable should be trained
     - trainb:        Indicate if variable should be trained
     - trainS:        Indicate if variable should be trained

The LinearGaussianEmission with p(y_t |xt ) = N(A*xt+ bt, S)
Elements of the Linear Gaussian Emission are either. We need to add typing here...



"""
struct LinearGaussianEmission{AT, BT, ST} <:GPSSMEmission
    A::AT
    b::BT
    S::ST
    trainA::Bool
    trainb::Bool
    trainS::Bool
end


function get_emission_spec(emission::LinearGaussianEmission)
    trainθy = (A = emission.trainA ? "T" : "F", b = emission.trainb ? "T" : "F" ,S = emission.trainS ? "TP" : "F")
    θy = (A = emission.A, b = emission.b, S = emission.S)
    return θy, trainθy
end



LinearGaussianEmission(A,b,S;trainA::Bool = false, trainb::Bool = false, trainS::Bool = false) = LinearGaussianEmission(A,b,S,trainA,trainb,trainS)
LinearGaussianEmission(dimx::Integer, noise::Real, dimy::Union{Nothing, Integer} = nothing ;trainA::Bool = false, trainb::Bool = false, trainS::Bool = false) =
    LinearGaussianEmission(Matrix{typeof(noise)}(I,dimx, dimy === nothing ? dimx : dimy), zeros(dimy === nothing ? dimx : dimy),noise .* ones(dimy === nothing ? dimx : dimy),trainA,trainb,trainS)


"""
    LinearGaussianInstance(dim::Integer, dimy::Integer, dimc::Integer,
        kernel::Stheno.Kernel = Stheno.Matern32(),
        mean::Stheno.MeanFunction = Stheno.ZeroMean(),
        A = nothing, b=nothing, S = nothing,
        emission_noise=1.0,
        init_noise = 1.0,
        Q_init::Union{AbstractVector, Real} = 1.0,
        Z_Ind::Union{Nothing,AbstractArray} = nothing,
        trainkernel::Bool = true,
        trainmean::Bool = true,
        traininit::Bool = false,
        trainZ_Ind::Bool = true,
        trainQ::Bool= true,
        trainA::Bool = false,
        trainb::Bool = false,
        trainS::Bool = false,
        init_scale = DEFAULT_VAR_SCALE_VALUES[:scale],
        init_var = DEFAULT_VAR_SCALE_VALUES[:var]
    )

## Arguments
    - dim::Integer:                  The dimension of the latent states
    - dimy::Integer:                 The dimension of the observations
    - dimc::Integer:                 The dimension of the controls
    - kernel::Stheno.Kernel:         The Stheno Kernel, default Matern32
    - mean::Stheno.MeanFunction:     The Stheno Mean function, default ZeroMean
    - A = nothing:                   y = N(A x +b, S), default the identity Matrix
    - b=nothing:                     y = N(A x +b, S), default a zero vector
    - S = nothing:                   y = N(A x +b, S), default the identity time emission_noise
    - emission_noise=1.0:            If A,b and S are not specified, we set y = N(x,emission_noise* I)
    - init_noise = 1.0:              x_0 = N(0.0, init_noise), where init_noise is either a real, vector or matrix.
    - Q_init:                        The Q initialisation
    - Z_Ind = nothing:               The inducing points
    - trainkernel::Bool = true:        trainkernel
    - trainmean::Bool = true:          trainmean
    - traininit::Bool = false:         train init
    - trainQ::Bool= true:              trainQ
    - trainA::Bool = false:            trainA
    - trainb::Bool = false:            trainb
    - trainS::Bool = false:            trainS
    - trainZ_Ind::Bool = true:          trainZ_Ind
    - init_scale = DEFAULT_VAR_SCALE_VALUES[:scale]:      The initial scale
    - init_var = DEFAULT_VAR_SCALE_VALUES[:var:           The initial variance

)

"""

function LinearGaussianInstance(dim::Integer = 1, dimy::Integer = 1, dimc::Integer = 0;
    kernel::Stheno.Kernel = Stheno.Matern32(),
    mean::Stheno.MeanFunction = Stheno.ZeroMean(),
    A = nothing, b=nothing, S = nothing,
    emission_noise=1.0,
    init_noise = 1.0,
    Q_init= 1.0,
    Z_Ind::Union{Nothing,AbstractArray} = nothing,
    trainkernel::Bool = true,
    trainmean::Bool = true,
    traininit::Bool = false,
    trainQ::Bool= true,
    trainZ_Ind::Bool = true,
    trainA::Bool = false,
    trainb::Bool = false,
    trainS::Bool = false,
    init_scale = DEFAULT_VAR_SCALE_VALUES[:scale],
    init_var = DEFAULT_VAR_SCALE_VALUES[:var]

)
    if A === nothing && b=== nothing && S === nothing
        emission = LinearGaussianEmission(dim, emission_noise, dimy;  trainA=trainA, trainb=trainb, trainS = trainS)
    else
        @assert A !== nothing && b !== nothing && S !== nothing "[Instance Initi] A,b,S either should be all nothing or none..."
        emission = LinearGaussianEmission(A,b,S; trainA=trainA, trainb=trainb, trainS = trainS)
    end

    ARDInstance(emission, dim ,dimy, dimc; kernel = kernel, mean = mean,  init_noise = init_noise, Q_init = Q_init,
        Z_Ind = Z_Ind,trainkernel = trainkernel, trainmean = trainmean,
        traininit = traininit, trainQ = trainQ,trainZ_Ind= trainZ_Ind, init_scale = init_scale, init_var = init_var)
end



@inline function GPSSM.get_distxy_and_args(instance::ARDInstance{N, NY ,NC, K, M , EM}, X::AbstractVector, A::AbstractMatrix, b::AbstractVector, S::AbstractVector) where {N,NY ,NC, K, M, EM <:LinearGaussianEmission}
    return GPSSM.Zygote_MvNormal, (A*X .+ b, Diagonal(S))
end
@inline function GPSSM.get_distxy_and_args(instance::ARDInstance{N, NY ,NC, K, M , EM}, X::AbstractVector, A::AbstractMatrix, b::AbstractVector, S::AbstractMatrix) where {N, NY ,NC,K, M, EM <:LinearGaussianEmission}
    return GPSSM.Zygote_MvNormal, (A*X .+ b, Symmetric(S))
end



struct NarendraLiEmission <:GPSSMEmission
    noise::Float64
    p4::Float64
    p5::Float64
    trainsigma::Bool
end


function get_emission_spec(emission::NarendraLiEmission)
    trainθy = (noise = emission.trainsigma ? "TP" : "F", )
    θy = (noise = emission.noise, )
    return θy, trainθy
end

NarendraLiEmission(sigma::Float64=1.0, trainsigma=true) = NarendraLiEmission(sigma, 0.52, 0.48, trainsigma)



"""
    NarendraLiInstance(;
        kernel::Stheno.Kernel = Stheno.Matern32(),
        mean::Stheno.MeanFunction = Stheno.ZeroMean(),
        emission_noise=1.0,
        init_noise = 1.0,
        Q_init::Union{AbstractVector, Real} = 1.0,
        Z_Ind::Union{Nothing,AbstractArray} = nothing,
        trainkernel::Bool = true,
        trainmean::Bool = true,
        traininit::Bool = false,
        trainZ_Ind::Bool = true,
        trainQ::Bool= true,
        trainemission::Bool = true

        init_scale = DEFAULT_VAR_SCALE_VALUES[:scale],
        init_var = DEFAULT_VAR_SCALE_VALUES[:var]
    )

## Arguments
    - kernel::Stheno.Kernel:         The Stheno Kernel, default Matern32
    - mean::Stheno.MeanFunction:     The Stheno Mean function, default ZeroMean
    - emission_noise=1.0:            If A,b and S are not specified, we set y = N(x,emission_noise* I)
    - init_noise = 1.0:              x_0 = N(0.0, init_noise), where init_noise is either a real, vector or matrix.
    - Q_init:                        The Q initialisation
    - Z_Ind = nothing:               The inducing points
    - trainkernel::Bool = true:        trainkernel
    - trainmean::Bool = true:          trainmean
    - traininit::Bool = false:         train init
    - trainQ::Bool= true:              trainQ
    - trainemission::Bool =false:      trainemission
    - trainZ_Ind::Bool = true:          trainZ_Ind
    - init_scale = DEFAULT_VAR_SCALE_VALUES[:scale]:      The initial scale
    - init_var = DEFAULT_VAR_SCALE_VALUES[:var:           The initial variance

)

"""



function NarendraLiInstance(;
    kernel::Stheno.Kernel = Stheno.Matern32(),
    mean::Stheno.MeanFunction = Stheno.ZeroMean(),
    emission_noise=1.0,
    init_noise = 1.0,
    Q_init = 1.0,
    Z_Ind::Union{Nothing,AbstractArray} = nothing,
    trainkernel::Bool = true,
    trainmean::Bool = true,
    traininit::Bool = false,
    trainQ::Bool= true,
    trainZ_Ind::Bool = true,
    trainemission::Bool = false,
    init_scale = DEFAULT_VAR_SCALE_VALUES[:scale],
    init_var = DEFAULT_VAR_SCALE_VALUES[:var]
)
    emission = NarendraLiEmission(emission_noise, trainemission)
    ARDInstance(emission, 2 ,1, 1; kernel = kernel, mean = mean,  init_noise = init_noise, Q_init = Q_init,
        Z_Ind = Z_Ind,trainkernel = trainkernel, trainmean = trainmean,
        traininit = traininit, trainQ = trainQ,trainZ_Ind= trainZ_Ind, init_scale = init_scale, init_var = init_var)
end


function GPSSM.get_distxy_and_args(instance::ARDInstance{N, NY ,NC, K, M , EM}, X::AbstractVector, noise::Union{Real,AbstractVector, AbstractMatrix}) where {N, NY ,NC,K, M, EM <:NarendraLiEmission}
    x1 =X[1]
    x2 = X[2]
    yt = x1/(1+instance.emission.p4*sin(x2)+instance.emission.p5*sin(x1))
    return GPSSM.Zygote_MvNormal, ([yt], ones(1,1) .* noise)
end



"""
    save_instance(instance)

## Arguments
     -   instace::GPSSMInstance     A GPSSM Instance

Save the instance.


"""

function save_instance(instance::GPSSMInstance)
    return string(typeof(instance))
end


"""
    restore_instance(args)

## Arguments
     -   args     The return value of save_instance

Restores the instance.


"""
function restore_instance(args)
    return eval(Meta.parse(args))()
end



"""
    GPSpec(instance)

## Arguments
    - instance: ARDInstance

Crete a GPSpec from an Instance.

"""

function GPSpec(instance::GPSSMInstance)
    error("Please specify this function or use GPSpec directly")
end

# Zero Mean, Linear Gaussian Emission
function get_gpspec_from_instance(instance::ARDInstance{N,NY,NC, K, M,EM}) where{N, NY ,NC,K ,M<:Stheno.ZeroMean, EM<:GPSSMEmission}

    Z_Ind = instance.Z_Ind === nothing ? zeros(N,0) : instance.Z_Ind
    trainQ = instance.trainQ ? "TP" : "F"
    trainθx0 = instance.traininit ? ( qx0 = "TP", ) : ( qx0 = "F", )
    trainθxkernel  = instance.trainkernel ? [(var = "TP", scale = "TP") for _ = 1:N] : [(var = "F", scale = "F") for _ = 1:N]
    trainθxmean = [NamedTuple()] # Zero Mean


    θy, trainθy = get_emission_spec(instance.emission)


    trainZ_Ind = instance.Z_Ind == nothing ? "F" : ( instance.trainZ_Ind ? "T" : "F")
    init_scale = typeof(instance.init_scale) <: Real ?  instance.init_scale.* ones(N+NC) : instance.init_scale

    θxkernel  = [(var =instance.init_var, scale = init_scale ) for _ = 1:N]
    θx0 = (qx0 = instance.init_noise, )

    θxmean = [NamedTuple()]
    Q = instance.Q_init


    return GPSpec(N,θx0, θxkernel, θxmean, Q, θy, Z_Ind; trainZ_Ind = trainZ_Ind, trainθy = trainθy, trainθxmean = trainθxmean, trainθxkernel = trainθxkernel, trainθx0 = trainθx0 , trainQ = trainQ)
end
