
"""
 	GPSpec



## Arguments

    - LatendDim::Integer:           Dimension of the latent states

    - θx0::NTθx0:                       NamedTuple containing parameters specifying p(x0)
    - θxkernel::NTθxkernel:             Vector of NamedTuple containing Kernel hyper parameters. Note that for every latent dimesion
                                        there is a different set of kernels. If the vector has only one element, this set of parameters
                                        will be duplicated for every dimension.
    - θxmean::NTθxmean:                 Same as kernel parameters but for the mean functions
    - Q::T:                             Vector of Process noises - This is the only one which is not a NamedTuple but a Vector
    - θy::NTθy:                         NamedTuple of emission parameters
    - Z_Ind::TZ:                        Locations of the inducing points, zeros(dim, 0) if there are no inducing points



    - trainθx0::NamedTuple
    - trainθxkernel::Vector{NamedTuple}
    - trainθxmean::Vector{NamedTuple}
    - trainQ::String
    - trainθy::NamedTuple
    - trainZ_Ind::String


## Usage
'''Julia

GPSpec(
        ndim,
        NamedTuple(),
        [(l1 =1.0, l2 =1.0,l3 =1.0,l4 = 1.0,l5=1.0, l6=1.0)],
        [NamedTuple()],
        0.5, #some what arbitrary...
        (A = ones(1,4) ,b=[0.0], S=0.2*ones(1,1)),
        rand(Uniform(), 5, 100, 4);
        trainθxkernel = [(l1 ="TP", l2 ="TP",l3 ="TP",l4 = "TP",l5 = "TP",l6="TP")],
        trainQ = "TP",
        trainθy = (S="TP",))

GPSpec(
        7,
        NamedTuple(),
        [param for param in params[:kernel_args]],
        typeof(NamedTuple())[NamedTuple()],
        Process_noise[1],
        params[:args_θy],
        Z_Ind;
        trainθxkernel = [(var = "F", scale = "F")],
        trainQ = "F",
        trainθy = (A = "F",b="F", S="F"))

'''

# Any of the θx arguments are NamedTuples or Vector of NamedTuples. Every parameter is either a distribution or a fixed value ( real, vector or matrix)

# Any of the trainθx arguments are NamedTuples or Vector of NamedTuples, specifiying which variables are positive definite or should not be trained
# in case of a non fully bayesian optimiser. By default, all parameters are set to "T"

# Additional arguments not needed and specified by the constructor.
#     # Keys are NamedTuple where the size and a unique name for each parameter is stored. This is important to prevent named conflicts in the latter on.
#     # The sizes are important to reconstruct the parameters after stroing them as vectors in a VarInfo struct.
#
#     - θx0keys::NTθx0keys
#     - θxkernelkeys::NTθxkernlkeys       The names are expaned by _kernel_d(dimension) in order to prevent name conflicts
#     - θxmeankeys::NTθxmeankeys          The names are expaned by _means_d(dimension) in order to prevent name conflicts
#     - θykeys::NTθykeys
#
#
# This struct defines the hyper parameters of the GP. In case that there are no inducing points, Z_Inds is solely a array of
# zero size. The hyper parameters are stroed in Namedtuples and for the kernel and mean hyper parameters in a Vector of Namedtuple
# where for every dimension a seperate NamedTuple is stored. The hyper parameters are either passed as fixed values ( For the Fully
# Bayesian Case this means that the Hyper Parameters are fixed) or as Distributions (For the Fully Bayesian Setting only these are
# trained).
# For the optimisation Algorithms (EM and VI) there exists also a train Namedtuple for any set of hyper parameters. Any hyper parameter
# is by default marked as 'T' which means that they are optimised. In case that these hyper parameters should not be optmisied 'F' or
# these hyper pareameters should stay positive definite 'TP', the belonging string has to be passed as argument.

"""


# Most of these arguments are self explaining from the definition of the constructor.
struct GPSpec{
    NTθx0,
    NTθx0keys,
    NTθxkernel,
    NTθxkernlkeys,
    NTθxmean,
    NTθxmeankeys,
    T,
    NTθy,
    NTθykeys,
    TZ
    }<: AbstractGPSpec where {
    NTθx0 <: NamedTuple,
    NTθx0keys <: NamedTuple,
    NTθxkernel<: NamedTuple,
    NTθxkernlkeys<: NamedTuple,
    NTθxmean<: NamedTuple,
    NTθxmeankeys <:NamedTuple,
    NTθy<: NamedTuple,
    NTθykeys<: NamedTuple,
    T<:Union{Distribution, Vector, Real},
    TZ<:Union{Union{AbstractArray{<:Real,3}, MultivariateDistribution}}
    }
    # These are often called and must be type stable!!
    θx0::NTθx0
    θx0keys::NTθx0keys
    θxkernel::NTθxkernel
    θxkernelkeys::NTθxkernlkeys
    θxmean::NTθxmean
    θxmeankeys::NTθxmeankeys
    Q::T
    θy::NTθy
    θykeys::NTθykeys
    Z_Ind::TZ

    # We only need this once, therefore, this must not be type stable!
    # The train is only relevant if we use gradient optimization for training!
    trainθx0::NamedTuple
    trainθxkernel::Vector{NamedTuple}
    trainθxmean::Vector{NamedTuple}
    trainQ::String # We know this shape!
    trainθy::NamedTuple
    trainZ_Ind::String
end




function GPSpec(
    LatendDim::Integer,
    specθx0::NamedTuple,
    specθxkernel::Vector{Tk},
    specθxmean::Vector{Tm},
    Q::Union{Distribution, Matrix, Vector, Real},
    specθy::NamedTuple,
    Z_Ind::TZ;
    trainθx0::NamedTuple = NamedTuple(),
    trainθxkernel::Vector{<:NamedTuple} = NamedTuple[],
    trainθxmean::Vector{<:NamedTuple} = NamedTuple[],
    trainQ::String = "TP", # We know this shape!
    trainθy::NamedTuple = NamedTuple(),
    trainZ_Ind::String = "T",
) where {Tm <:NamedTuple, Tk<:NamedTuple, TZ<:Union{AbstractArray{<:Real,2}, MultivariateDistribution}}
    Ndim = LatendDim
    names = Tuple([Symbol("d$i") for i in 1:Ndim])
    # Some sanety chaecks
    @assert length(specθxkernel) == 1 || length(specθxkernel) == Ndim "[HuperParameterInit] The specifications for θxkernel must have either length 1 or ndim"
    @assert length(specθxmean) == 1 || length(specθxmean) == Ndim "[HuperParameterInit] The specifications for θxmean must have either length 1 or ndim"
    @assert length(trainθxkernel) == 0  || length(trainθxkernel) == 1 || length(trainθxkernel) == Ndim "[HuperParameterInit] The specifications for trainθxkernel must have either length 1 or ndim"
    @assert length(trainθxmean) == 0  || length(trainθxmean) == 1 || length(trainθxmean) == Ndim "[HuperParameterInit] The specifications for trainθxmean must have either length 1 or ndim"

    # If the length is one, we simply replicate the Named Tuple for every input dimension. In this code block, we are extracting the sizes of our hyper parameters and storing unique names in the
    # Keys NamedTuples.
    if length(specθxkernel) == 1
        valuesθxkernel = Tuple([specθxkernel[1] for _ in 1:Ndim])
        keysθxkernel = Tuple([NamedTuple{Tuple([Symbol(string(key)*"_kernel_d$i") for key in keys(specθxkernel[1])])}(
            Tuple([typeof(val)<:Distribution ? size(rand(val)) : size(val) for val in specθxkernel[1]])) for i in 1:Ndim])
    else
        valuesθxkernel = Tuple(specθxkernel)
        keysθxkernel = Tuple([NamedTuple{Tuple([Symbol(string(key)*"_kernel_d$i") for key in keys(specθxkernel[i])])}(
            Tuple([typeof(val)<:Distribution ? size(rand(val)) : size(val) for val in specθxkernel[i]])) for i in 1:Ndim])
    end
    if length(specθxmean) == 1
        valuesθxmean = Tuple([specθxmean[1] for _ in 1:Ndim])
        keysθxmean = Tuple([NamedTuple{Tuple([Symbol(string(key)*"_mean_d$i") for key in keys(specθxmean[1])])}(
            Tuple([typeof(val)<:Distribution ? size(rand(val)) : size(val) for val in specθxmean[1]])) for i in 1:Ndim])
    else
        valuesθxmean = Tuple(specθxmean)
        keysθxmean = Tuple([NamedTuple{Tuple([Symbol(string(key)*"_mean_d$i") for key in keys(specθxmean[i])])}(
            Tuple([typeof(val)<:Distribution ? size(rand(val)) : size(val) for val in specθxmean[i]])) for i in 1:Ndim])
    end

    if length(trainθxkernel) == 1 && Ndim != 1
        tθxkernel = [trainθxkernel[1] for i =1:Ndim] #Simply duplicate
    else
        tθxkernel = trainθxkernel
    end
    if length(trainθxmean) == 1 && Ndim != 1
        tθxmean = [trainθxmean[1] for i =1:Ndim] #Simply duplicate
    else
        tθxmean = trainθxmean
    end


    θx0keys = NamedTuple{Tuple([Symbol(string(key)*"_x0") for key in keys(specθx0)])}(
        Tuple([typeof(val)<:Distribution ? size(rand(val)) : size(val) for val in specθx0]))
    θykeys = NamedTuple{Tuple([Symbol(string(key)*"_xy") for key in keys(specθy)])}(
        Tuple([typeof(val)<:Distribution ? size(rand(val)) : size(val) for val in specθy]))


    θxkernel = NamedTuple{names}(valuesθxkernel)
    θxkernelkeys =NamedTuple{names}(keysθxkernel)
    θxmean = NamedTuple{names}(valuesθxmean)
    θxmeankeys = NamedTuple{names}(keysθxmean)
    GPSpec{typeof(specθx0),
        typeof(θx0keys),
        typeof(θxkernel),
        typeof(θxkernelkeys),
        typeof(θxmean),
        typeof(θxmeankeys),
        typeof(Q),
        typeof(specθy),
        typeof(θykeys),
        typeof(Z_Ind)
    }(specθx0, θx0keys, θxkernel, θxkernelkeys, θxmean, θxmeankeys, Q, specθy, θykeys, Z_Ind, trainθx0, tθxkernel, tθxmean, trainQ, trainθy, trainZ_Ind)
end


function GPSpec(
    LatendDim::Integer,
    specθx0::NamedTuple,
    specθxkernel::Vector{Tk},
    specθxmean::Vector{Tm},
    Q::Union{Distribution, Vector, Real},
    specθy::NamedTuple;
    trainθx0 = NamedTuple(),
    trainθxkernel=  NamedTuple[],
    trainθxmean = NamedTuple[],
    trainQ = "TP", # We know this shape!
    trainθy = NamedTuple()
) where {Tm <:NamedTuple, Tk<:NamedTuple}
    # In case that there exists no Inducing points, we simply set them to be a vector of zero dimension.
    return GPSpec(LatendDim, specθx0, specθxkernel, specθxmean, Q, specθy, zeros(LatendDim,0);
        trainθx0 = trainθx0, trainθxkernel = trainθxkernel, trainθxmean=trainθxmean , trainQ= trainQ, trainθy= trainθy, trainZ_Ind = "F")
end





"""
    generate_vi(hp, additional_params, traine_addtional_params)

## Arguments
	- hp::GPSpec:				A gp specification
	- additional_params:		Additional params like the inducing point covariance and mean
	- traind_additional_paras:	Whether to train the additional params.


Generate a Turing VarInfo struct out of the GPSpec. Any parameters which needs to be trained is marekd using TrainableSelector.
additional_params and train_additional_params allow to include further parameters.


"""




function generate_vi(hp::GPSpec; additional_params= NamedTuple(), train_additional_params = NamedTuple())
    varinfo = DynamicPPL.VarInfo()
    # First, we initilaize our x's
    dlist = hp.θx0
    nkeys = keys(hp.θx0keys)
    dtrain = hp.trainθx0
    dkeys = keys(dlist)
    for j in 1:length(dkeys)
        if typeof(dlist[dkeys[j]]) <: Distribution
            val = rand(dlist[dkeys[j]])
        else
            @assert typeof(dlist[dkeys[j]])<:Union{Real, AbstractVector{<:Real}, AbstractMatrix{<:Real}} "[HP Model] We only support subtypes of Union{Real, AbstractVector{<:Real}, AbstractMatrix{<:Real}} for the varaible $(dkeys[j])"
            val = dlist[dkeys[j]]
        end
        vn = varname(string(nkeys[j]))

        if ~haskey(varinfo,vn)
            pdist = haskey(dtrain, dkeys[j])&&dtrain[dkeys[j]] == "TP" ? get_placeholder_dist_positive(val) : get_placeholder_dist(val)
            DynamicPPL.push!(varinfo, vn , val, pdist)
            setgid!(varinfo, ConstSelector, vn)
            if ~haskey(dtrain, dkeys[j]) || dtrain[dkeys[j]]  in ["T", "TP"]
                setgid!(varinfo, TrainableSelector, vn)
            end
        else
            error("[HyperParamsInit] Hyper params must have distinct names!")
        end
    end
    # First, we initilaize our x's
    dlist = hp.θy
    nkeys = keys(hp.θykeys)
    dtrain = hp.trainθy
    dkeys = keys(dlist)
    for j in 1:length(dkeys)
        if typeof(dlist[dkeys[j]]) <: Distribution
            val = rand(dlist[dkeys[j]])
        else
            @assert typeof(dlist[dkeys[j]])<:Union{Real, AbstractVector{<:Real}, AbstractMatrix{<:Real}} "[HP Model] We only support subtypes of Union{Real, AbstractVector{<:Real}, AbstractMatrix{<:Real}} for the varaible $(dkeys[j])"
            val = dlist[dkeys[j]]
        end
        vn = varname(string(nkeys[j]))

        if ~haskey(varinfo,vn)
            pdist = haskey(dtrain, dkeys[j])&&dtrain[dkeys[j]] == "TP" ? get_placeholder_dist_positive(val) : get_placeholder_dist(val)
            DynamicPPL.push!(varinfo, vn , val, pdist)
            setgid!(varinfo, ConstSelector, vn)
            if ~haskey(dtrain, dkeys[j]) || dtrain[dkeys[j]] in ["T", "TP"]
                setgid!(varinfo, TrainableSelector, vn)
            end
        else
            error("[HyperParamsInit] Hyper params must have distinct names!")
        end
    end
    for i= 1:length(hp.θxkernel)
        dlist = hp.θxkernel[Symbol("d$i")]
        dkeys = keys(dlist)
        nkeys = keys(hp.θxkernelkeys[Symbol("d$i")])
        dtrain = length(hp.trainθxkernel) == 0 ? NamedTuple() : hp.trainθxkernel[i]

        for j in 1:length(dkeys)
            if typeof(dlist[dkeys[j]]) <: Distribution
                val = rand(dlist[dkeys[j]])
            else
                @assert typeof(dlist[dkeys[j]])<:Union{Real, AbstractVector{<:Real}, AbstractMatrix{<:Real}} "[HP Model] We only support subtypes of Union{Real, AbstractVector{<:Real}, AbstractMatrix{<:Real}} for the varaible $(dkeys[j])"
                val = dlist[dkeys[j]]
            end
            vn = varname(string(nkeys[j]))
            if ~haskey(varinfo,vn)
                pdist = haskey(dtrain, dkeys[j])&&dtrain[dkeys[j]] == "TP" ? get_placeholder_dist_positive(val) : get_placeholder_dist(val)
                DynamicPPL.push!(varinfo, vn , val, pdist)
                setgid!(varinfo, ConstSelector, vn)
                if ~haskey(dtrain, dkeys[j]) || dtrain[dkeys[j]]  in ["T", "TP"]
                    setgid!(varinfo, TrainableSelector, vn)
                end
            else
                error("[HyperParamsInit] Hyper params must have distinct names!")
            end
        end
        dlist = hp.θxmean[Symbol("d$i")]
        nkeys = keys(hp.θxmeankeys[Symbol("d$i")])
        dkeys = keys(dlist)
        dtrain = length(hp.trainθxmean) == 0 ? NamedTuple() : hp.trainθxmean[i]
        for j in 1:length(dkeys)
            if typeof(dlist[dkeys[j]]) <: Distribution
                val = rand(dlist[dkeys[j]])
            else
                @assert typeof(dlist[dkeys[j]])<:Union{Real, AbstractVector{<:Real}, AbstractMatrix{<:Real}} "[HP Model] We only support subtypes of Union{Real, AbstractVector{<:Real}, AbstractMatrix{<:Real}} for the varaible $(dkeys[j])"
                val = dlist[dkeys[j]]
            end
            vn = varname(string(nkeys[j]))
            if ~haskey(varinfo,vn)
                pdist = haskey(dtrain, dkeys[j])&&dtrain[dkeys[j]] == "TP" ? get_placeholder_dist_positive(val) : get_placeholder_dist(val)
                DynamicPPL.push!(varinfo, vn , val, pdist)
                setgid!(varinfo, ConstSelector, vn)
                if ~haskey(dtrain, dkeys[j]) || dtrain[dkeys[j]]  in ["T", "TP"]
                    setgid!(varinfo, TrainableSelector, vn)
                end
            else
                error("[HyperParamsInit] Hyper params must have distinct names!")
            end
        end
    end

    Q = Vector{Real}(undef, length(hp.θxkernel))
    if typeof(hp.Q) <: UnivariateDistribution
        for i= 1:length(hp.θxkernel)
            Q[i]  =rand(hp.Q)
        end
    elseif typeof(hp.Q) <: MultivariateDistribution
        Q  = rand(hp.Q)
    elseif typeof(hp.Q) <: Real
        # We have already checked that
        #@assert typeof(dlist[dkeys[j]])<:Union{Real, AbstractVector{<:Real}, AbstractMatrix{<:Real}} "[HP Model] We only support subtypes of Union{Real, AbstractVector{<:Real}, AbstractMatrix{<:Real}} for the varaible $(dkeys[j])"
        for i = 1:length(hp.θxkernel)
            Q[i] = hp.Q
        end
    else
        @assert typeof(hp.Q) <: AbstractVector "[HyperParamInit] Q must be either a vector, distribution or single element"
        @assert length(hp.Q) ==length(hp.θxkernel) "[HyperParamInit] Q must have length $(sizes[1])"
        Q = hp.Q
    end
    vn = varname("Q")


    if ~haskey(varinfo,vn)
        pdist = (hp.trainQ == "TP" ? get_placeholder_dist_positive(Q) : get_placeholder_dist(Q))
        DynamicPPL.push!(varinfo, vn, Q, pdist)
        setgid!(varinfo, ConstSelector, vn)
        if hp.trainQ  in ["T", "TP"]
            setgid!(varinfo, TrainableSelector, vn)
        end
    else
        error("[HyperParamsInit] Hyper params must have distinct names!")
    end


    if hasfield(typeof(hp), :Z_Ind)  && hp.Z_Ind !== zeros(0,0,0)
        if typeof(hp.Z_Ind) <: MultivariateDistribution
            valz = rand(hp.Z_Ind,length(hp.θxkernel),1)
        else
            @assert typeof(hp.Z_Ind)<:Union{AbstractArray{<:Real,2}} "[HP Model] We only support subtypes of AbstractMatrix{<:Real} for the varaible Z_Ind"
            valz = vec(hp.Z_Ind)
        end
        vn = @varname Z_Ind
        if ~haskey(varinfo,vn)
            DynamicPPL.push!(varinfo, vn , valz, get_placeholder_dist(valz))
            setgid!(varinfo, ConstSelector, vn)
            if hp.trainZ_Ind in ["T"]
                setgid!(varinfo, TrainableSelector, vn)
            end
        else
            error("[HyperParamsInit] Hyper params must have distinct names!")
        end
    end
    for (i,param) in enumerate(additional_params)
        if typeof(param)<:AbstractArray
            val = param
        elseif typeof(param)<:Distribution
            val = rand(param)
        else
            error("[HyperParamsInit] Additional params must be either a subtype of AbstractArray or Distribution")
        end
        key = keys(additional_params)[i]
        vn = varname(string(key))
        if ~haskey(varinfo,vn)
            pdist = haskey(train_additional_params,key)&&train_additional_params[key] == "TP" ? get_placeholder_dist_positive(val) : get_placeholder_dist(val)
            DynamicPPL.push!(varinfo, vn , val, pdist)
            setgid!(varinfo, ConstSelector, vn)
            if !haskey(train_additional_params,key)||train_additional_params[key] in ["T", "TP"]
                setgid!(varinfo, TrainableSelector, vn)
            end
        else
            error("[HyperParamsInit] Hyper params must have distinct names!")
        end
    end
    # Now we have a typed varinfo!
    vari =  VarInfo{<:NamedTuple}(varinfo)
    # Now we still need to link all the trainable variables!
end





"""
    HyperParams


## Fields
    - kernel_args::NamedTuple:          NamedTuple with a NamedTuple for every dimension.
    - mean_args::NamedTuple:            NamedTuple with a NamedTuple for every dimension.
	- Q::Vector{<:Real}: 		        Vector of Q with parameters
	- args_θy::NamedTuple:              NamedTuple with parameters
    - args_θx0::NamedTuple:  	        NamedTuple with parameters
	- gpinstance::GPSSMInstance:		The GPSSM instance
	- kernels							A tuple of kernel functions
	- mean_fs 							A tuple of mean functions

This struct contains the hyper parameters shared accross all GPSSMs. The structure of the Hyper Parameters are
defined by the GPSpec.



"""



struct HyperParams{T1<:NamedTuple,T2<:NamedTuple,T3<:AbstractVector,T4<:NamedTuple,T5<:NamedTuple, I<:GPSSMInstance, K, M}
    kernel_args::T1
    mean_args::T2
    Q::T3
    args_θy::T4
    args_θx0::T5
	gpinstance::I
	kernels::K
	mean_fs::M
end

# We store the arguments using vectors in order to manipulate them.
function HyperParams(
	kernel_args,
    mean_args,
	Q,
	args_θy,
    args_θx0,
	gpinstance::GPSSMInstance
)

	#compute kernel and mena fucntions
	kernels = get_kernels(gpinstance, kernel_args)
	mean_fs = get_means(gpinstance, mean_args)
	HyperParams(kernel_args,
		mean_args,
		Q,
		args_θy,
		args_θx0,
		gpinstance,
		kernels,
		mean_fs
	)
end

"""

This is used to save the hyper param struct as namedtuple

"""
tonamedtuple(params::HyperParams) =  (kernel_args = params.kernel_args, mean_args = params.mean_args, Q=params.Q, args_θy = params.args_θy, args_θx0 = params.args_θx0, gpssminstance = save_instance(params.gpinstance))

"""
	restore_frome_namedtuple(nt)

## Arguments
	- instance::GPSSMInstance:	The GPSSM Instance
	- nt::NamedTuple: 			The named tuple of the hyper params struct generated using tonamedtuple

"""
function restore_from_namedtuple(instance::GPSSMInstance, nt::NamedTuple)
	kernels = get_kernels(instance, nt.kernel_args)
	mean_fs = get_means(instance, nt.mean_args)
	HyperParams(nt.kernel_args,
		nt.mean_args,
		nt.Q,
		nt.args_θy,
		nt.args_θx0,
		instance,
		kernels,
		mean_fs
	)
end

function restore_frome_namedtuple(nt::NamedTuple)
	instance = restore_intance(nt.gpssminstance)
	restore_from_namedtuple(instance, nt)
end
