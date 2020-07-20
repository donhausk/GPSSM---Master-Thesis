

"""
	FBModel

## Fields
	- spce::GPSpec: 			An instance of the GPSpec
	- inputs::NamedTuple:		The inputs of the GP.
	- instance::GPSSMInstance:  The instance of the GPSSM model.
The default HP Model.


The Fully Bayesian Sampler Model containing the turing samplers.

"""

struct FBModel{ALG, I, S, MX, MY, NT} <:AbstractHPFBModel where {A<:AbstractHPAlgorithm,  I<:GPSSMInstance, S<:GPSpec, NT<: NamedTuple}
    θxmodel::MX
    θymodel::MY
	spec::S
	inputs::NT
	gpinstance::I
end



function get_hp_model(
    spec::GPSpec,
	alg::ALG,
	inputs::NamedTuple,
	gpinstance::GPSSMInstance

) where ALG <: FBAlgorithm
	# We can already initilaize our models!
	lsspec = inputs[:sizes]
	Y = inputs[:observations]
	U = inputs[:controls]
	# Get the initialisation for X. We need this to have an initial set of hyper paramters.
	X = get_X_Init(gpinstance, lsspec)
	@assert lsspec[1] == size(X)[1] && lsspec[2]+1 == size(X)[2] && lsspec[3] == size(X)[3] " [GPSSM] The size of the latent states must have (dim, N+1, K) with N observations and K chains"
	@assert lsspec[2]+1 == size(U)[2] && lsspec[3] == size(U)[3] " [GPSSM] The size of the controls must have (n_controls, N+1, K) with N observations and K chains"
	@assert lsspec[2] == size(Y)[2] && lsspec[3] == size(Y)[3] " [GPSSM] The size of the observations must have (n_out, N, K) with N observations and K chains"

	# Build the turing models
	mx = θxprobfullgpmodel(spec, gpinstance, lsspec, X, U)
	my = θyprobmodel(spec, gpinstance, lsspec, X, Y)
	nt =  (Y=Y,U=U,lsspec = lsspec)
    return  FBModel{typeof(alg),  typeof(gpinstance),typeof(spec),  typeof(mx), typeof(my), typeof(nt) }(mx, my, spec, nt, gpinstance)
end



"""

We need to overwrite get_hyperparams because we have vix and viy

"""

function get_hyperparams_from_vi(vi_x::VarInfo{<:NamedTuple}, vi_y::VarInfo{<:NamedTuple}, hpm::FBModel, inputs::NamedTuple)
    args_θx0 = get_distθx0_args(vi_x, hpm.spec)
    kernel_args = get_kernel_args(vi_x, hpm.spec)
    mean_args = get_mean_args(vi_x, hpm.spec)
	# Not very nice way of doing this...
	Q = vi_x.metadata.Q.vals
    args_θy = get_distθy_args(vi_y, hpm.spec)
	hps = HyperParams(kernel_args, mean_args , Q, args_θy , args_θx0,  hpm.gpinstance)
    return hps, NamedTuple()
end







Turing.@model θxprobfullgpmodel(hp, instance, sizes, X, U, ::Type{XT}=get_latent_type()) where {XT} = begin
    # First, we initilaize our x's
    dlist = hp.θx0
    dkeys = keys(dlist)
    nkeys = keys(hp.θx0keys)
    for j in 1:length(dkeys)
        if typeof(dlist[dkeys[j]]) <: Distribution
                v ~ NamedDist(dlist[dkeys[j]], nkeys[j])
                v = nothing
        else
            @assert typeof(dlist[dkeys[j]])<:Union{Real, Vector{Real}, Matrix{Real}} "[HP Model] We only support subtypes of Union{Real, Vector{Real}, Matrix{Real}} for the varaible $(dkeys[j])"
            vi = @varinfo
            vn = varname(string(nkeys[j]))
            if ~haskey(vi,vn)
                push!(vi, vn , dlist[dkeys[j]], get_placeholder_dist(dlist[dkeys[j]]))
                setgid!(vi, ConstSelector, vn)
            end
        end
    end
    vi = @varinfo
    d0_args = get_distθx0_args(vi,hp)
    # This is important so that turing can take care of the positive definiteness etc.
    distx0, xargs = get_distx0_and_args(instance, d0_args...)
    for chn = 1:sizes[3]
        X[:, 1, chn] ~ distx0(xargs...)
    end
    for i= 1:sizes[1]
        dlist = hp.θxkernel[Symbol("d$i")]
        dkeys = keys(dlist)
        nkeys = keys(hp.θxkernelkeys[Symbol("d$i")])
        for j in 1:length(dkeys)
            if typeof(dlist[dkeys[j]]) <: Distribution
                    v ~ NamedDist(dlist[dkeys[j]], nkeys[j])
                    v = nothing
            else
                @assert typeof(dlist[dkeys[j]])<:Union{Real, Vector, Matrix} "[HP Model] We only support subtypes of Union{Real, Vector{Real}, Matrix{Real}} for the varaible $(dkeys[j])"
                vi = @varinfo
                vn = varname(string(nkeys[j]))
                if ~haskey(vi,vn)
                    push!(vi, vn , dlist[dkeys[j]], get_placeholder_dist(dlist[dkeys[j]]))
                    setgid!(vi, ConstSelector, vn)
                end
            end
        end
        dlist = hp.θxmean[Symbol("d$i")]
        nkeys = keys(hp.θxmeankeys[Symbol("d$i")])
        dkeys = keys(dlist)
        for j in 1:length(dkeys)
            if typeof(dlist[dkeys[j]]) <: Distribution
                    v ~ NamedDist(dlist[dkeys[j]], nkeys[j])
                    v = nothing
            else
                @assert typeof(dlist[dkeys[j]])<:Union{Real, Vector, Matrix} "[HP Model] We only support subtypes of Union{Real, Vector{Real}, Matrix{Real}} for the varaible $(dkeys[j])"
                vi = @varinfo
                vn = varname(string(nkeys[j]))
                if ~haskey(vi,vn)
                    push!(vi, vn , dlist[dkeys[j]], get_placeholder_dist(dlist[dkeys[j]]))
                    setgid!(vi, ConstSelector, vn)
                end
            end
        end
    end
    Q = Vector{XT}(undef, sizes[1])
    if typeof(hp.Q) <: Distributions.UnivariateDistribution
        for i= 1:sizes[1]
            Q[i]  ~ hp.Q
        end
    elseif typeof(hp.Q) <: Distributions.Distributions.MultivariateDistribution
        Q  ~ hp.Q
    elseif typeof(hp.Q) <: Real
        # We have already checked that
        #@assert typeof(dlist[dkeys[j]])<:Union{Real, Vector{Real}, Matrix{Real}} "[HP Model] We only support subtypes of Union{Real, Vector{Real}, Matrix{Real}} for the varaible $(dkeys[j])"
        vi = @varinfo
        for i = 1: sizes[1]
            vn = @varname Q[i]
            if ~haskey(vi,vn)
                push!(vi, vn , hp.Q, get_placeholder_dist(hp.Q))
                setgid!(vi, ConstSelector, vn)
            end
            Q[i] = hp.Q
        end

    else
        @assert typeof(hp.Q) == Vector "[HyperParamInit] Q must be either a vector, distribution or single element"
        @assert length(hp.Q) == sizes[1] "[HyperParamInit] Q must have length $(sizes[1])"
        Q = hp.Q
        vn = @varname Q
        if ~haskey(vi,vn)
            push!(vi, vn , hp.Q, get_placeholder_dist(hp.Q))
            setgid!(vi, ConstSelector, vn)
        end
    end

    vi = @varinfo
    # This is not type stable
    k_args = get_kernel_args(vi, hp)
    m_args = get_mean_args(vi, hp)

    kernels = get_kernels(instance, k_args)
    mean_fs = get_means(instance, m_args)
    for j = 1:sizes[3]
        xandu = vcat(X[:, 1:end-1, j],U[:, 1:end-1, j])
        for i in 1:sizes[1]
			covm = symmetric_helper_f(Kxx(instance,  kernels[i], xandu), Diagonal(ones(XT,sizes[2]) * Q[i]))
            X[i, 2:end, j] ~ MvNormal(mx(instance, mean_fs[i], xandu),  Matrix(Hermitian(covm)))
        end
    end
end


symmetric_helper_f(K, D) = Symmetric(K + D)

# this model does not depend on inducing points....
Turing.@model θyprobmodel(hp, instance,  sizes, X, Y) = begin
    # First, we initilaize our x's
    dlist = hp.θy
    dkeys = keys(dlist)
    nkeys = keys(hp.θykeys)

    for j in 1:length(dkeys)
        if typeof(dlist[dkeys[j]]) <: Distribution
                v ~ NamedDist(dlist[dkeys[j]], nkeys[j])
        else
            @assert typeof(dlist[dkeys[j]])<:Union{Real, Vector, Matrix} "[HP Model] We only support subtypes of Union{Real, Vector{Real}, Matrix{Real}} for the varaible $(dkeys[j])"
            vi = @varinfo
            vn = varname(string(nkeys[j]))
            if ~haskey(vi,vn)
                push!(vi, vn , dlist[dkeys[j]], get_placeholder_dist(dlist[dkeys[j]]))
                setgid!(vi, ConstSelector, vn)
            end
        end
    end
    vi = @varinfo
    dy_args = get_distθy_args(vi, hp)

    for i in 1:sizes[2]
        for j in 1:sizes[3]
            distxy, yargs = get_distxy_and_args(instance, X[:, i+1, j], dy_args...)
            Y[:, i, j] ~ distxy(yargs...)
        end
    end
end
