

"""
 	FBAlgorithm{Ax, Ay}(θxargs, θyargs, nsteps)

## Arguments
    - Ax:       The type of the θ_x and Q Turing Sampler
    - Ay:       The type of th θ_y Turing Sampler
    - θxargs:   The arguments for the θ_x and Q Turing Sampler
    - θyargs:   The arguments for the θ_y Turing Sampler
    - nsteps:   The amoutn of iterations the Turing Sampler is run in every step.

The particle Gibbs implementation of Frigola, where the emission hyper parameters are trained seperately from the rest.
Unfortunately, we can not accept the full Turing inference method. This is due to the implementation. Therefore, the
arguemtents and the Inference type have to be passed seperately.

Usage
'''Julia

HPFBAlgorithm{HMC,HMC}((ϵ=0.002,n_leapfrog=5),(ϵ=0.02,n_leapfrog=5), 5)

'''

Original paper: Bayesian Inference and Learning in Gaussian Process State-Space Models with Particle MCMC (Frigola, 2013)
"""

struct FBAlgorithm{Ax, Ay} <: AbstractFBAlgorithm where {
    I<:InferenceModel,
    Ax <:Union{Nothing,Turing.Inference.InferenceAlgorithm}, # Inference Type.
    Ay <:Union{Nothing,Turing.Inference.InferenceAlgorithm} # Inference Type.
}
    θxargs::NamedTuple # Arguemtents
    θyargs::NamedTuple # Arguemtents.
    nsteps::Int # N turing iteration for the hyper parameters sampling.
    # function FBAlgorithm{Ax, Ay}( θxargs::NamedTuple, θyargs::NamedTuple, nsptes::Int) where {Ax<:Union{Nothing,Turing.Inference.InferenceAlgorithm}, Ay<:Union{Nothing,Turing.Inference.InferenceAlgorithm}}
    #     new{FullGPInferece, Ax, Ay}( θxargs, θyargs, nsteps)
    # end
end

FBAlgorithm{Ax, Ay}( θxargs::NamedTuple, θyargs::NamedTuple) where {Ax<:Union{Nothing,Turing.Inference.InferenceAlgorithm}, Ay<:Union{Nothing,Turing.Inference.InferenceAlgorithm}} = FBAlgorithm{Ax, Ay}( θxargs, θyargs, 1)

get_inferencemodel(::FBAlgorithm) = FullGPInference()




"""
    TuringHPSampler

## Arguments
    - θxSampler::<:Union{DynamicPPL.VarInfo,Turing.Sampler}:    Turing Sampler (MCMC Kernel) for the transition noise and the gp hyper parameters. In case that all hyper parameters are fixed, we store a VarInfo struct instead.
    - θySampler::<:Union{DynamicPPL.VarInfo,Turing.Sampler}     Turing Sampler (MCMCM Kernel) for the emission hyper parameters. In case that all hyper parameters are fixed, we store a VarInfo struct instead.
    - inputs::NamedTuple                                        Inputs of the sampler (observations, controls, sizes)
    - nsteps::Int                                               Amount of iterations exectuing step function for every iteration.


# This is the fully bayesian hyper parameter sampler, which is called TuringHPSampler becasue it is implemented using Turing.

"""




struct TuringHPSampler{ALG<:AbstractFBAlgorithm, Sx<:Union{DynamicPPL.VarInfo,Turing.Sampler}, Sy<:Union{DynamicPPL.VarInfo,Turing.Sampler}}<:AbstractHPFBSampler
    alg::ALG
    θxSampler::Sx
    θySampler::Sy
    inputs::NamedTuple
    nsteps::Int
end

function Sampler(inf::AbstractFBAlgorithm, s1::Sx, s2::Sy, inputs::NamedTuple, nsteps::Int) where {Sx<:Union{DynamicPPL.VarInfo,DynamicPPL.Sampler}, Sy<:Union{DynamicPPL.VarInfo,DynamicPPL.Sampler}}
    return TuringHPSampler(inf, s1,s2, inputs, nsteps)
end


function Sampler(mdl::AbstractModel, inf::Union{FBAlgorithm{Ax, Ay}}, inputs::NamedTuple) where {
    Ax <:Union{Nothing,Turing.Inference.InferenceAlgorithm},
    Ay <:Union{Nothing,Turing.Inference.InferenceAlgorithm}
}
    # The hyper parameter models which are defined in the model section.
    θxmodel = mdl.θxmodel
    θymodel = mdl.θymodel

    # Create VarInfo for the modesl.
    viθx = VarInfo(θxmodel)
    viθy = VarInfo(θymodel)

    # Unfortunately, we must at least sample one Variable in turing, hence, we have
    # to count how many hyper parameters are trainable...
    indxs = _getidcs_nonempty(viθx, ConstSelector)
    vsym = Symbol[] # Array of symbols which are sampled
    for key in keys(indxs)
        if length(indxs[key]) == 0 # If the variable is not marked by the Constselector, it means that we are sampling it.
            push!(vsym,key)
        end
    end
    if length(vsym) == 0
        # If there are no trainable parameters, we simply set the sampler θxSampler to be the VarInfo.
        Sx = viθx
    else
        # Else, we create a Turing samplr.
        @assert Ax != Nothing "[HyperParamsInit] There must be an alogirhtm specified for Ax, since some parameters have non deterministic priors"
        # But we only sample the variables which are not mareked by the Constselector
        Sx = Turing.Sampler(Ax(inf.θxargs..., vsym...), θxmodel, TrainableSelector) # This is why we need the odd definition of the FBAlgorithm struct.
        vi = Sx.state.vi

        # This is only important becasue we add the TrainableSelector
        for sym in keys(vi.metadata)
            vns = getfield(vi.metadata, sym).vns
            for vn in vns
                DynamicPPL.updategid!(vi, vn, Sx)
            end
        end
    end

    # Do the same for the emission parameters.
    indxs = _getidcs_nonempty(viθy, ConstSelector)
    vsym = Symbol[]
    for key in keys(indxs)
        if length(indxs[key]) == 0
            push!(vsym, key)
        end
    end

    if length(vsym) == 0
        Sy = viθy
    else
        @assert Ay != Nothing "[HyperParamsInit] There must be an alogirhtm specified for Ay, since some parameters have non deterministic priors"
        Sy = Turing.Sampler(Ay(inf.θyargs..., vsym...), θymodel, TrainableSelector)
        vi = Sy.state.vi
        for sym in keys(vi.metadata)
            vns = getfield(vi.metadata, sym).vns
            for vn in vns
                DynamicPPL.updategid!(vi, vn, Sy)
            end
        end
    end
    return Sampler(inf, Sx, Sy, inputs, inf.nsteps)
end


function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::TuringHPSampler,
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    if typeof(spl.θxSampler) <: AbstractSampler
        AbstractMCMC.sample_init!(
        rng,
        model.θxmodel,
        spl.θxSampler,
        N;
        verbose=verbose,
        resume_from=resume_from,
        kwargs...
        )
    end
    if typeof(spl.θySampler) <: AbstractSampler
        AbstractMCMC.sample_init!(
        rng,
        model.θymodel,
        spl.θySampler,
        N;
        verbose=verbose,
        resume_from=resume_from,
        kwargs...
        )
    end
    return get_hyperparams(spl, model, spl.inputs)
end


function AbstractMCMC.step!(
    rng::AbstractRNG,
    mdl::AbstractModel,
    spl::TuringHPSampler,
    N::Integer,
    states::NT;
    kwargs...
) where {NT<:NamedTuple}
    if typeof(spl.θxSampler) <: AbstractSampler
        # This is a bit hacky, however, it works!
        # We reset the arguments form the model. This is only possible because
        # vecotrs and matric are mutuable altough the arguments are itself immutable.
        AbstractMCMC.sample_init!(rng,mdl.θxmodel,spl.θxSampler,spl.alg.nsteps+1) # Unfortunately, we have to make this....

        # we take the first one if we have multiple trajectories. This is because we have not defined it for sevarl trajectories yet
        mdl.θxmodel.args.X[:,:,:] = @view states[:X][:,:,1,:]
        for i =1:spl.nsteps
            t1 = AbstractMCMC.step!(rng, mdl.θxmodel, spl.θxSampler, N; kwargs...)
        end
    else
        t1 = ConstTransition(tonamedtuple(spl.θxSampler))
    end
    if typeof(spl.θySampler) <: AbstractSampler
        AbstractMCMC.sample_init!(rng,mdl.θymodel,spl.θySampler,5) # Unfortunately, we have to make this....
        # This is a bit hacky, however, it works!
        # We reset the arguments form the model. This is only possible because
        # vecotrs and matric are mutuable altough the arguments are itself immutable.
        # we take the first one if we have multiple trajectories. This is because we have not defined it for sevarl trajectories yet
        mdl.θxmodel.args.X[:,:,:] = @view states[:X][:,:,1,:]

        for i =1:spl.nsteps
            t2 = AbstractMCMC.step!(rng, mdl.θymodel, spl.θySampler, N; kwargs...)
        end
    else
        t2 = ConstTransition(tonamedtuple(spl.θySampler))
    end
    return PriorTransition{typeof(t1),typeof(t2)}(t1, t2), deepcopy(get_hyperparams(spl, mdl, spl.inputs))...
end


function get_hyperparams(spl::TuringHPSampler, hpm::AbstractModel, inputs::NamedTuple)
    vi_x = typeof(spl.θxSampler) <: AbstractSampler ? spl.θxSampler.state.vi : spl.θxSampler
    vi_y =  typeof(spl.θySampler) <: AbstractSampler ? spl.θySampler.state.vi : spl.θySampler
    return get_hyperparams_from_vi(vi_x, vi_y, hpm, inputs)
end
