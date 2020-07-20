###
# We have changed a lot because of the changes in AbstractMCMC
# This is not very nice and it should be changed at some point...
###



# We need to make a small hack around this
# In order to infer the basemodel for the latenstates, we need to have acces
# to the hyper paramters, which we have only after the initialisation
mutable struct GPSSMState
    mdl # We view the basemodel as state since it mainly contains hyper paramters...
    states
    hp
    additional_args
end

#The base sampler
struct GPSSMSampler{SLT, SHP} <: AbstractSampler where {SLT <: AbstractLSSampler, SHP <:Union{AbstractHPOptimiser, AbstractHPFBSampler}}
    slt::SLT
    shp::SHP
    state::GPSSMState
end

#The base Model
struct GPSSMModel{I,NT,S,M} <: AbstractModel where {I<:GPSSMInstance, NT<:NamedTuple, S<:AbstractGPSpec, M<:AbstractHPModel}
    instace::I
    inputs::NT
    spec::S
    mhp::M
end


# Well regarding type stability, this is not great...
GPSSMState() = GPSSMState(nothing, nothing, nothing, nothing)


function AbstractMCMC.sample(
    instance::GPSSMInstance,
    alghp::AbstractHPAlgorithm,
    pfalg::AbstractLSAlgorithm,
    inputs::NamedTuple,
    N::Integer;
    progress::Bool=true,
    kwargs...
)
    # For every specifiaction and algorithm there is a model and a corresponding sampler.
    spec = get_gpspec_from_instance(instance)
    return AbstractMCMC.sample(instance, spec, alghp, pfalg, inputs, N; progress=progress, kwargs...)
end





function AbstractMCMC.sample(
    instance::GPSSMInstance,
    spec::GPSpec,
    alghp::AbstractHPAlgorithm,
    pfalg::AbstractLSAlgorithm,
    inputs::NamedTuple,
    N::Integer;
    progress::Bool=true,
    kwargs...
)
    # Get the specific latentent states algorithm
    alglt = BaseLSAlgorithm(pfalg, alghp)

    # For every specifiaction and algorithm there is a model and a corresponding sampler.
    mhp = get_hp_model(spec, alghp, inputs, instance)
    shp = Sampler( mhp, alghp, inputs)
    slt = Sampler( alglt, inputs)

    model = GPSSMModel(instance, inputs, spec, mhp)
    sampler = GPSSMSampler(slt, shp, GPSSMState())
    return AbstractMCMC.sample(Random.GLOBAL_RNG, model, sampler, N; progress=progress, kwargs...)
end



function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::GPSSMModel,
    sampler::GPSSMSampler,
    N::Integer;
    kwargs...
)
    hp, additional_args = AbstractMCMC.sample_init!(rng, model.mhp, sampler.shp, N; kwargs...)
    # Attention, we obtain a new model mdl for the latent states. This was necessary for the a previous
    # verion and is therefore still existent.
    mdl, states =  AbstractMCMC.sample_init!(rng, sampler.slt, N, hp, additional_args; kwargs...)
    # Update the new model
    sampler.state.mdl = mdl
    sampler.state.states = states
    sampler.state.hp = hp
    sampler.state.additional_args = additional_args

end


function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::GPSSMModel,
    sampler::GPSSMSampler,
    N::Integer,
    transition = nothing;
    iteration,
    kwargs...
)

    timeshp= @elapsed transitionhp, hp, additional_args =  AbstractMCMC.step!(rng, model.mhp, sampler.shp, N, sampler.state.states; iteration=iteration, kwargs...)
    #Then the latent states
    sampler.state.hp = hp
    sampler.state.additional_args= additional_args
    timesls = @elapsed transitionlt, states   =  AbstractMCMC.step!(rng, sampler.state.mdl, sampler.slt, N, hp, additional_args;  iteration=iteration, kwargs...)
    sampler.state.states = states

    savestates = states[:X][:,:,1:1,:]
    return (times = [timeshp, timesls], transitionhp = transitionhp, transitionlt= transitionlt, save_args = merge(merge(tonamedtuple(hp), additional_args), (X = savestates,)))
end

function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    model::GPSSMModel,
    sampler::GPSSMSampler,
    N::Integer,
    transitions,
    ::Type{Any};
    kwargs...
)
    tshp =transition_type(sampler.shp)[]
    tslt = transition_type(sampler.slt)[]
    save_args = []
    times =[]
    for i = 1:length(transitions)
        ti = transitions[i]
        push!(tshp, ti.transitionhp)
        push!(tslt, ti.transitionlt)
        push!(times, ti.times)
        push!(save_args, ti.save_args)
    end
    return AbstractMCMC.bundle_samples(rng, model.mhp, sampler.shp, sampler.slt, N,tshp, tslt; kwargs...), save_args, times
end
