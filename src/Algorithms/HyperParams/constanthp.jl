"""
	ConstSampler

## Arguments
	- hyperparams::NamedTuple: 	Pre specified Hyper Parameters
	- inputs::NamedTuple:		The inputs of the sampler (obseravitons, controls, sizes)

# This sampler takes a NamedTuple of hyper parameters as inputs and fix them throughout the whole sampling procedure.
# No Hyper parameters are trained at all!

"""


struct ConstSampler <:AbstractGPSSMSampler # Just a container for a namedtuple
	hyperparams::HyperParams
	additional_args::NamedTuple
	inputs::NamedTuple
end


Sampler(mdl::AbstractHPModel, inf::ConstantHP, inputs::NamedTuple) = ConstSampler(inf.hyperparams, merge((trainedtransition = inf.trainedtrans, ), inf.additiononal_args), inputs)

AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::AbstractHPModel,
    spl::ConstSampler,
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
) = (spl.hyperparams, spl.additional_args)


AbstractMCMC.step!(
    rng::AbstractRNG,
    mdl::AbstractHPModel,
    spl::ConstSampler,
    N::Integer,
    states::NT;
	iteration,
    kwargs...
) where {NT<:NamedTuple}  =  ConstTransition(NamedTuple()), spl.hyperparams, spl.additional_args
